"""OpenAI GPT-Live（gpt-live-1）作为 S2S provider：WebSocket 直连平台（wss://api.openai.com/v1/live/sessions），平台 API key。

实验性（11-02 探针起）。工具走 Responses delegation（11-02 夜实测）：session.start 的 delegation.responses.tools 声明 func_handler 的工具，
后台模型调用函数时服务端发 response.event{event: response.output_item.done(function_call)}，这里经 ToolRunner 执行后发
response.item.create(function_call_output)，这一轮的调用全部交回且 response.completed 之后再发 response.create，后台据此给出结论、Live 说出来。
工具清单（HA 插件、MCP）在会话建立之后才注册齐：after_components_ready 和每次听到用户开口时比对，变了就 session.update。

与 Realtime 的协议差异（11-02 Tokyo 实测）：
  开场 session.start → session.started；上行 session.input_audio.append；
  下行 session.output_audio.delta 是**连续流**：模型不说话时也按实时节奏发全零 PCM，没有"回复开始 / 结束"事件；
  轮次和插话全由模型自己判（说"停一下"它就不再往下说），没有 response.cancel / truncate。
所以这里按能量把下行流切成"回复段"：段内的帧（含短停顿）照发，连续静音超过 end_silence_ms 算一段结束（等设备播完发 tts stop）；
段外的静音帧直接丢，不让设备一直处在播放状态（否则回声门控永远当作在播放）。
设备 abort（按键 / 固件侧插话）时丢掉未发的帧，并丢弃本段剩下的音频，直到模型下一次静音再开口。
用户这句的转写里出现停止词时（播放中、设备已经 abort、或刚停不久）进入"停下"：一直丢模型的声音，补一声缓存的"好"，直到用户再开口（见 _stop_on）。
上行：设备 16 kHz PCM → 残余回声门控（可关，echo_gate: false）→ 直接 16 kHz 送上游。
日志只记事件、时长、字数，不记转写正文（log_text: true 才记）；api_key 不打印。
"""
from __future__ import annotations

import asyncio
import audioop
import base64
import contextlib
import json
import os
import re
import time

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from core.handle.abortHandle import handleAbortMessage
from core.providers.s2s import ha_names
from core.providers.s2s.audio import Resampler
from core.providers.s2s.base import DeviceSpeaker, S2SProviderBase, UplinkTrace
from core.providers.s2s.echo_gate import EchoGate, load_vad
from core.providers.s2s.speech_gate import SpeechGate
from core.providers.s2s.tools import ToolRunner, to_realtime_tools

TAG = __name__

LIVE_RATE = 16000       # 上下行都用 16 kHz（Live 支持 16 k / 24 k）
WAKE_TRIGGER = "嘿，你好呀"
TTS_RATE = 24000        # /v1/audio/speech 的 pcm 是 24 kHz 16-bit 单声道
CONTEXT_CHUNK_CHARS = 400   # session.thinking.append 每条 content 上限 500 token，中文按 400 字切
_GREETING_CACHE: dict[tuple, bytes] = {}   # (voice, 文本, 设备采样率) → PCM，进程内共用：只有第一次唤醒要等 TTS
_HA_CACHE: dict = {"t": 0.0, "entries": []}  # GetLiveContext 解析出的设备表，进程内各连接共用（见 _load_ha_names）；只按一个 HA 缓存
# 喊停那句里不算"别的话"的词：口头的"一下 / 一点 / 下来"、称呼、客气话和语气词（见 _has_more），长的在前
_STOP_FILLERS = ("好不好", "可以吗", "一下", "一点", "下来", "好吗", "行吗", "好的", "谢谢", "那个", "先", "你", "我", "好",
                 "吧", "啊", "呀", "呢", "嗯", "哦", "噢", "喔", "唉", "诶", "欸", "哎", "嘛", "啦", "了", "吗")
_SENTENCE_END = re.compile(r"[。！？!?….]+")   # 判断告别时切出回复的最后一句


def _trim_silence(pcm: bytes, rate: int, rms: int = 200, pad_ms: int = 40) -> bytes:
    """裁掉首尾低于 rms 的 10 ms 帧，两头各留 pad_ms。"""
    step = rate // 100 * 2
    frames = [pcm[i:i + step] for i in range(0, len(pcm) - len(pcm) % 2, step)]
    loud = [i for i, f in enumerate(frames) if len(f) >= 2 and audioop.rms(f, 2) > rms]
    if not loud:
        return pcm
    pad = pad_ms // 10
    return b"".join(frames[max(0, loud[0] - pad):loud[-1] + 1 + pad])


class S2SProvider(S2SProviderBase):
    def __init__(self, config: dict, conn):
        super().__init__(config, conn)
        c = self.config
        self.api_key = str(c.get("api_key") or "")
        self.url = str(c.get("url") or "wss://api.openai.com/v1/live/sessions")
        self.model = str(c.get("model_name") or "gpt-live-1")
        self.voice = str(c.get("voice") or "marin")
        # Live 没有"让模型先开口"的入口（session.instructions.append 只对 delegation 有效，11-02 实测）：
        # 默认把设备 listen start 之前的音频（唤醒词那几十帧）也送给模型，让它自己听到"你好小智"再回应
        self.forward_before_listen = bool(c.get("forward_before_listen", True))
        # 唤醒应答：Live 没法让模型先开口（XVF3800 走"服务端收唤醒词音频"的路径，固件也不播提示音），
        # 所以用同一把 key 的 TTS 合成一句（默认"我在。"、同一个音色）缓存起来，唤醒时直接播。空串关掉
        self.wake_greeting_text = str(c.get("wake_greeting_text", "我在。") or "").strip()
        self.greeting_tts_model = str(c.get("greeting_tts_model") or "gpt-4o-mini-tts")
        self.tts_url = str(c.get("tts_url") or "https://api.openai.com/v1/audio/speech")
        self._greeting_task = None
        # 告别：Live 说完"再见"不会自己结束会话（11-03 R19：要等 45 秒空闲关闭）。用户这句里有告别词、
        # 模型这段回复又短又是告别，就在这段播完后关连接；工具（handle_exit_intent）置了 close_after_chat 也照办
        # 建成前攒下的设备音频要不要补送（core/utils/s2s.drain_early_audio）："你好小智，……"连着说时问题在里面
        self.wants_early_audio = bool(c.get("early_audio", True))
        self.close_on_goodbye = bool(c.get("close_on_goodbye", True))
        self.goodbye_words = tuple(c.get("goodbye_words") or ("再见", "拜拜", "回头见", "下次聊"))
        # 喊停：Live 听到"停一下"后会先把正在说的那句讲完（11-03 讲故事中 2.5～3.5 s 才停），设备 abort 之后也可能接着讲。
        # 用户这句的转写里一出现停止词就进"停下"（见 _stop_on）；设备还在播就先由服务端截断（同设备 abort）。
        # 转写跟不上说话：第一个转写片段在出声后 1.2～2.2 s 才到（11-03 实测），"停"字更晚，所以截断只是兜底，快的那条路仍是设备。
        # 不收"闭嘴"：11-03 讲故事中一声附和"对对"被转写成了"闭嘴"。收"听一下"：11-03 23:37 Mac 17 外放三次"停一下"
        # 有两次被转写成它，播放中说"听一下"本身也接近打断。"不要停""别停""要停"不算（"不"字最容易被转写丢：
        # 11-04 00:35"给我讲一个两分钟的故事，不要停"被转写成"什么故事?要停"）；上一段刚结束、模型不在播的那几秒里，
        # 单字"停"也不算，要"停一下""停下""停止"这样连着的才算（见 _stop_word_at）。stop_words: [] 关掉
        self.stop_words = tuple(c["stop_words"] if "stop_words" in c else
                                ("停", "等一下", "等等", "别说了", "暂停", "打住", "安静", "听一下"))
        self._pending_barge = None
        self.last_user_text = ""
        self._t_close_flag = None
        self.connect_timeout_s = float(c.get("connect_timeout_s", 4))
        self.idle_close_s = float(c.get("idle_close_s", 45))
        self.speech_rms = int(c.get("speech_rms", 30))          # 下行帧 RMS 超过它算"在说话"（模型静音时发的是全零）
        self.end_silence_ms = int(c.get("end_silence_ms", 700))  # 段内连续静音这么久算一段结束
        self.log_text = bool(c.get("log_text", False))
        # 工具：tools: false 关掉 delegation；后台模型默认 gpt-6-luna（官方示例）
        self.tools_enabled = bool(c.get("tools", True))
        self.delegation_model = str(c.get("delegation_model") or "gpt-6-luna")
        self.delegation_reasoning = str(c.get("delegation_reasoning") or "").strip()   # 后台推理强度（low / medium …），空 = 模型默认
        self.delegation_instructions = str(c.get("delegation_instructions") or
                                           "你是小智的后台，负责调用工具（家居控制、设备音量等）并给出一两句中文结论；不需要工具的问题直接简短回答。")
        self.tools = ToolRunner(conn, conn.logger, per_minute=int(c.get("tool_calls_per_minute", 12)),
                                allow=c.get("tools_allow") or None, log_args=self.log_text)
        # 委托方式：responses = Live 自己调后台 Responses 模型（带整段对话上下文，11-03 真人测工具答复 16 s）；
        # client = Live 只发 delegation.created，这里用小模型 + 最近几句话选工具、执行、session.commentary.append 交回，
        # Live 自己转述（API 探针：说完到说出答案 3.7 s）
        self.delegation_mode = str(c.get("delegation_mode") or "responses")
        self.tool_model = str(c.get("tool_model") or "gpt-4.1-mini")
        self.tool_url = str(c.get("tool_url") or "https://api.openai.com/v1/responses")
        self.ack_delay_ms = int(c.get("ack_delay_ms", 800))
        self.tool_rounds = int(c.get("tool_rounds", 3))
        self.direct_max_chars = int(c.get("tool_direct_max_chars", 200))   # 单个工具结果不超过这么长就直接交给 Live 转述，省一轮
        self.context_max_chars = int(c.get("tool_context_max_chars", 1200))  # 更长的结果总结后，原文作背景给 Live 的上限（0 关掉）
        # 选工具的第一步：model = tool_model 自己选；decisions = 先用 /v1/decisions（gpt-6-luna，只做多选一）挑工具，
        # 不要参数的直接调用、要参数的再让 tool_model 强制只填这一个工具的参数；拿不准（置信度 < decision_min_confidence）
        # 或判成"不需要工具"就退回 model 的做法。11-03 夜 72 题对比：两种都 72/72，decisions 选工具中位 0.21 s（mini 0.94 s）
        self.tool_picker = str(c.get("tool_picker") or "model")
        self.decision_url = str(c.get("decision_url") or "https://api.openai.com/v1/decisions")
        self.decision_model = str(c.get("decision_model") or "gpt-6-luna")
        self.decision_min_conf = float(c.get("decision_min_confidence", 0.6))
        # 智能设备列表以外的 HA 设备（风扇、窗帘……）：调度指令带上从 GetLiveContext 生成的名单，HA 意图调用执行前过地点守卫
        # （core/providers/s2s/ha_names.py）。只在 client 委托里用；ha_names: false 关掉，取不到 GetLiveContext 时照旧不带名单
        self.ha_names = bool(c.get("ha_names", True))
        self.ha_names_domains = tuple(c.get("ha_names_domains") or ha_names.DEFAULT_DOMAINS)
        self.ha_names_ttl_s = float(c.get("ha_names_ttl_s", 600))
        self.ha_names_wait_s = float(c.get("ha_names_wait_s", 15))   # 等工具处理器初始化完（HA MCP 连上）的上限
        self._ha_entries: list[dict] = []
        self._guard_mark = 0          # 上次委托结束时 history 的长度：地点守卫只看这之后用户说的话
        self._ha_kinds_asked: set[str] = set()   # 这次委托问"要控制哪一类"时，那一处能开关的设备
        self._ha_kinds_prev: set[str] = set()    # 上一次委托问的那一批（判断用户这句是不是在答）
        self._ha_asked: set[str] = set()   # 上一次委托回问时列的设备
        self._all_asked = False            # 上一次委托问过"全部"是哪一处、哪一类（见 _fan_out_all）
        self.history: list[tuple[str, str]] = []
        self._session_sig = None
        self._calls: dict[str, asyncio.Task] = {}   # call_id → 执行中的工具
        self._resp_completed = False                # 当前后台回复已 completed，等工具结果交齐再 response.create
        self._had_calls = False
        # 按设备追加到提示词末尾的一段话（登记表 s2s: 里写字符串即可），用来试附和 / 旁人这类轮次规则
        self.instructions_suffix = str(c.get("instructions_suffix") or "").strip()
        # 上行开口 / 收声日志（量插话和应答延迟用，onset_rms: 0 关掉）
        self.trace = UplinkTrace(conn.logger, TAG, "Live", int(c.get("onset_rms", 800)))
        self.resume_after_user_s = float(c.get("resume_after_user_ms", 300)) / 1000
        self.resume_max_wait_s = float(c.get("resume_max_wait_ms", 1500)) / 1000
        # 喊停后的"停下"（见 _stop_on）：停止词到的时候模型在播、设备已经 abort，或者上一段结束不到 stop_recent_ms，才算喊停
        self.stop_recent_s = float(c.get("stop_recent_ms", 3000)) / 1000
        # 停下后补的一声应答，同唤醒应答一样用 TTS 合成后缓存。空串关掉
        self.stop_ack_text = str(c.get("stop_ack_text", "好。") or "").strip()
        self._ack_task = None

        gate = None
        if c.get("echo_gate", True):
            gate = EchoGate(vad=load_vad(self._vad_model_path()),
                            lag_prior=float(c.get("gate_lag_prior_ms", 200)) / 1000,
                            lag_max=float(c.get("gate_lag_max_ms", 1200)) / 1000)
        self.gate = gate
        # 上行人声门（speech_gate: true 打开）：不是人声的帧换成静音再送模型，见 core/providers/s2s/speech_gate.py
        self.speech_gate = None
        if c.get("speech_gate", False):
            vad = load_vad(self._vad_model_path())
            if vad is None:
                conn.logger.bind(tag=TAG).warning("speech_gate 打开了但 silero 模型不在，人声门不生效")
            else:
                self.speech_gate = SpeechGate(vad, threshold=float(c.get("speech_gate_threshold", 0.5)),
                                              hold_ms=int(c.get("speech_gate_hold_ms", 600)),
                                              lookahead_ms=int(c.get("speech_gate_lookahead_ms", 120)))
        self.t_wake_trigger = None
        out_rate = getattr(conn, "sample_rate", LIVE_RATE)
        self.speaker = DeviceSpeaker(conn, out_rate, gate=gate)
        self.down_rs = Resampler(LIVE_RATE, out_rate) if out_rate != LIVE_RATE else None

        self.ws = None
        self.closing = False
        self.in_seg = False        # 正在把一段回复发给设备
        self.silence_ms = 0.0      # 段内连续静音时长
        self.suppress = False      # 设备 abort 后丢掉模型的音频，直到用户说完、模型也静音（见 _may_resume）
        self.t_abort = 0.0
        self._stop = None          # 喊停后的"停下"：{ack, onset_ok, text_ok, dropped_s}，见 _stop_on
        self.t_seg_end = -1e9      # 上一段结束的时刻（monotonic）
        self.in_text = []
        self.out_text = []
        self._idle_handle = None
        self.t_activity = time.monotonic()
        self.t_last_input_text = None
        self.t_seg = time.monotonic()
        self.stats = {"segments": 0, "barge_device": 0, "barge_transcript": 0, "up_frames": 0, "seg_after_input_ms": [],
                      "input_utts": 0}

    def _vad_model_path(self):
        try:
            selected = (self.conn.config.get("selected_module") or {}).get("VAD")
            model_dir = ((self.conn.config.get("VAD") or {}).get(selected) or {}).get("model_dir")
        except AttributeError:
            model_dir = None
        if not model_dir:
            return None
        return os.path.join(model_dir, "src", "silero_vad", "data", "silero_vad.onnx")

    def _instructions(self) -> str:
        instructions = (getattr(self.conn, "prompt", None) or (self.conn.config.get("prompt") if isinstance(getattr(self.conn, "config", None), dict) else None)
                        or self.config.get("instructions") or "")
        instructions = str(instructions)
        if self.instructions_suffix:
            instructions = f"{instructions}\n\n{self.instructions_suffix}"
        return instructions

    # ---------- 会话 ----------
    def _tool_list(self) -> list[dict]:
        fh = getattr(self.conn, "func_handler", None)
        funcs = []
        if fh is not None:
            try:
                funcs = fh.get_functions() or []
            except Exception:  # noqa: BLE001
                funcs = []
        return to_realtime_tools(funcs, self.tools.allow or None)   # Responses 的 function 工具与 Realtime 同为拍平格式

    def _mutable_session(self, update: bool = False) -> dict:
        """session.start 里的指令 + delegation；update=True 时只给 delegation——session.update 不接受 instructions
        （11-02 实测 unknown_parameter: 'session.instructions'，整条更新被拒），所以 Live 主模型的指令只在建会话时定，
        之后才挂上的 HA 设备清单等经后台模型的 instructions 带过去。"""
        instructions = self._instructions()
        out = {} if update else {"instructions": instructions}
        if self.tools_enabled and self.delegation_mode == "client":
            # client 模式工具清单在每次委托时现取，Live 这边没有要同步的
            self._session_sig = (instructions, tuple(sorted(t["name"] for t in self._tool_list())))
            if not update:
                out["delegation"] = {"type": "client", "acknowledgement_delay_ms": self.ack_delay_ms}
        elif self.tools_enabled:
            tools = self._tool_list()
            self._session_sig = (instructions, tuple(sorted(t["name"] for t in tools)))
            responses = {"model": self.delegation_model, "instructions": f"{instructions}\n\n{self.delegation_instructions}",
                         "tools": tools, "tool_choice": "auto"}
            if self.delegation_reasoning:
                responses["reasoning"] = {"effort": self.delegation_reasoning}
            out["delegation"] = {"type": "responses", "responses": responses}
        else:
            self._session_sig = (instructions, ())
        return out

    def after_components_ready(self) -> None:
        asyncio.run_coroutine_threadsafe(self._after_ready(), self.conn.loop)

    async def _after_ready(self) -> None:
        await self._maybe_refresh()
        from core.utils.s2s import drain_pending_texts
        await drain_pending_texts(self.conn)   # provider 建成前到的唤醒词 detect 现在补送（要播唤醒应答）
        if self.tools_enabled and self.delegation_mode == "client" and self.ha_names:
            self.spawn(self._load_ha_names())

    async def _load_ha_names(self) -> None:
        """只读调一次 HA 的 GetLiveContext，解析成名单 / 守卫用的设备表，进程内缓存 ha_names_ttl_s 秒（各连接共用）。
        取到之前、取不到、解析不出时，调度照旧不带名单。"""
        if not self.ha_names:
            return
        if _HA_CACHE["entries"] and time.monotonic() - _HA_CACHE["t"] < self.ha_names_ttl_s:
            self._ha_entries = _HA_CACHE["entries"]
            return
        fh = getattr(self.conn, "func_handler", None)
        # 工具处理器在后台初始化（connection._initialize_intent 用 run_coroutine_threadsafe 起，不等它），组件就绪时
        # HA 的 MCP 工具常常还没挂上（11-03 部署后探针：会话建立时 tools=0），所以等它 finish_init，最多 ha_names_wait_s 秒
        t_end = time.monotonic() + self.ha_names_wait_s
        while fh is not None and not getattr(fh, "finish_init", True) and time.monotonic() < t_end:
            await asyncio.sleep(0.2)
        name = next((t["name"] for t in self._tool_list() if t["name"].endswith("GetLiveContext")), None)
        if not name or fh is None:
            self.logger.bind(tag=TAG).info("Live HA 设备名单：工具清单里没有 GetLiveContext，调度不带名单")
            return
        t0 = time.time()
        try:
            text, _ = ToolRunner._to_text(await fh.handle_llm_function_call(self.conn, {"name": name, "arguments": "{}"}))
            entries = ha_names.parse_live_context(text)
        except Exception as e:  # noqa: BLE001
            self.logger.bind(tag=TAG).warning(f"Live 取 HA 设备名单失败 {type(e).__name__}: {e}")
            return
        if not entries:
            self.logger.bind(tag=TAG).warning(f"Live 取 HA 设备名单：{name} 没解析出设备（{len(text)} 字），调度不带名单")
            return
        _HA_CACHE.update(t=time.monotonic(), entries=entries)
        self._ha_entries = entries
        self.logger.bind(tag=TAG).info(f"Live HA 设备名单：{len(entries)} 个实体、{len(ha_names.area_table(entries))} 个区域，"
                                       f"{round(1000 * (time.time() - t0))} ms")

    async def _maybe_refresh(self) -> None:
        """提示词或工具清单变了就 session.update（HA 设备清单、MCP 工具都在会话建立之后才挂上）。"""
        tools = self._tool_list() if self.tools_enabled else []
        sig = (self._instructions(), tuple(sorted(t["name"] for t in tools)))
        if sig == self._session_sig or self.ws is None:
            return
        upd = self._mutable_session(update=True)
        if not upd:
            return
        await self._send({"type": "session.update", "session": upd})
        self.logger.bind(tag=TAG).info(f"Live 会话配置已同步：指令 {len(sig[0])} 字、工具 {len(sig[1])} 个")

    async def open(self) -> None:
        if not self.api_key or self.api_key.startswith("你的"):
            raise RuntimeError("api_key 未配置")

        async def _connect():
            ws = await connect(self.url, additional_headers={"Authorization": f"Bearer {self.api_key}"},
                               max_size=None, open_timeout=None, ping_interval=20, ping_timeout=20)
            try:
                session = {"model": self.model, **self._mutable_session(),
                           "audio": {"format": {"type": "audio/pcm", "rate": LIVE_RATE}, "output": {"voice": self.voice}}}
                await ws.send(json.dumps({"type": "session.start", "session": session}, ensure_ascii=False))
                while True:
                    ev = json.loads(await ws.recv())
                    if ev.get("type") == "session.started":
                        return ws
                    if "error" in str(ev.get("type")):
                        code = (ev.get("error") or {}).get("code")
                        raise RuntimeError(f"first event {ev.get('type')} {code or ''}".strip())
            except BaseException:
                self.spawn(ws.close())
                raise

        t0 = time.monotonic()
        self.ws = await asyncio.wait_for(_connect(), self.connect_timeout_s)
        if self.wake_greeting_text:
            self._greeting_task = self.spawn(self._greeting_pcm())
        if self.stop_ack_text and self.stop_words:
            self._ack_task = self.spawn(self._ack_pcm())
        self.spawn(self._rx_loop())
        self._arm_idle()
        self.logger.bind(tag=TAG).info(
            f"Live 会话已建立 model={self.model} voice={self.voice} echo_gate={'on' if self.gate else 'off'} "
            f"speech_gate={'on' if self.speech_gate else 'off'} "
            f"speech_rms={self.speech_rms} end_silence_ms={self.end_silence_ms} "
            f"tools={'off' if not self.tools_enabled else len(self._session_sig[1])}@"
            f"{self.tool_model + '(client)' if self.delegation_mode == 'client' else self.delegation_model} in {time.monotonic() - t0:.2f}s")

    async def _send(self, obj: dict) -> None:
        if self.ws is None:
            return
        with contextlib.suppress(ConnectionClosed):
            await self.ws.send(json.dumps(obj, ensure_ascii=False))

    async def close(self) -> None:
        self.closing = True
        if self._idle_handle is not None:
            self._idle_handle.cancel()
        await super().close()
        if self.ws is not None:
            with contextlib.suppress(Exception):
                await self._send({"type": "session.close"})
                await self.ws.close()
        if self.gate is not None:
            self.stats["gate"] = dict(self.gate.stats, lag_ms=round(self.gate.lag * 1000))
        if self.speech_gate is not None:
            self.stats["speech_gate"] = self.speech_gate.stats
        self.logger.bind(tag=TAG).info(f"Live 会话结束: {self.stats}")
        self.record_session(self.model, self.stats)

    # ---------- 设备 → 模型 ----------
    async def on_pcm(self, pcm16k: bytes) -> None:
        if self.closing or not pcm16k:
            return
        was_loud = self.trace.loud
        self.trace.feed(pcm16k, self.in_seg)
        if self._stop is not None and self.trace.loud and not was_loud and time.monotonic() >= self._stop["onset_ok"]:
            await self._end_stop("用户又开口")
        if self.gate is not None:
            pcm16k, _ = self.gate.process(pcm16k, time.monotonic(), self.speaker.speaking)
        if not self.forward_before_listen and not getattr(self.conn, "client_is_listening", False):
            return
        if self.speech_gate is not None:
            pcm16k = self.speech_gate.process(pcm16k, time.monotonic())
            if not pcm16k:
                return   # 前瞻队列还没填满
        await self._send({"type": "session.input_audio.append", "audio": base64.b64encode(pcm16k).decode()})
        self.stats["up_frames"] += 1

    async def on_text(self, text: str) -> None:
        # Live 没有"按文字起一轮回复"的入口：唤醒词那句只用来播缓存的唤醒应答，其余文字忽略
        text = (text or "").strip()
        if text == WAKE_TRIGGER:
            self.t_wake_trigger = time.monotonic()
            await self._play_greeting()
        elif text:
            self.logger.bind(tag=TAG).info("Live 暂不支持文字输入，已忽略")

    async def _greeting_pcm(self) -> bytes:
        return await self._tts_pcm(self.wake_greeting_text, "唤醒应答")

    async def _ack_pcm(self) -> bytes:
        return await self._tts_pcm(self.stop_ack_text, "停下应答")

    async def _tts_pcm(self, text: str, what: str) -> bytes:
        key = (self.voice, text, self.speaker.sample_rate)
        if key in _GREETING_CACHE:
            return _GREETING_CACHE[key]
        import aiohttp
        body = {"model": self.greeting_tts_model, "voice": self.voice, "input": text,
                "response_format": "pcm", "instructions": "自然、亲切、简短地应一声。"}
        # 后台预取，超时放宽到 30 s：11-04 Tokyo 实测同样的请求有时 19 s 才回（"好。"19.0 s、2.0 s，"我在。"1.1 s），
        # 6 s 超时会让重启后缓存还空的那批会话一直没有应答。播放那边最多只等 2.5 s，取得慢不耽误
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as s:
                async with s.post(self.tts_url, json=body, headers={"Authorization": f"Bearer {self.api_key}"}) as r:
                    if r.status != 200:
                        self.logger.bind(tag=TAG).warning(f"{what} TTS 失败 HTTP {r.status}")
                        return b""
                    pcm = await r.read()
        except Exception as e:  # noqa: BLE001
            self.logger.bind(tag=TAG).warning(f"{what} TTS 失败 {type(e).__name__}")
            return b""
        pcm = _trim_silence(pcm, TTS_RATE)   # TTS 首尾各带几百毫秒静音（11-02 实测整句 2.35 秒）
        if self.speaker.sample_rate != TTS_RATE:
            pcm = Resampler(TTS_RATE, self.speaker.sample_rate).process(pcm)
        _GREETING_CACHE[key] = pcm
        return pcm

    def _user_spoke_since_wake(self) -> bool:
        """唤醒后用户是不是接着在说话：只认模型转写出了话。
        不看电平：11-03 v9 复测，唤醒后一阵 13780 的非人声响声让"我在"被跳过，用户以为没唤醒。
        也不看人声门：设备连上后会把唤醒词音频再发一遍，人声门把它判成人声，"我在"同样被跳过（11-03 12:19）。"""
        t0 = self.t_wake_trigger
        if t0 is None:
            return False
        return self.t_last_input_text is not None and self.t_last_input_text >= t0

    async def _play_greeting(self) -> None:
        if not self.wake_greeting_text or self.in_seg or self.speaker.speaking:
            return
        try:
            pcm = await asyncio.wait_for(asyncio.shield(self._greeting_task), 2.5) if self._greeting_task else b""
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001
            pcm = b""
        if not pcm or self.in_seg or self.speaker.speaking:
            return
        await asyncio.sleep(0.3)   # 给攒下的音频一点时间补送进来（drain_early_audio 在下一个音频包时执行）
        if self._user_spoke_since_wake():
            self.logger.bind(tag=TAG).info("Live 唤醒应答跳过：用户唤醒后接着在说")
            return
        await self.speaker.begin()
        await self.speaker.push(pcm)
        self.spawn(self.speaker.end())
        self.logger.bind(tag=TAG).info(f"Live 唤醒应答「{self.wake_greeting_text}」{round(len(pcm) / 2 / self.speaker.sample_rate * 1000)} ms")

    # ---------- 打断 ----------
    async def cancel(self, reason: str) -> None:
        """来自 handleAbortMessage：设备 abort，或者 _barge（喊停兜底先调 handleAbortMessage，真正的原因放在 _pending_barge）。
        模型侧的插话 Live 自己处理，表现为下行变静音。"""
        if reason == "device" and self._pending_barge:
            reason, self._pending_barge = self._pending_barge, None
        played = self.speaker.played_ms()
        self.suppress = True
        self.t_abort = time.time()
        await self._end_segment(interrupted=True)
        self.stats[f"barge_{reason}"] = self.stats.get(f"barge_{reason}", 0) + 1
        self.logger.bind(tag=TAG).info(f"打断（{reason}）：本段已播 {played} ms t={time.time():.3f}")

    async def _barge(self, reason: str) -> None:
        self._pending_barge = reason
        await handleAbortMessage(self.conn)  # 发 tts stop、清队列，再经 abortHandle 调到 cancel()

    async def _stop_on(self, word: str, after: str) -> None:
        """用户这句里有停止词：不论设备是否已经 abort、是否在 suppress 期间，都不再按 _may_resume 放出模型接下来的话，
        一直丢模型的声音，直到用户再开口（上行开口，或新的一句转写）。11-03 T4：设备 1.37 s 判停后，模型没理会"别说了"，
        等用户说完又接着讲了约 20 s；转写截断之后，模型也常把被截那句讲完再说"好"（T1："那儿有答案。好。"）。
        模型自己的"好"也一起丢了，所以等用户说完补一声缓存的"好"（只补一次），让用户知道它听到了。
        只有设备 abort、转写里没有停止词（误停、打字）时，照旧按 _may_resume 接着讲。
        喊停后同一句里还接着问了别的（"等一下，一加一等于几"），认出来就结束停下（见 _has_more）。"""
        playing = (self.in_seg or self.speaker.speaking) and not self.suppress
        how = "播放中由服务端截断" if playing else ("设备已经停了" if self.suppress else "刚停")
        self.logger.bind(tag=TAG).info(f"Live 转写里有停止词「{word}」，{how}；停到用户再开口 t={time.time():.3f}")
        if playing:
            await self._barge("transcript")
        elif not self.suppress:
            self.suppress, self.t_abort = True, time.time()
        now = time.monotonic()
        # 进停下后 0.6 s 内的上行开口不算：转写截断时设备还要约 0.1 s 才停，残余回声再拖一两百毫秒；
        # 11-03 T1 讲故事中没人说话时也一直有"上行开口 playing=True"（回声过了 onset_rms）
        self._stop = {"ack": bool(self.stop_ack_text), "onset_ok": now + 0.6, "text_ok": now, "dropped_s": 0.0,
                      "after": after}
        self.stats["stop_holds"] = self.stats.get("stop_holds", 0) + 1
        if self._has_more(after):
            await self._end_stop("喊停后接着说了别的")

    def _has_more(self, after: str) -> bool:
        """喊停那句里，停止词后面还有没有别的话：去掉"小智"、停止词、语气词和标点，还剩 2 个字以上就算。
        有的话停下就结束，交回 _may_resume：等用户说完 0.3 s，模型的回答照常放（"停一下，现在几点"）。
        Live 没有"这句转写定稿"的事件，所以不等这一句结束再判。"""
        t = after.replace("小智", "")
        for w in sorted(self.stop_words, key=len, reverse=True) + list(_STOP_FILLERS):
            t = t.replace(w, "")
        return sum(1 for ch in t if ch.isalnum()) >= 2

    async def _end_stop(self, why: str) -> None:
        s, self._stop = self._stop, None
        self.out_text.clear()                 # 停下期间丢掉的话不显示、不进 history
        await self._flush_input_text()        # 喊停那句先记下；新的一句从头算，不带着旧的停止词
        self.logger.bind(tag=TAG).info(f"Live 停下结束（{why}）：丢了 {round(s['dropped_s'] * 1000)} ms 有声帧 t={time.time():.3f}")

    def _stop_frame(self, loud: bool, ms: float) -> None:
        """停下期间：模型的声音一律丢掉（静音帧也不往下走，_may_resume 不许把 suppress 清掉）；用户说完后补一声应答。"""
        s = self._stop
        if loud:
            s["dropped_s"] += ms / 1000
        tr = self.trace
        if s["ack"] and not self.speaker.speaking and not tr.loud and time.time() - tr.t_last >= self.resume_after_user_s:
            s["ack"] = False
            self.spawn(self._play_ack())

    async def _play_ack(self) -> None:
        try:
            pcm = await asyncio.wait_for(asyncio.shield(self._ack_task), 2.5) if self._ack_task else b""
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001
            pcm = b""
        if not pcm:
            if self._ack_task is None or self._ack_task.done():
                self._ack_task = self.spawn(self._ack_pcm())   # 上次没取到（TTS 慢或失败）：再取一次，下次喊停用
            return
        s = self._stop
        if s is None or self.in_seg or self.speaker.speaking:
            return
        dur = len(pcm) / 2 / self.speaker.sample_rate
        now = time.monotonic()
        s["onset_ok"] = now + dur + 0.5      # 这声"好"的回声不算用户开口
        s["text_ok"] = now + dur + 2.5       # 回声被转写出来还要一两秒
        await self.speaker.begin()
        await self.speaker.push(pcm)
        self.spawn(self.speaker.end())
        self.logger.bind(tag=TAG).info(f"Live 停下应答「{self.stop_ack_text}」{round(dur * 1000)} ms t={time.time():.3f}")

    def _stop_word_at(self, text: str, bare_ting: bool = True) -> tuple[str, int] | None:
        """用户这句里的停止词和它后面的位置；"不要停""别停""要停"这类否定的不算（"给我讲个故事，不要停"）。
        bare_ting=False（上一段刚结束、模型不在播）时单字"停"不算，后面要跟着"一下 / 下 / 止 / 停"。"""
        for w in self.stop_words:
            i = text.find(w)
            while i >= 0:
                if w != "停" or not (text[max(0, i - 2):i].endswith(("要", "别", "不", "没"))
                                     or not bare_ting and not text[i + 1:].startswith(("一下", "下", "止", "停"))):
                    return w, i + len(w)
                i = text.find(w, i + 1)
        return None

    # ---------- 模型 → 设备 ----------
    async def _rx_loop(self) -> None:
        try:
            async for raw in self.ws:
                try:
                    ev = json.loads(raw)
                except ValueError:
                    continue
                try:
                    await self._on_event(ev.get("type", ""), ev)
                except Exception as e:  # noqa: BLE001
                    self.logger.bind(tag=TAG).error(f"处理 Live 事件 {ev.get('type')} 出错: {type(e).__name__}: {e}")
        except ConnectionClosed:
            pass
        if not self.closing:
            self.closing = True
            self.logger.bind(tag=TAG).warning("Live 上游断开，关闭本连接")
            self.spawn(self.conn.close())

    async def _on_event(self, et: str, ev: dict) -> None:
        if et == "session.output_audio.delta":
            await self._on_audio(base64.b64decode(ev.get("delta") or ev.get("audio") or ""))
        elif et == "session.input_transcript.delta":
            now = time.monotonic()
            delta = ev.get("delta") or ""
            new_utt = self.t_last_input_text is None or now - self.t_last_input_text > 1.5
            if new_utt:
                # 一句话的第一个转写片段：探针用它对齐"用户开口 → 模型听到"（时间戳给毫秒，日志自带的只到秒）
                self.logger.bind(tag=TAG).info(f"Live 听到用户 t={time.time():.3f} playing={self.in_seg}")
                self.spawn(self._maybe_refresh())   # 用户开口到后台调工具之间还有一两秒，够把新注册的工具同步上去
                if self._stop is not None and now >= self._stop["text_ok"]:
                    await self._end_stop("用户新的一句")
            self.in_text.append(delta)
            self.t_last_input_text = now
            self.t_activity = now
            if self._stop is not None and not new_utt:       # 喊停那句还没说完
                self._stop["after"] += delta
                if self._has_more(self._stop["after"]):
                    await self._end_stop("喊停后接着说了别的")
            elif self.stop_words and self._stop is None:
                text = "".join(self.in_text)
                live = self.in_seg or self.suppress or self.speaker.speaking      # 在播，或者设备刚 abort
                if live or now - self.t_seg_end <= self.stop_recent_s:
                    hit = self._stop_word_at(text, bare_ting=live)
                    if hit:
                        await self._stop_on(hit[0], text[hit[1]:])
        elif et == "session.output_transcript.delta":
            self.out_text.append(ev.get("delta") or "")
        elif et == "response.event":
            await self._on_backend_event(ev.get("event") or {})
        elif et == "session.delegation.created":
            d = ev.get("delegation") or {}
            self.stats["delegations"] = self.stats.get("delegations", 0) + 1
            self.logger.bind(tag=TAG).info(f"Live 委托后台 target={d.get('target')} t={time.time():.3f}")
            if d.get("target") == "client" and d.get("id"):
                self.spawn(self._client_delegation(str(d["id"])))
        elif "error" in et:
            err = ev.get("error") or {}
            self.logger.bind(tag=TAG).warning(f"Live error type={err.get('type')} code={err.get('code')} {str(err.get('message') or '')[:120]}")
        elif et not in ("session.usage.updated", "session.updated"):
            self.logger.bind(tag=TAG).info(f"Live 事件 {et}")

    # ---------- 工具（Responses delegation） ----------
    async def _on_backend_event(self, inner: dict) -> None:
        it = inner.get("type", "")
        if it == "response.created":
            self._resp_completed, self._had_calls = False, False
        elif it == "response.output_item.done":
            item = inner.get("item") or {}
            if item.get("type") == "function_call" and item.get("call_id"):
                self._had_calls = True
                cid = item["call_id"]
                self._calls[cid] = self.spawn(self._run_tool(cid, item.get("name") or "", item.get("arguments") or "{}"))
        elif it == "response.completed":
            self._resp_completed = True
            await self._maybe_continue()
        elif it in ("response.failed", "response.incomplete", "error"):
            self.logger.bind(tag=TAG).warning(f"Live 后台 {it}: {json.dumps(inner, ensure_ascii=False)[:200]}")

    async def _run_tool(self, call_id: str, name: str, arguments) -> None:
        self.stats["tools"] = self.stats.get("tools", 0) + 1
        try:
            output = await self.tools.run(name, arguments)
            await self._send({"type": "response.item.create",
                              "item": {"type": "function_call_output", "call_id": call_id, "output": output}})
        finally:
            self._calls.pop(call_id, None)
        await self._maybe_continue()

    # ---------- 工具（client delegation：自己选工具） ----------
    def _dispatch_instructions(self) -> str:
        now = time.strftime("%Y-%m-%d %H:%M %A", time.localtime())
        # HA 意图工具的填法规则 + 智能设备列表以外的设备名单（decisions 和小模型都看；Live 主模型的指令里没有）
        ha = ha_names.dispatch_text(self._tool_list(), self._ha_entries, self.ha_names_domains) if self.ha_names else ""
        return (f"{self._instructions()}\n\n你现在是小智的工具调度：根据对话里用户最后一句话决定调用哪个工具。"
                "日期、农历、时间、天气、家里设备的状态和控制、本机音量这类实时信息必须调用工具，不要凭记忆回答。"
                # 11-03 夜对比：没有这一句时 gpt-4.1-mini 4/4 把"停一下"当成告别去调 handle_exit_intent
                "“停一下”“等一下”“别说了”只是让小智先停下，不是结束对话，不要调用 handle_exit_intent；只有用户明确告别（再见、拜拜、没事了）才结束对话。"
                f"拿到工具结果后用一两句中文说出结论；不需要工具就直接用一句话回答。当前时间：{now}" + (f"\n{ha}" if ha else ""))

    @staticmethod
    def _dispatch_tools(tools: list) -> list:
        """农历插件的说明叫"今天农历是多少"这类问题别调工具（原版级联的提示词里带当天农历），Live 链路里没有这份上下文，
        照说明做模型就会凭记忆答错，所以去掉那一句。HA 意图工具的 position 没有说明，补上方向（见下）。
        都只改副本，传进来的工具对象不动。"""
        out = []
        for t in tools:
            d = t.get("description") or ""
            if t.get("name") == "get_lunar" and "对于'今天农历是多少'" in d:
                t = dict(t, description=d.split("对于'今天农历是多少'")[0] + "今天的农历日期也用它查。")
            props = (t.get("parameters") or {}).get("properties") or {}
            pos = props.get("position")
            # HA 意图工具（MCP 里叫"平台__Hass意图"，如 intent__HassSetPosition）的 position 只有 0～100，没说哪头是开：
            # 11-03 空跑里 4 种"拉上"说法 16/16 被填成 100（HA 里 100 是全开），补上这句后 16/16 填 0。只补在参数说明里，
            # decisions 只看顶层说明，不受影响；写进调度指令（"拉上是关"）则会让模型给没有窗帘的那家抄出别家窗帘的别名，3 次里 2 次控错房子
            if (str(t.get("name") or "").rpartition("__")[2].startswith("Hass") and isinstance(pos, dict)
                    and not pos.get("description")):
                pos = dict(pos, description="位置百分比：0 是全关（拉上、合上），100 是全开（拉开）。")
                t = dict(t, parameters=dict(t["parameters"], properties=dict(props, position=pos)))
            out.append(t)
        return out

    async def _decisions_call(self, ctx: str, tools: list) -> tuple:
        """/v1/decisions 多选一：返回 (工具名或 None, 置信度)。"""
        import aiohttp
        choices = [{"value": t["name"], "description": (t.get("description") or "")[:300]} for t in tools]
        choices.append({"value": "none", "description": "不需要工具：闲聊、讲故事、讲笑话、常识问答"})
        body = {"model": self.decision_model, "input": f"{self._dispatch_instructions()}\n\n{ctx}",
                "questions": [{"type": "choice", "name": "tool", "choices": choices,
                               "instructions": "根据用户最后一句话，选出小智要调用的工具；不需要工具就选 none。"
                                               "日期、时间、农历、设备状态和控制、音量这类实时信息必须选工具。"}]}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=6)) as s:
            async with s.post(self.decision_url, json=body, headers={"Authorization": f"Bearer {self.api_key}"}) as r:
                if r.status != 200:
                    raise RuntimeError(f"HTTP {r.status}")
                a = ((await r.json()).get("answers") or [{}])[0]
        choice = a.get("choice")
        return (None if choice in (None, "none") else str(choice)), a.get("confidence")

    async def _responses_call(self, items: list, tools: list, force: str | None = None) -> dict:
        import aiohttp
        # 工具显式 strict: false。不写时 Responses 会尽量把 schema 规整成严格模式，可选参数全变必填，小模型只好硬填：
        # HA intent 的 floor 填空串、device_class / domain 猜成 switch，HA 按猜出来的条件匹配不上设备；强制填参和自选都这样（11-03 空跑）。
        # 级联走 chat completions，默认就是非严格。复制一份再加，传进来的工具对象 _decide_first 还在用
        tools = [dict(t, strict=False) for t in tools]
        body = {"model": self.tool_model, "instructions": self._dispatch_instructions(), "input": items,
                "tools": tools, "tool_choice": {"type": "function", "name": force} if force else "auto", "store": False}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
            async with s.post(self.tool_url, json=body, headers={"Authorization": f"Bearer {self.api_key}"}) as r:
                if r.status != 200:
                    raise RuntimeError(f"HTTP {r.status}")
                return await r.json()

    async def _decide_first(self, ctx: str, items: list, tools: list, trust_set_state: bool = False):
        """decisions 挑第一个工具；返回这一轮要执行的 function_call 列表，拿不准 / 不需要工具 / 出错时返回 None（退回 tool_model）。
        trust_set_state：选了 hass_set_state 就不看置信度（见 _client_delegation）。"""
        t0 = time.time()
        try:
            name, conf = await self._decisions_call(ctx, tools)
        except Exception as e:  # noqa: BLE001
            self.logger.bind(tag=TAG).warning(f"Live decisions 失败，退回 {self.tool_model}：{type(e).__name__}: {e}")
            return None
        by = {t["name"]: t for t in tools}
        ms = round(1000 * (time.time() - t0))
        low = conf is None or conf < self.decision_min_conf
        if not name or name not in by or (low and not (trust_set_state and name == "hass_set_state")):
            self.logger.bind(tag=TAG).info(f"Live decisions 选 {name}（置信度 {conf}）{ms} ms，退回 {self.tool_model}")
            return None
        self.logger.bind(tag=TAG).info(f"Live decisions 选 {name}（置信度 {conf}）{ms} ms")
        tool = by[name]
        if not (tool.get("parameters") or {}).get("properties"):
            return [{"type": "function_call", "call_id": f"dec_{int(t0 * 1000)}", "name": name, "arguments": "{}"}]
        resp = await self._responses_call(items, [tool], force=name)      # 只让小模型填这一个工具的参数
        calls = [o for o in (resp.get("output") or []) if o.get("type") == "function_call"]
        return calls or None

    def _check_ha_call(self, call: dict, intents: dict, utterance: str, recent: str | None = None,
                       asked: set[str] | frozenset[str] = frozenset()) -> str | None:
        """HA 意图调用执行前：domain 误填成 device_class 的挪过去（改写 call 里的参数，后面记进对话的也是改后的）；
        用户说的地点和会命中的设备不在一处、或者没说清是哪一台 / 哪一类时，返回交给 Live 的说明、不执行。
        utterance 是上次委托之后用户说的话，recent 是小模型手边的那几句（默认同 utterance），asked 是上一次委托回问时列的设备。"""
        name = call.get("name") or ""
        try:
            args = json.loads(call.get("arguments") or "{}")
        except ValueError:
            return None
        if not isinstance(args, dict):
            return None
        tidy = ha_names.tidy_args(intents[name], args, self._ha_entries)
        if tidy != args:
            self.logger.bind(tag=TAG).info(f"Live HA 参数整理 {name}：domain {args.get('domain')} → "
                                           f"domain {tidy.get('domain')} / device_class {tidy.get('device_class')}")
            call["arguments"] = json.dumps(tidy, ensure_ascii=False)
        blocked = ha_names.place_conflict(utterance, name, tidy, self._ha_entries)
        if blocked:
            self.logger.bind(tag=TAG).info(f"Live 地点守卫拦下 {name}：{blocked}")
            return blocked
        # 上一次问了"哪一类"，这次的调用还落在那一批里：这句在答那个问题（"窗帘"），上一句说过"都"，下面两道"哪一台"都不问
        switched = {e["names"][0] for e in ha_names.would_switch(name, tidy, self._ha_entries)}
        answering = bool(self._ha_kinds_prev) and switched <= self._ha_kinds_prev
        # 没填 name、会一次控制多台、用户又没说"都 / 全部"：先问是哪一台（设备名没识别出来时小模型会退成只填区域）
        blocked = None if answering else ha_names.area_ambiguity(utterance, name, tidy, self._ha_entries)
        if blocked:
            self.logger.bind(tag=TAG).info(f"Live 区域歧义拦下 {name}：{blocked}")
            return blocked
        # 用户说了"都"，小模型却只填了区域、没填类别：那一处能开关的全算（"把甲地的灯都关了"会连窗帘一起动）。先问要控制哪一类。
        # 问过了，这句又说了全部（"都关"），才算真要全部；答了"灯"小模型还是只填区域，接着问
        mixed = ha_names.mixed_kinds(name, tidy, self._ha_entries)
        if mixed and not (answering and ha_names.said_all_kinds(utterance)):
            self._ha_kinds_asked |= switched
            self._all_asked = True          # 答"灯"时小模型多半只填一盏，上一句说过"都"，按展开控制那一处的每一盏
            blocked = ha_names.ask_kind(mixed, tidy)
            self.logger.bind(tag=TAG).info(f"Live 类别歧义拦下 {name} {call.get('arguments')}：{blocked}")
            return blocked
        if answering:
            return None
        # 填了 name，但用户的话分不开这台和同类的另一台：先问。只放行一种情况：还是上一次问的那一批设备，用户最后这句话里也没有
        # 这些设备的词（"第一个""靠窗那幅"，名单里没有，靠小模型手边的对话上下文，再问就没完了）。答了半截（"客厅的"）时这一批
        # 会变小，接着问剩下的；带着设备的词却还是分不开（小模型选错了，或者上次问完没答、这次又说了一遍），也要问
        group = ha_names.name_rivals(utterance, utterance if recent is None else recent, name, tidy,
                                     self._ha_entries, self.ha_names_domains)
        if not group:
            return None
        names = {e["names"][0] for e in group}
        if names == asked and not ha_names.says_device_word(utterance.rsplit("\n", 1)[-1], group):
            return None
        self._ha_asked = names
        blocked = ha_names.ask_which(group)
        self.logger.bind(tag=TAG).info(f"Live 设备歧义拦下 {name} {call.get('arguments')}：{blocked}")
        return blocked

    def _fan_out_all(self, calls: list, said: str, recent: str, asked: bool) -> tuple[list, str | None]:
        """"把甲地的灯全关了"：强制填参只出一个 hass_set_state，结果又只有几个字，不展开就只控制到第一盏。动作用小模型填的，
        换成那一处同类设备每台一个调用（设备照设备列表，见 ha_names.all_at_place），结果交给小模型总结。不让小模型自己补：
        11-04 模拟里它有时漏灯还说都关了（位置写法不统一的那一处），给了提示就跨房子多控，给了名单又抄错 entity_id。
        没说地点（"把灯全关了"）时第二个返回值是交给 Live 的回问，这次委托的调用都不执行。said 是用户最后一句话，asked 是上一次委托问过。"""
        if any(c.get("name") != "hass_set_state" for c in calls) or not (asked or ha_names.says_all(said)):
            return calls, None
        try:
            args = json.loads(calls[0].get("arguments") or "{}")
        except ValueError:
            return calls, None
        if not isinstance(args, dict):
            return calls, None
        targets, question = ha_names.all_at_place(self.conn.prompt, said, recent, str(args.get("entity_id") or ""),
                                                  self._ha_entries, asked)
        if question:
            self._all_asked = True
            self.logger.bind(tag=TAG).info(f"Live 全部回问 hass_set_state：{question}")
            return calls, question
        if not targets or len(calls) > 1:
            return calls, None
        self.logger.bind(tag=TAG).info(f"Live 全部展开 hass_set_state：{len(targets)} 个设备")
        return [dict(calls[0], call_id=f"{calls[0].get('call_id')}_{n}",
                     arguments=json.dumps(dict(args, entity_id=e), ensure_ascii=False)) for n, e in enumerate(targets)], None

    async def _client_delegation(self, did: str) -> None:
        """Live 把一件事委托出来（没有请求正文）：用最近几句对话拼出请求，小模型选工具、ToolRunner 执行，结果交回 Live 转述。"""
        t0 = time.time()
        pending = "".join(self.in_text).strip()
        turns = self.history[-6:] + ([("user", pending)] if pending else [])
        ctx = "\n".join(f"{'用户' if r == 'user' else '小智'}：{t}" for r, t in turns) or self.last_user_text
        items: list = [{"role": "user", "content": ctx}]
        tools = self._dispatch_tools(self._tool_list())
        # 地点守卫看的是上次委托之后用户说的话：还没落进 history 的转写，加上这期间落进 history 的用户发言
        # （Live 先开口说一句时会把转写刷进 history）。更早的不看，否则上一轮提到的地方会误拦这一轮；
        # 多看到的地点只会让守卫放行，一句都看不到（转写还没到）就不拦
        said = [t for r, t in self.history[self._guard_mark:] if r == "user"] + ([pending] if pending else [])
        # 设备歧义看小模型手边的这几句：刚问过"甲地客厅的纱帘开着吗"，接着说"把它关了"不用再问；回问之后用户答半截也接得上
        recent = "\n".join(t for r, t in turns if r == "user")
        self._ha_kinds_prev, self._ha_kinds_asked = self._ha_kinds_asked, set()
        asked, self._ha_asked = self._ha_asked, set()
        asked_all, self._all_asked = self._all_asked, False
        intents = ha_names.device_intents(tools) if self.ha_names and self._ha_entries else {}
        answer, used, raw, ask = "", [], [], None
        # 只看最后一句：上一次委托时还没落进 history 的那句也会算进 said，它的"全关"不是这句的意思
        last = said[-1] if said else ""
        try:
            # 会展开成一处的灯的句子，decisions 选了 hass_set_state 就信它：展开只用小模型填的动作。置信度常不到 0.6（只答"乙地"时 0.33～0.41），
            # 退回小模型自选会只填区域，HA 按区域控制的不只是灯（11-04 模拟："把乙地的灯全打开"填成 HassTurnOn {area}）
            sure = ((asked_all or ha_names.says_all(last))
                    and bool(ha_names.all_at_place(self.conn.prompt, last, recent, "light.", self._ha_entries, asked_all)[0]))
            first = await self._decide_first(ctx, items, tools, sure) if self.tool_picker == "decisions" else None
            for i in range(max(1, self.tool_rounds)):
                if i == 0 and first:
                    calls, out = first, []
                else:
                    resp = await self._responses_call(items, tools)
                    out = resp.get("output") or []
                    calls = [o for o in out if o.get("type") == "function_call"]
                text = "".join(c.get("text", "") for o in out if o.get("type") == "message"
                               for c in (o.get("content") or []) if c.get("type") == "output_text").strip()
                if not calls:
                    answer = text
                    break
                if i == 0:
                    calls, ask = self._fan_out_all(calls, last, recent, asked_all)
                outputs = []
                for call in calls:
                    used.append(call.get("name"))
                    blocked = ask or (self._check_ha_call(call, intents, "\n".join(said), recent, asked)
                                      if call.get("name") in intents else None)
                    result = blocked or await self.tools.run(call.get("name") or "", call.get("arguments") or "{}")
                    outputs.append(result)
                    items.append({"type": "function_call", "call_id": call.get("call_id"), "name": call.get("name"),
                                  "arguments": call.get("arguments") or "{}"})
                    items.append({"type": "function_call_output", "call_id": call.get("call_id"), "output": result})
                if len(calls) == 1 and len(outputs[0]) <= self.direct_max_chars:
                    # 短结果直接交给 Live 转述，省一轮小模型。带上调用了什么、参数是什么：
                    # 设音量这类工具只回"成功"，光给结果 Live 只会说"好了"（11-03 R26），带上参数它会说"调到六十了"
                    call = calls[0]
                    answer = f"已调用 {call.get('name')}（参数 {call.get('arguments') or '{}'}），结果：{outputs[0]}"
                    break
                # 长结果交给小模型总结成一两句，总结常会删掉细节（农历插件自己写着"仅在要求查询宜忌时才返回宜忌"），
                # 用户接着问宜忌时 Live 手边没有真数据，就凭记忆编（11-03 直连 API：10 次接话 2 次编）；原文另作背景交给 Live
                raw += [o for o in outputs if len(o) > self.direct_max_chars]
            else:
                answer = answer or "这个我暂时没查到。"
        except Exception as e:  # noqa: BLE001
            self.logger.bind(tag=TAG).warning(f"Live 工具调度失败 {type(e).__name__}: {e}")
            answer = "这个我暂时查不到，稍后再试。"
        # 背景用 session.thinking.append：只当上下文、不要求念出来；每条 content 上限 500 token，按 400 字切，总长有上限
        context = "\n".join(raw)[:self.context_max_chars]
        for i in range(0, len(context), CONTEXT_CHUNK_CHARS):
            await self._send({"type": "session.thinking.append", "delegation_id": did,
                              "content": context[i:i + CONTEXT_CHUNK_CHARS]})
        self._guard_mark = len(self.history)
        await self._send({"type": "session.commentary.append", "delegation_id": did, "content": (answer or "好的。")[:1500]})
        self.logger.bind(tag=TAG).info(f"Live 委托交回 tools={used} {round(1000 * (time.time() - t0))} ms 答复 {len(answer)} 字"
                                       f"{f'，背景 {len(context)} 字' if context else ''} t={time.time():.3f}")

    async def _maybe_continue(self) -> None:
        """这一轮后台回复里的函数调用全部交回、且回复已 completed，才让后台继续（否则报 function_call_outputs_required）。"""
        if self._had_calls and self._resp_completed and not self._calls:
            self._had_calls = self._resp_completed = False
            await self._send({"type": "response.create"})

    async def _on_audio(self, pcm: bytes) -> None:
        if not pcm:
            return
        ms = len(pcm) / 2 / LIVE_RATE * 1000
        loud = audioop.rms(pcm, 2) > self.speech_rms
        if self._stop is not None:
            self._stop_frame(loud, ms)
            return
        if not loud:
            if self.suppress and self._may_resume():
                self.suppress = False  # 用户说完了、模型也停了：下一次开口是对用户的回应
            if not self.in_seg:
                return
            self.silence_ms += ms
            if self.silence_ms >= self.end_silence_ms:
                await self._end_segment()
                return
        else:
            if self.suppress:
                return
            self.silence_ms = 0.0
            self.t_activity = time.monotonic()
            if not self.in_seg:
                await self._begin_segment()
        if self.down_rs is not None:
            pcm = self.down_rs.process(pcm)
        await self.speaker.push(pcm)

    def _may_resume(self) -> bool:
        """设备打断之后什么时候恢复播放。11-02 R3：打断 0.1 秒后模型还在接着数（"十七，十八，好"），
        遇到第一个静音帧就恢复会把这几个数在用户还没说完时放出来。所以要等用户说完（上行收声 resume_after_user_s 秒），
        打断后一直没听到用户开口（误打断）就等 resume_max_wait_s 秒。"""
        now = time.time()
        tr = self.trace
        if tr.t_last >= self.t_abort - 0.5:          # 打断前后用户在说话
            return not tr.loud and now - tr.t_last >= self.resume_after_user_s
        return now - self.t_abort >= self.resume_max_wait_s

    async def _begin_segment(self) -> None:
        self.in_seg = True
        self.t_seg = time.monotonic()
        self.stats["segments"] += 1
        self.logger.bind(tag=TAG).info(f"Live 段开始 #{self.stats['segments']} t={time.time():.3f}")
        await self._flush_input_text()
        if self.t_last_input_text is not None:
            self.stats["seg_after_input_ms"].append(round(1000 * (time.monotonic() - self.t_last_input_text)))
        await self.speaker.begin()

    async def _end_segment(self, interrupted: bool = False) -> None:
        if not self.in_seg:
            return
        self.in_seg = False
        self.t_seg_end = time.monotonic()
        tail = self.silence_ms
        self.silence_ms = 0.0
        self.logger.bind(tag=TAG).info(
            f"Live 段结束 #{self.stats['segments']} t={time.time():.3f} dur={round(1000 * (time.monotonic() - self.t_seg) - tail)}ms"
            f"{' interrupted' if interrupted else ''}")
        txt = "".join(self.out_text).strip()
        self.out_text.clear()
        if self.log_text and txt:
            self.logger.bind(tag=TAG).info(f"assistant: {txt}")
        if txt:
            await self.show_assistant_text(txt)
            self.history.append(("assistant", txt))
        if not interrupted and self._is_goodbye(txt):
            self.logger.bind(tag=TAG).info("Live 告别：这段播完后结束会话")
            self.conn.close_after_chat = True    # speaker.end() 等设备播完、发 tts stop 后关连接
        if interrupted:
            await self.speaker.interrupt()
        else:
            self.spawn(self.speaker.end())
        await self._flush_input_text()

    def _is_goodbye(self, assistant_txt: str) -> bool:
        if not self.close_on_goodbye or not assistant_txt:
            return False
        said = (self.last_user_text + "".join(self.in_text))
        if not any(w in said for w in self.goodbye_words):
            return False
        # 11-04 真机：用户在模型念长回复时说拜拜，模型的告别会和长回复并成一段（"…还要查别的吗?拜拜!"）。
        # 整段超过 16 字就只看最后一句，否则要等空闲关闭，这期间喊唤醒词会被当成普通的话
        farewell = assistant_txt
        if len(farewell) > 16:
            farewell = ([s for s in _SENTENCE_END.split(farewell) if s.strip()] or [""])[-1]
        return len(farewell) <= 16 and any(w in farewell for w in self.goodbye_words)

    async def _flush_input_text(self) -> None:
        txt = "".join(self.in_text).strip()
        self.in_text.clear()
        if not txt:
            return
        self.last_user_text = txt
        self.history.append(("user", txt))
        self.stats["input_utts"] += 1
        if self.log_text:
            self.logger.bind(tag=TAG).info(f"user: {txt}")
        await self.show_user_text(txt)

    # ---------- 空闲关闭 ----------
    def _arm_idle(self) -> None:
        if self.idle_close_s <= 0 or self.closing:
            return
        self._idle_handle = self.conn.loop.call_later(1.0, self._idle_tick)

    def _idle_tick(self) -> None:
        if self.closing:
            return
        # 工具（handle_exit_intent）置了"说完就结束"，而模型没有再出声（没有段结束去触发 speaker.end 里的关闭）
        if getattr(self.conn, "close_after_chat", False) and not self.speaker.speaking and not self.in_seg:
            now = time.monotonic()
            self._t_close_flag = self._t_close_flag or now
            if now - self._t_close_flag > 2.0:
                self.logger.bind(tag=TAG).info("Live 结束会话（close_after_chat）")
                self.closing = True
                self.spawn(self.conn.close())
                return
        else:
            self._t_close_flag = None
        if not self.speaker.speaking and not self.in_seg and time.monotonic() - self.t_activity > self.idle_close_s:
            self.logger.bind(tag=TAG).info(f"空闲超过 {self.idle_close_s:.0f} 秒，关闭连接")
            self.closing = True
            self.spawn(self.conn.close())
            return
        self._arm_idle()
