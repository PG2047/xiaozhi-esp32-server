"""OpenAI Realtime 作为 S2S provider：WebSocket 直连平台（wss://api.openai.com/v1/realtime），平台 API key。

会话：握手时先建上游会话、等 session.created、发 session.update（指令 = 连接的系统提示词，工具 = func_handler 的清单）；
      原版把提示词增强、工具处理器建好后（after_components_ready）再同步一次。
上行：设备 16 kHz PCM → 残余回声门控（可关）→ 16 k → 24 k → input_audio_buffer.append；模型侧 server_vad 判轮次。
下行：response.output_audio.delta（24 kHz PCM）→ 60 ms Opus → 上游节拍器 → 设备；转写用 tts sentence_start / stt 消息给设备显示，
      并写进 conn.dialogue 供记忆使用。
打断：模型侧 speech_started、本地门控最近 W 帧里 ≥ N 帧新判出人声（保持期的帧不算）、设备 abort 三条路都汇到 handleAbortMessage → cancel()：
      丢掉未发的帧、response.cancel（模型侧判的不用，它自己已取消）、conversation.item.truncate 截到设备实际播到的位置。
工具：response.output_item.done 里的 function_call → core/providers/s2s/tools.py → function_call_output → response.create。
结束：模型最后一次真的出声回答之后 idle_close_s 秒没有新回答就关连接（设备回待机）；上游中途断开也关连接，
      设备重连时若上游仍不可用，attach_s2s 会让它回退到原版级联。
日志只记事件、时长、字数，不记转写正文（log_text: true 才记）；api_key 不打印。
"""
from __future__ import annotations

import asyncio
import base64
import collections
import contextlib
import json
import os
import re
import time

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from core.handle.abortHandle import handleAbortMessage
from core.providers.s2s.audio import Resampler
from core.providers.s2s.base import DeviceSpeaker, S2SProviderBase, UplinkTrace
from core.providers.s2s.echo_gate import EchoGate, load_vad
from core.providers.s2s.tools import ToolRunner, to_realtime_tools

TAG = __name__

RT_RATE = 24000         # Realtime 的 audio/pcm 只有 24 kHz
DEVICE_RATE = 16000     # 设备上行
WAKE_TRIGGER = "嘿，你好呀"   # 上游 listenMessageHandler 把唤醒词统一换成这句再进 startToChat
RESPONSE_START_WAIT_S = 10.0  # 我们发了 response.create 之后，最多等这么久算"回答进行中"（空闲关闭不打断它）
SPEECH_GRACE_S = 3.0          # 刚说完这么久内不按空闲关闭：留给模型起回复
_PUNCT = re.compile(r"[\s，。,.!?！？、；;：:\"'“”‘’…~～\-]+")


def _norm_text(s: str) -> str:
    """去掉空白和标点，用来比较转写是不是只剩提示词。"""
    return _PUNCT.sub("", s or "")
BENIGN_ERRORS = ("response_cancel_not_active", "conversation_already_has_active_response")
LAG_WINDOW_S = 0.25            # 锁定滞后之前，参考窗只在先验 ±这么多秒内找回声（窗越宽门槛越高：取的是窗内最大功率）
_LAST_LAG: dict[str, float] = {}   # device-id → 上一段会话锁定的滞后（秒）：同一台设备的滞后很稳，下次会话直接当先验
# 用记忆值当先验的会话，估了这么多次滞后都没确认，就在会话结束时清掉记忆，下一段回到配置先验。11-02 XVF3800：一个坏值
# （落在搜索上限的 1420～1500 ms）连续 3 段会话没被确认却一直当先验，真实约 1 秒的候选全因"离记忆值远"被拒
LAG_FORGET_AFTER = 3


class S2SProvider(S2SProviderBase):
    def __init__(self, config: dict, conn):
        super().__init__(config, conn)
        c = self.config
        self.api_key = str(c.get("api_key") or "")
        self.url = str(c.get("url") or "wss://api.openai.com/v1/realtime")
        self.model = str(c.get("model_name") or "gpt-realtime")
        self.voice = str(c.get("voice") or "marin")
        self.transcribe_model = str(c.get("transcribe_model") or "gpt-4o-mini-transcribe")
        # 转写提示：language 填 ISO-639-1（如 zh），prompt 填专名（如唤醒词"小智"）。只影响转写文本（设备显示、对话历史），
        # 不影响模型听到的音频。11-02 真人测试：没填语言时噪声会被转成韩语 / 日语短词，"小智"常被写成"乔治""在这"
        self.transcribe_language = str(c.get("transcribe_language") or "").strip()
        self.transcribe_prompt = str(c.get("transcribe_prompt") or "").strip()
        # 转写只剩提示词本身（例如整句只有"小智"）按空转写处理。11-02 加了 prompt=小智 之后，静场里的底噪（设备输入 40～150 rms）
        # 4 次被转写成"小智"：不是空串，空转写兜底不删，模型当成用户在叫它就开口，长故事讲完一截又自己接着讲。
        # 代价：用户在会话里整句只喊一声"小智"也会被当成空转写，模型不再回"我在"
        self.prompt_only_as_empty = bool(c.get("prompt_only_as_empty", True))
        self._prompt_tokens = {t for t in (_norm_text(x) for x in re.split(r"[，,、;；\s]+", self.transcribe_prompt)) if t}
        if self.transcribe_prompt:
            self._prompt_tokens.add(_norm_text(self.transcribe_prompt))
        self.turn_detection_type = str(c.get("turn_detection") or "server_vad")
        self.vad_threshold = float(c.get("vad_threshold", 0.5))
        self.vad_silence_ms = int(c.get("vad_silence_ms", 400))
        self.vad_eagerness = str(c.get("vad_eagerness") or "auto")
        self.connect_timeout_s = float(c.get("connect_timeout_s", 4))
        self.idle_close_s = float(c.get("idle_close_s", 45))
        self.wake_greeting = c.get("wake_greeting", "用户刚叫醒你，只用两三个字回应，比如“我在”。")
        self.local_barge_frames = int(c.get("local_barge_frames", 3)) if c.get("local_barge", True) else 0
        # 滑窗：最近 W 帧里 ≥ N 帧新判出人声就本地停播。原来要求 N 帧连续、一帧回声判决就清零——11-01 真人测试里一段 13 帧人声
        # 被几帧回声判决打散，没触发（silero 对双讲时被 DSP 处理过的人声时认时不认）
        self.local_barge_window = max(self.local_barge_frames, int(c.get("local_barge_window", 2 * self.local_barge_frames)))
        self._barge_marks = collections.deque(maxlen=max(1, self.local_barge_window))
        # 空转写兜底只取消"短开口"触发的回复：假开口（噪声、尾音、AGC 回升）通常不到 1 秒；真提问被设备侧压碎成空转写时开口有
        # 2～4 秒，这时取消回复用户听到的是沉默，不如让模型回"没听清"（11-01 15:27 真人测试：提问被 XMOS 双讲压碎后答案被吞掉）
        self.empty_cancel_max_s = float(c.get("empty_cancel_max_s", 1.5))
        self._turn_ms: collections.OrderedDict[str, list] = collections.OrderedDict()   # item_id → [audio_start_ms, audio_end_ms]
        self.log_text = bool(c.get("log_text", False))
        # 按设备追加到提示词末尾的一段话（登记表 s2s: 里写字符串，与 Live 同名同义），用来试附和 / 旁人这类轮次规则
        self.instructions_suffix = str(c.get("instructions_suffix") or "").strip()
        # 模型侧输入降噪：near_field（近讲）/ far_field（远场）；其它值忽略
        self.noise_reduction = str(c.get("noise_reduction") or "").strip()
        if self.noise_reduction and self.noise_reduction not in ("near_field", "far_field"):
            self.logger.bind(tag=TAG).warning(f"noise_reduction={self.noise_reduction} 不是 near_field / far_field，忽略")
            self.noise_reduction = ""

        gate = None
        self.lag_max_ms = int(c.get("gate_lag_max_ms", 1200))
        self.device_id = str(getattr(conn, "device_id", None) or "")
        remembered = _LAST_LAG.get(self.device_id) if self.device_id else None
        self.lag_prior_ms = int(round(remembered * 1000)) if remembered else int(c.get("gate_lag_prior_ms", 200))
        self.lag_prior_remembered = remembered is not None
        if c.get("echo_gate", True):
            # 锁定滞后之前，参考窗在先验 ±LAG_WINDOW_S 内找回声。11-01 曾把窗开到 80 ms～上限全程：窗内取最大功率，
            # 门槛高到音量 25 的插话都过不去；现在靠"记住上一段会话的滞后"把先验放准，窗就可以窄
            prior = self.lag_prior_ms / 1000
            gate = EchoGate(vad=load_vad(self._vad_model_path()),
                            lag_prior=prior, lag_max=self.lag_max_ms / 1000, g_mult=float(c.get("gate_g_mult", 8.0)),
                            near_prelock=LAG_WINDOW_S,
                            start_window=(max(0.08, prior - LAG_WINDOW_S), min(self.lag_max_ms / 1000, prior + LAG_WINDOW_S)))
        self.gate = gate
        self.speaker = DeviceSpeaker(conn, getattr(conn, "sample_rate", RT_RATE), gate=gate)
        self.tools = ToolRunner(conn, conn.logger, per_minute=int(c.get("tool_calls_per_minute", 12)),
                                allow=c.get("tools_allow") or None)
        self.up_rs = Resampler(DEVICE_RATE, RT_RATE)
        self.trace = UplinkTrace(conn.logger, TAG, "Realtime", int(c.get("onset_rms", 800)))  # 上行开口 / 收声日志，0 关掉

        self.ws = None
        self.rx_task = None
        self.closing = False
        self.resp_done = True
        self.suppress = False          # 被打断后丢掉本条回复剩下的音频，直到下一条 response.created
        self.item_id = None            # 当前助手消息（截断用）
        self.want_response = False     # 工具结果已交回、还没能发 response.create（当时有回复在进行）
        self.in_speech = False
        self.barge_run = 0
        self._pending_barge = None
        self._interrupted = None       # 本条回复已被哪条路打断（server / local / device / empty）；response.created 时清
        self._tool_tasks: set[asyncio.Task] = set()
        self._session_sig = None   # (指令全文, 工具名)：两者任一变了就重新 session.update
        self._idle_handle = None
        self.t_reply = time.monotonic()
        self.t_speech_stopped = None
        self.t_response_requested = None
        self.stats = {"turns": 0, "cancelled": 0, "speech_started": 0, "barge_server": 0, "barge_local": 0,
                      "barge_device": 0, "up_frames": 0, "tools": 0, "first_audio_ms": [],
                      "empty_transcripts": 0, "empty_cancelled": 0, "empty_long": 0, "barge_dup": 0,
                      "prompt_only": 0}

    # ---------- 会话 ----------
    def _vad_model_path(self):
        try:
            selected = (self.conn.config.get("selected_module") or {}).get("VAD")
            model_dir = ((self.conn.config.get("VAD") or {}).get(selected) or {}).get("model_dir")
        except AttributeError:
            model_dir = None
        if not model_dir:
            return None
        return os.path.join(model_dir, "src", "silero_vad", "data", "silero_vad.onnx")

    def _turn_detection(self) -> dict:
        both = {"create_response": True, "interrupt_response": True}
        if self.turn_detection_type == "semantic_vad":
            return {"type": "semantic_vad", "eagerness": self.vad_eagerness, **both}
        return {"type": "server_vad", "threshold": self.vad_threshold, "prefix_padding_ms": 300,
                "silence_duration_ms": self.vad_silence_ms, **both}

    def _tool_list(self) -> list[dict]:
        fh = getattr(self.conn, "func_handler", None)
        funcs = []
        if fh is not None:
            try:
                funcs = fh.get_functions() or []
            except Exception:  # noqa: BLE001
                funcs = []
        return to_realtime_tools(funcs, self.tools.allow or None)

    def _instructions(self) -> str:
        # 建成那一刻 conn.prompt 往往还是 None（快速提示词在之后才设置）：先用配置里的原始 prompt，别让第一轮没有人设
        instructions = (getattr(self.conn, "prompt", None) or (self.conn.config.get("prompt") if isinstance(getattr(self.conn, "config", None), dict) else None)
                        or self.config.get("instructions") or "")
        instructions = str(instructions)
        if self.instructions_suffix:
            instructions = f"{instructions}\n\n{self.instructions_suffix}"
        return instructions

    def _session(self) -> dict:
        tools = self._tool_list()
        instructions = self._instructions()
        self._session_sig = (instructions, tuple(sorted(t["name"] for t in tools)))
        transcription = {"model": self.transcribe_model}
        if self.transcribe_language:
            transcription["language"] = self.transcribe_language
        if self.transcribe_prompt:
            transcription["prompt"] = self.transcribe_prompt
        audio_in = {"format": {"type": "audio/pcm", "rate": RT_RATE},
                    "transcription": transcription,
                    "turn_detection": self._turn_detection()}
        if self.noise_reduction:
            audio_in["noise_reduction"] = {"type": self.noise_reduction}
        return {
            "type": "realtime",
            "instructions": instructions,
            "tools": tools,
            "tool_choice": "auto",
            "output_modalities": ["audio"],
            "audio": {
                "input": audio_in,
                "output": {"format": {"type": "audio/pcm", "rate": RT_RATE}, "voice": self.voice},
            },
        }

    async def open(self) -> None:
        if not self.api_key or self.api_key.startswith("你的"):
            raise RuntimeError("api_key 未配置")

        async def _connect():
            ws = await connect(f"{self.url}?model={self.model}",
                               additional_headers={"Authorization": f"Bearer {self.api_key}"},
                               max_size=None, open_timeout=None, ping_interval=20, ping_timeout=20)
            try:
                ev = json.loads(await ws.recv())
                if ev.get("type") != "session.created":
                    code = (ev.get("error") or {}).get("code") if ev.get("type") == "error" else ""
                    raise RuntimeError(f"first event {ev.get('type')} {code or ''}".strip())
                await ws.send(json.dumps({"type": "session.update", "session": self._session()}, ensure_ascii=False))
                return ws
            except BaseException:
                self.spawn(ws.close())
                raise

        t0 = time.monotonic()
        self.ws = await asyncio.wait_for(_connect(), self.connect_timeout_s)
        self.rx_task = self.spawn(self._rx_loop())
        self._arm_idle()
        self.logger.bind(tag=TAG).info(
            f"Realtime 会话已建立 model={self.model} turn={self.turn_detection_type} "
            f"echo_gate={'on' if self.gate else 'off'} noise_reduction={self.noise_reduction or 'off'} "
            f"local_barge_frames={self.local_barge_frames} lag_prior_ms={self.lag_prior_ms}{'(记忆)' if self.lag_prior_remembered else ''} "
            f"lag_max_ms={self.lag_max_ms} in {time.monotonic() - t0:.2f}s")

    def after_components_ready(self) -> None:
        asyncio.run_coroutine_threadsafe(self._after_ready(), self.conn.loop)

    async def _after_ready(self) -> None:
        await self._refresh_session()
        from core.utils.s2s import drain_pending_texts
        await drain_pending_texts(self.conn)  # provider 建成前到的文字（唤醒词 detect 等）现在补送

    async def _refresh_session(self) -> None:
        session = self._session()
        await self._send({"type": "session.update", "session": session})
        self.logger.bind(tag=TAG).info(f"会话配置已同步：指令 {len(session['instructions'])} 字、工具 {len(session['tools'])} 个")

    async def _maybe_refresh_tools(self) -> None:
        """提示词或工具清单变了就重新同步：上游的 Home Assistant 设备清单（append_devices_to_prompt）和 MCP 工具
        都是在我们第一次同步之后才异步挂上 conn.prompt / func_handler 的。"""
        tools = self._tool_list()
        sig = (self._instructions(), tuple(sorted(t["name"] for t in tools)))
        if sig != self._session_sig:
            await self._refresh_session()

    async def _send(self, obj: dict) -> None:
        if self.ws is None:
            return
        with contextlib.suppress(ConnectionClosed):
            await self.ws.send(json.dumps(obj, ensure_ascii=False))

    async def close(self) -> None:
        self.closing = True
        if self._idle_handle is not None:
            self._idle_handle.cancel()
        for t in list(self._tool_tasks):
            t.cancel()
        await super().close()
        if self.ws is not None:
            with contextlib.suppress(Exception):
                await self.ws.close()
        if self.gate is not None:
            self.stats["gate"] = dict(self.gate.stats, lag_ms=round(self.gate.lag * 1000),
                                      g=None if self.gate.g is None else round(self.gate.g, 4))
            if self.gate.lag_corr is not None and self.device_id:
                _LAST_LAG[self.device_id] = self.gate.lag   # 下一段会话的先验
            elif (self.device_id and self.lag_prior_remembered
                  and self.gate.stats.get("lag_est", 0) >= LAG_FORGET_AFTER):
                _LAST_LAG.pop(self.device_id, None)
                self.stats["gate"]["lag_forgotten"] = True
                self.logger.bind(tag=TAG).info(
                    f"滞后记忆 {self.lag_prior_ms} ms 本段估了 {self.gate.stats['lag_est']} 次都没确认，清掉，"
                    f"下一段回到配置先验 {int(self.config.get('gate_lag_prior_ms', 200))} ms")
        self.logger.bind(tag=TAG).info(f"Realtime 会话结束: {self.stats}")

    # ---------- 设备 → 模型 ----------
    async def on_pcm(self, pcm16k: bytes) -> None:
        if self.closing or not pcm16k:
            return
        self.trace.feed(pcm16k, self.speaker.speaking)
        if self.gate is not None:
            pcm16k, info = self.gate.process(pcm16k, time.monotonic(), self.speaker.speaking)
            if self.local_barge_frames:
                # 只数门控新判出的人声帧（pass）；保持期（pass-hold）和回声判决都记 0 但不清窗——否则一帧 pass 带出的
                # 300 ms 保持就凑够了 N 帧（11-01 真机：模型开口 240～420 ms 就被一声响动本地打断），或一帧回声判决
                # 就把攒了 4 帧的人声清零（11-01 真人测试：13 帧人声没触发）
                if not self.speaker.speaking:
                    self._barge_marks.clear()
                else:
                    self._barge_marks.append(1 if info == "pass" else 0)
                    self.barge_run = sum(self._barge_marks)
                    if self.barge_run >= self.local_barge_frames:
                        self._barge_marks.clear()
                        await self._barge("local")
        # 设备 listen start 之前（唤醒词那几十帧）的音频不送模型：否则模型侧 VAD 把"你好小智"当一轮提问，唤醒后答两次。
        # 状态记在连接上（listenMessageHandler 维护）：provider 建成常晚于 listen start，自己记会错过它
        if not getattr(self.conn, "client_is_listening", False):
            return
        pcm24 = self.up_rs.process(pcm16k)
        await self._send({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm24).decode()})
        self.stats["up_frames"] += 1

    async def on_text(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        await self._maybe_refresh_tools()  # 插件 / MCP 可能刚注册完：第一轮就要看得到
        if text == WAKE_TRIGGER:
            if self.wake_greeting and self.resp_done and not self.speaker.speaking:
                await self._create_response({"instructions": self.wake_greeting})
            return
        await self.show_user_text(text)
        await self._send({"type": "conversation.item.create",
                          "item": {"type": "message", "role": "user",
                                   "content": [{"type": "input_text", "text": text}]}})
        await self._create_response()

    # ---------- 打断 ----------
    async def cancel(self, reason: str) -> None:
        """reason=device 时来自 handleAbortMessage（它已发 tts stop、置 client_abort）；
        模型侧 / 本地插话由 _barge 先调 handleAbortMessage，再经钩子到这里，真正的原因放在 _pending_barge。"""
        if reason == "device" and self._pending_barge:
            reason, self._pending_barge = self._pending_barge, None
        if self._interrupted and not self.resp_done:
            # 同一条回复已经被别的路打断过（例如模型侧 speech_started 和设备侧 abort 同一秒到）：不再发 response.cancel /
            # truncate——上游已经取消，再发只换来 response_cancel_not_active（11-01 XVF3800 设备侧打断真机里每次都有）
            self.stats["barge_dup"] += 1
            self.logger.bind(tag=TAG).info(f"打断（{reason}，重复）：本条回复已被（{self._interrupted}）打断，不再取消")
            return
        had_audio = self.speaker.speaking or self.speaker.frames_sent > 0
        played = self.speaker.played_ms()
        self.suppress = True
        await self.speaker.interrupt()
        if reason != "server" and not self.resp_done:
            await self._send({"type": "response.cancel"})  # 模型侧自己判的插话（interrupt_response）已经取消过
        if had_audio and self.item_id:
            await self._send({"type": "conversation.item.truncate", "item_id": self.item_id,
                              "content_index": 0, "audio_end_ms": played})
        self.stats[f"barge_{reason}"] = self.stats.get(f"barge_{reason}", 0) + 1
        if not self.resp_done:
            self._interrupted = reason
        self.logger.bind(tag=TAG).info(f"打断（{reason}）：截到 {played if had_audio else -1} ms")

    async def _barge(self, reason: str) -> None:
        self._pending_barge = reason
        await handleAbortMessage(self.conn)  # 发 tts stop、置 client_abort，再经 abortHandle 的钩子调到 cancel()

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
                    self.logger.bind(tag=TAG).error(f"处理 Realtime 事件 {ev.get('type')} 出错: {type(e).__name__}: {e}")
        except ConnectionClosed:
            pass
        if not self.closing:
            self.closing = True
            self.logger.bind(tag=TAG).warning("Realtime 上游断开，关闭本连接（设备重连时若上游仍不可用会回退级联）")
            self.spawn(self.conn.close())

    async def _on_event(self, et: str, ev: dict) -> None:
        conn = self.conn
        if et == "input_audio_buffer.speech_started":
            self.in_speech = True
            conn.last_activity_time = time.time() * 1000
            self.stats["speech_started"] += 1
            self._note_turn(ev.get("item_id"), ev.get("audio_start_ms"), None)
            if (self.speaker.speaking or self.speaker.frames_queued > self.speaker.frames_sent) and not self.suppress:
                await self._barge("server")
            await self._maybe_refresh_tools()  # 用户开口到模型决定调工具之间还有几百毫秒，够把新注册的工具同步上去
        elif et == "input_audio_buffer.speech_stopped":
            self.in_speech = False
            self.t_speech_stopped = time.monotonic()
            conn.last_activity_time = time.time() * 1000
            self._note_turn(ev.get("item_id"), None, ev.get("audio_end_ms"))
        elif et == "response.created":
            self.resp_done = False
            self.suppress = False
            self._interrupted = None
            self.t_response_requested = None
            self.item_id = None
            await self.speaker.begin()
        elif et == "response.output_item.added":
            item = ev.get("item") or {}
            if item.get("type") == "message":
                self.item_id = item.get("id")
        elif et == "response.output_item.done":
            item = ev.get("item") or {}
            if item.get("type") == "function_call":
                t = self.spawn(self._run_tool(item))
                self._tool_tasks.add(t)
                t.add_done_callback(self._tool_tasks.discard)
        elif et in ("response.output_audio.delta", "response.audio.delta"):
            if self.suppress:
                return
            b = base64.b64decode(ev.get("delta") or "")
            if b and self.t_speech_stopped is not None and not self.speaker.speaking:
                ms = round(1000 * (time.monotonic() - self.t_speech_stopped))
                self.stats["first_audio_ms"].append(ms)
                self.t_speech_stopped = None
                self.logger.bind(tag=TAG).info(f"首个音频片段：说完后 {ms} ms")
            await self.speaker.push(b)
        elif et in ("response.output_audio_transcript.done", "response.audio_transcript.done"):
            txt = ev.get("transcript") or ""
            if self.log_text:
                self.logger.bind(tag=TAG).info(f"assistant: {txt}")
            if txt and not self.suppress:
                await self.show_assistant_text(txt)
        elif et == "conversation.item.input_audio_transcription.completed":
            txt = (ev.get("transcript") or "").strip()
            if self.log_text:
                self.logger.bind(tag=TAG).info(f"user: {txt}")
            if txt and self._prompt_only(txt):
                self.stats["prompt_only"] += 1
                self.logger.bind(tag=TAG).info(f"转写只有提示词「{txt}」，按空转写处理")
                await self._on_empty_transcript(ev.get("item_id"))
            elif txt:
                await self.show_user_text(txt)
            else:
                await self._on_empty_transcript(ev.get("item_id"))
        elif et == "response.done":
            r = ev.get("response") or {}
            self.resp_done = True
            self.stats["turns"] += 1
            if r.get("status") != "completed":
                self.stats["cancelled"] += 1
            self.t_reply = time.monotonic()
            if not self.suppress:
                self.spawn(self.speaker.end())  # 等设备播完再发 tts stop；不阻塞事件循环，插话仍能及时处理
            self.logger.bind(tag=TAG).info(
                f"回复 {r.get('status')} ({(r.get('status_details') or {}).get('reason') or '-'})")
            if self.want_response:
                await self._create_response()
            await self._maybe_refresh_tools()
        elif et == "error":
            err = ev.get("error") or {}
            code = err.get("code")
            if code == "conversation_already_has_active_response" and not self.want_response:
                self.want_response = True  # 工具结果的 response.create 撞上了刚开始的新回复：等它结束再发
            msg = f"Realtime error type={err.get('type')} code={code}"
            if code in BENIGN_ERRORS:
                self.logger.bind(tag=TAG).info(msg)
            else:
                self.logger.bind(tag=TAG).warning(msg)

    def _note_turn(self, item_id, start_ms, end_ms) -> None:
        """按 item_id 记模型侧判出的开口起止（毫秒，缓冲区时间轴），空转写到了用它算开口时长。只留最近 32 条。"""
        if not item_id:
            return
        rec = self._turn_ms.setdefault(item_id, [None, None])
        if start_ms is not None:
            rec[0] = start_ms
        if end_ms is not None:
            rec[1] = end_ms
        while len(self._turn_ms) > 32:
            self._turn_ms.popitem(last=False)

    def _turn_seconds(self, item_id):
        rec = self._turn_ms.get(item_id) if item_id else None
        if not rec or rec[0] is None or rec[1] is None:
            return None
        return max(0.0, (rec[1] - rec[0]) / 1000.0)

    def _prompt_only(self, txt: str) -> bool:
        return self.prompt_only_as_empty and bool(self._prompt_tokens) and _norm_text(txt) in self._prompt_tokens

    async def _on_empty_transcript(self, item_id) -> None:
        """模型侧 VAD 判出"开口"但转写不出一个字：房间里的响动、设备 AGC 回升、播完后的尾音（11-01 真机：每次播完后
        1～7 秒出现一两次，模型回"不知道你想表达什么"）。把这条空输入从会话里删掉，别让它留在上下文里；
        它触发的回复若还没出声、且开口很短（< empty_cancel_max_s）就一并取消。开口长的多半是真提问被设备侧压碎
        （11-01 15:27：XMOS 双讲把近端压到 rms 450～750），这时不取消，让模型回"没听清"，用户知道要重说。
        已经出声的不动——转写是异步的，这时在播的也可能是别的回复，掐断更怪。时长不明（没收到起止事件）按短开口处理。"""
        self.stats["empty_transcripts"] += 1
        dur = self._turn_seconds(item_id)
        dur_txt = "?" if dur is None else f"{dur:.1f}"
        if item_id:
            await self._send({"type": "conversation.item.delete", "item_id": item_id})
        unvoiced = not self.resp_done and not self.speaker.speaking and self.speaker.frames_sent == 0
        if dur is not None and dur >= self.empty_cancel_max_s:
            self.stats["empty_long"] += 1
            self.logger.bind(tag=TAG).info(f"空转写：开口 {dur_txt} s ≥ {self.empty_cancel_max_s:g} s，当作真提问被压碎：只删除该输入，让模型回应")
        elif unvoiced:
            self.suppress = True
            self._interrupted = "empty"
            await self._send({"type": "response.cancel"})
            self.stats["empty_cancelled"] += 1
            self.logger.bind(tag=TAG).info(f"空转写：开口 {dur_txt} s，删除该输入并取消尚未出声的回复")
        else:
            self.logger.bind(tag=TAG).info(f"空转写：开口 {dur_txt} s，删除该输入（回复已出声或没有进行中的回复，不取消）")

    # ---------- 工具 ----------
    async def _run_tool(self, item: dict) -> None:
        self.stats["tools"] += 1
        output = await self.tools.run(item.get("name"), item.get("arguments") or "{}")
        await self._send({"type": "conversation.item.create",
                          "item": {"type": "function_call_output", "call_id": item.get("call_id"), "output": output}})
        if self.resp_done:
            await self._create_response()
        else:
            self.want_response = True  # 当前还有回复在进行（例如函数调用所在的那条还没 done）：done 时再发

    async def _create_response(self, response: dict | None = None) -> None:
        self.want_response = False
        self.t_response_requested = time.monotonic()
        await self._send({"type": "response.create", **({"response": response} if response else {})})

    # ---------- 空闲关闭（call_later，不用 sleep 循环） ----------
    def _arm_idle(self) -> None:
        if self.idle_close_s <= 0 or self.closing:
            return
        self._idle_handle = self.conn.loop.call_later(1.0, self._idle_tick)

    def _idle_due(self, now: float) -> bool:
        if self.speaker.speaking or not self.resp_done or self.want_response or self._tool_tasks:
            return False
        if self.t_response_requested is not None and now - self.t_response_requested < RESPONSE_START_WAIT_S:
            return False
        if self.in_speech:
            return False
        if self.t_speech_stopped is not None and now - self.t_speech_stopped < SPEECH_GRACE_S:
            return False
        return now - self.t_reply > self.idle_close_s

    def _idle_tick(self) -> None:
        if self.closing:
            return
        if self._idle_due(time.monotonic()):
            self.logger.bind(tag=TAG).info(f"空闲超过 {self.idle_close_s:.0f} 秒，关闭连接")
            self.closing = True
            self.spawn(self.conn.close())
            return
        self._arm_idle()
