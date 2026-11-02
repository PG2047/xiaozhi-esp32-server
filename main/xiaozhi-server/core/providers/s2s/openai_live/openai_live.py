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
import time

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from core.providers.s2s.audio import Resampler
from core.providers.s2s.base import DeviceSpeaker, S2SProviderBase, UplinkTrace
from core.providers.s2s.echo_gate import EchoGate, load_vad
from core.providers.s2s.tools import ToolRunner, to_realtime_tools

TAG = __name__

LIVE_RATE = 16000       # 上下行都用 16 kHz（Live 支持 16 k / 24 k）
WAKE_TRIGGER = "嘿，你好呀"


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
        self.connect_timeout_s = float(c.get("connect_timeout_s", 4))
        self.idle_close_s = float(c.get("idle_close_s", 45))
        self.speech_rms = int(c.get("speech_rms", 30))          # 下行帧 RMS 超过它算"在说话"（模型静音时发的是全零）
        self.end_silence_ms = int(c.get("end_silence_ms", 700))  # 段内连续静音这么久算一段结束
        self.log_text = bool(c.get("log_text", False))
        # 工具：tools: false 关掉 delegation；后台模型默认 gpt-6-luna（官方示例）
        self.tools_enabled = bool(c.get("tools", True))
        self.delegation_model = str(c.get("delegation_model") or "gpt-6-luna")
        self.delegation_instructions = str(c.get("delegation_instructions") or
                                           "你是小智的后台，负责调用工具（家居控制、设备音量等）并给出一两句中文结论；不需要工具的问题直接简短回答。")
        self.tools = ToolRunner(conn, conn.logger, per_minute=int(c.get("tool_calls_per_minute", 12)),
                                allow=c.get("tools_allow") or None)
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

        gate = None
        if c.get("echo_gate", True):
            gate = EchoGate(vad=load_vad(self._vad_model_path()),
                            lag_prior=float(c.get("gate_lag_prior_ms", 200)) / 1000,
                            lag_max=float(c.get("gate_lag_max_ms", 1200)) / 1000)
        self.gate = gate
        out_rate = getattr(conn, "sample_rate", LIVE_RATE)
        self.speaker = DeviceSpeaker(conn, out_rate, gate=gate)
        self.down_rs = Resampler(LIVE_RATE, out_rate) if out_rate != LIVE_RATE else None

        self.ws = None
        self.closing = False
        self.in_seg = False        # 正在把一段回复发给设备
        self.silence_ms = 0.0      # 段内连续静音时长
        self.suppress = False      # 设备 abort 后丢掉模型的音频，直到用户说完、模型也静音（见 _may_resume）
        self.t_abort = 0.0
        self.in_text = []
        self.out_text = []
        self._idle_handle = None
        self.t_activity = time.monotonic()
        self.t_last_input_text = None
        self.t_seg = time.monotonic()
        self.stats = {"segments": 0, "barge_device": 0, "up_frames": 0, "seg_after_input_ms": [],
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
        if self.tools_enabled:
            tools = self._tool_list()
            self._session_sig = (instructions, tuple(sorted(t["name"] for t in tools)))
            out["delegation"] = {"type": "responses", "responses": {
                "model": self.delegation_model, "instructions": f"{instructions}\n\n{self.delegation_instructions}",
                "tools": tools, "tool_choice": "auto"}}
        else:
            self._session_sig = (instructions, ())
        return out

    def after_components_ready(self) -> None:
        asyncio.run_coroutine_threadsafe(self._maybe_refresh(), self.conn.loop)

    async def _maybe_refresh(self) -> None:
        """提示词或工具清单变了就 session.update（HA 设备清单、MCP 工具都在会话建立之后才挂上）。"""
        tools = self._tool_list() if self.tools_enabled else []
        sig = (self._instructions(), tuple(sorted(t["name"] for t in tools)))
        if sig == self._session_sig or self.ws is None:
            return
        await self._send({"type": "session.update", "session": self._mutable_session(update=True)})
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
        self.spawn(self._rx_loop())
        self._arm_idle()
        self.logger.bind(tag=TAG).info(
            f"Live 会话已建立 model={self.model} voice={self.voice} echo_gate={'on' if self.gate else 'off'} "
            f"speech_rms={self.speech_rms} end_silence_ms={self.end_silence_ms} "
            f"tools={'off' if not self.tools_enabled else len(self._session_sig[1])}@{self.delegation_model} in {time.monotonic() - t0:.2f}s")

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
        self.logger.bind(tag=TAG).info(f"Live 会话结束: {self.stats}")

    # ---------- 设备 → 模型 ----------
    async def on_pcm(self, pcm16k: bytes) -> None:
        if self.closing or not pcm16k:
            return
        self.trace.feed(pcm16k, self.in_seg)
        if self.gate is not None:
            pcm16k, _ = self.gate.process(pcm16k, time.monotonic(), self.speaker.speaking)
        if not self.forward_before_listen and not getattr(self.conn, "client_is_listening", False):
            return
        await self._send({"type": "session.input_audio.append", "audio": base64.b64encode(pcm16k).decode()})
        self.stats["up_frames"] += 1

    async def on_text(self, text: str) -> None:
        # Live 没有"按文字起一轮回复"的入口（探针未验证）：唤醒词那句不处理，用户开口后模型自己接话
        if (text or "").strip() and text.strip() != WAKE_TRIGGER:
            self.logger.bind(tag=TAG).info("Live 暂不支持文字输入，已忽略")

    # ---------- 打断 ----------
    async def cancel(self, reason: str) -> None:
        """只会来自设备 abort（handleAbortMessage）：模型侧的插话 Live 自己处理，表现为下行变静音。"""
        played = self.speaker.played_ms()
        self.suppress = True
        self.t_abort = time.time()
        await self._end_segment(interrupted=True)
        self.stats["barge_device"] += 1
        self.logger.bind(tag=TAG).info(f"打断（{reason}）：本段已播 {played} ms t={time.time():.3f}")

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
            if self.t_last_input_text is None or now - self.t_last_input_text > 1.5:
                # 一句话的第一个转写片段：探针用它对齐"用户开口 → 模型听到"（时间戳给毫秒，日志自带的只到秒）
                self.logger.bind(tag=TAG).info(f"Live 听到用户 t={time.time():.3f} playing={self.in_seg}")
                self.spawn(self._maybe_refresh())   # 用户开口到后台调工具之间还有一两秒，够把新注册的工具同步上去
            self.in_text.append(ev.get("delta") or "")
            self.t_last_input_text = now
            self.t_activity = now
        elif et == "session.output_transcript.delta":
            self.out_text.append(ev.get("delta") or "")
        elif et == "response.event":
            await self._on_backend_event(ev.get("event") or {})
        elif et == "session.delegation.created":
            d = ev.get("delegation") or {}
            self.stats["delegations"] = self.stats.get("delegations", 0) + 1
            self.logger.bind(tag=TAG).info(f"Live 委托后台 target={d.get('target')} t={time.time():.3f}")
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
        if interrupted:
            await self.speaker.interrupt()
        else:
            self.spawn(self.speaker.end())
        await self._flush_input_text()

    async def _flush_input_text(self) -> None:
        txt = "".join(self.in_text).strip()
        self.in_text.clear()
        if not txt:
            return
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
        if not self.speaker.speaking and not self.in_seg and time.monotonic() - self.t_activity > self.idle_close_s:
            self.logger.bind(tag=TAG).info(f"空闲超过 {self.idle_close_s:.0f} 秒，关闭连接")
            self.closing = True
            self.spawn(self.conn.close())
            return
        self._arm_idle()
