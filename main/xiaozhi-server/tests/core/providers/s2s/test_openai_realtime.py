"""OpenAI Realtime provider 的契约测试：起一个假的 Realtime WebSocket 服务端按脚本发事件，断言 provider 对上游发了什么、
对设备发了什么。不出网、不用真 key、不碰 ConnectionHandler（连接对象用最小的假对象）。

运行：在 main/xiaozhi-server 目录下 `pytest tests/core/providers/s2s -q`。
"""
from __future__ import annotations

import asyncio
import base64
import json
import time

import numpy as np
import pytest
from websockets.asyncio.server import serve

from config.logger import setup_logging
from core.handle.abortHandle import handleAbortMessage
from core.providers.s2s.openai_realtime.openai_realtime import WAKE_TRIGGER, S2SProvider
from core.utils.dialogue import Dialogue
from core.utils.s2s import attach_s2s
from plugins_func.register import Action, ActionResponse

# conftest 在 mock 模式把 asyncio.sleep 换成了不让出事件循环的空函数；上游节拍器和本测试的等待都要真睡。
_real_sleep = asyncio.tasks.sleep
# setup_logging() 没有配置缓存时会 asyncio.run(load_config())，在事件循环里会炸；生产由 app.py 的 load_config 填缓存，测试这里自己填
from tests.conftest import CONFIG  # noqa: E402
from core.utils.cache.manager import CacheType, cache_manager  # noqa: E402

cache_manager.set(CacheType.CONFIG, "main_config", CONFIG)
LOGGER = setup_logging()


@pytest.fixture(autouse=True)
def _restore_sleep(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _real_sleep)


def _opus_available() -> bool:
    try:
        from core.utils.opus_encoder_utils import OpusEncoderUtils
        OpusEncoderUtils(24000, 1, 60)
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _opus_available(), reason="Opus 编码器不可用（opuslib_next / libopus）")


class FakeDeviceWS:
    def __init__(self):
        self.sent = []

    async def send(self, m):
        self.sent.append(m)

    def texts(self):
        return [json.loads(m) for m in self.sent if isinstance(m, str)]

    def frames(self):
        return [m for m in self.sent if isinstance(m, bytes)]

    def states(self):
        return [(t.get("type"), t.get("state")) for t in self.texts()]


class FakeFuncHandler:
    def __init__(self):
        self.calls = []

    def get_functions(self):
        return [{"type": "function", "function": {
            "name": "hass_state", "description": "控制家里的设备",
            "parameters": {"type": "object", "properties": {"entity_id": {"type": "string"}}, "required": ["entity_id"]}}}]

    async def handle_llm_function_call(self, conn, data):
        self.calls.append(data)
        return ActionResponse(Action.RESPONSE, None, "客厅灯已打开")


class FakeConn:
    def __init__(self, loop):
        self.loop = loop
        self.logger = LOGGER
        self.config = {"selected_module": {"VAD": "SileroVAD"}, "VAD": {"SileroVAD": {"model_dir": "/nonexistent"}},
                       "tts_audio_send_delay": 0, "enable_stop_tts_notify": False, "end_prompt": {}}
        self.sample_rate = 24000
        self.session_id = "sess"
        self.websocket = FakeDeviceWS()
        self.dialogue = Dialogue()
        self.prompt = "你是测试助手"
        self.func_handler = FakeFuncHandler()
        self.client_abort = False
        self.client_is_speaking = False
        self.client_is_listening = False
        self.close_after_chat = False
        self.last_activity_time = 0.0
        self.conn_from_mqtt_gateway = False
        self.tts = None
        self.sentence_id = None
        self.s2s = None
        self.closed = False

    def clearSpeakStatus(self):
        self.client_is_speaking = False

    def clear_queues(self):
        pass

    async def close(self, ws=None):
        self.closed = True


class FakeRealtime:
    """假上游：握手回 session.created，之后记录收到的每条事件；测试用 emit() 往下发。"""

    def __init__(self, first_event=None):
        self.received = []
        self.ws = None
        self.first_event = first_event or {"type": "session.created", "session": {}}

    async def handler(self, ws):
        self.ws = ws
        self.closed = False
        await ws.send(json.dumps(self.first_event))
        async for raw in ws:
            self.received.append(json.loads(raw))
        self.closed = True

    async def emit(self, ev: dict):
        await self.ws.send(json.dumps(ev))

    def types(self):
        return [m["type"] for m in self.received]

    def of(self, typ):
        return [m for m in self.received if m["type"] == typ]

    async def wait_for(self, typ, n=1, timeout=3.0):
        t0 = time.monotonic()
        while len(self.of(typ)) < n:
            if time.monotonic() - t0 > timeout:
                raise AssertionError(f"上游 {timeout}s 内没收到 {n} 条 {typ}，收到的是 {self.types()}")
            await _real_sleep(0.01)
        return self.of(typ)


async def wait_until(pred, timeout=5.0, what="condition"):
    t0 = time.monotonic()
    while not pred():
        if time.monotonic() - t0 > timeout:
            raise AssertionError(f"{timeout}s 内没等到 {what}")
        await _real_sleep(0.01)


def pcm(ms: int, rate: int = 16000, amp: int = 0) -> bytes:
    return (np.ones(rate * ms // 1000, np.int16) * amp).tobytes()


def b64(ms: int, rate: int = 24000) -> str:
    return base64.b64encode(pcm(ms, rate, 2000)).decode()


def make_provider(conn, url, **over):
    cfg = {"api_key": "test-key", "url": url, "model_name": "gpt-realtime", "echo_gate": False,
           "idle_close_s": 0, "connect_timeout_s": 2, **over}
    return S2SProvider(cfg, conn)


def run_scenario(scenario, first_event=None):
    async def go():
        rt = FakeRealtime(first_event)
        async with serve(rt.handler, "127.0.0.1", 0) as srv:
            url = f"ws://127.0.0.1:{srv.sockets[0].getsockname()[1]}/v1/realtime"
            conn = FakeConn(asyncio.get_running_loop())
            await scenario(rt, conn, url)
    asyncio.run(go())


async def _stream_response(rt, frames=8, item_id="item_1"):
    await rt.emit({"type": "response.created", "response": {"id": "resp_1"}})
    await rt.emit({"type": "response.output_item.added", "item": {"id": item_id, "type": "message"}})
    for _ in range(frames):
        await rt.emit({"type": "response.output_audio.delta", "delta": b64(60)})
        await _real_sleep(0.005)


# ---------------------------------------------------------------- 测试 ----------------------------------------------------------------

def test_open_sends_session_update_then_forwards_resampled_audio():
    async def sc(rt, conn, url):
        p = make_provider(conn, url)
        conn.s2s = p
        await p.open()
        (upd,) = await rt.wait_for("session.update")
        s = upd["session"]
        assert s["instructions"] == "你是测试助手" and s["type"] == "realtime"
        assert [t["name"] for t in s["tools"]] == ["hass_state"] and s["tools"][0]["type"] == "function"
        assert s["audio"]["input"]["format"]["rate"] == 24000 and s["audio"]["input"]["turn_detection"]["type"] == "server_vad"
        await p.on_pcm(pcm(60))                                                # listen start 之前（唤醒词音频）：不送
        await _real_sleep(0.05)
        assert rt.of("input_audio_buffer.append") == []
        conn.client_is_listening = True                                        # 上游 listen 处理器在 listen start 时置位
        await p.on_pcm(pcm(60))
        (app,) = await rt.wait_for("input_audio_buffer.append")
        assert abs(len(base64.b64decode(app["audio"])) - 24000 * 60 // 1000 * 2) <= 4   # 16 k → 24 k（重采样器首包差 1 样本）
        conn.client_is_listening = False
        await p.on_pcm(pcm(60))                                                # 不在聆听：不送
        await _real_sleep(0.05)
        assert len(rt.of("input_audio_buffer.append")) == 1
        conn.client_is_listening = True
        await p.on_pcm(pcm(60))
        await rt.wait_for("input_audio_buffer.append", 2)
        await p.close()                                                        # 连接关闭 → 上游会话也关
        await wait_until(lambda: rt.closed, what="上游看到客户端关闭")
        assert p.stats["up_frames"] == 2
    run_scenario(sc)


def test_session_noise_reduction_and_gate_window():
    """noise_reduction 透传进 session.audio.input；非法值忽略并告警；门控锁定前的参考窗盖到 gate_lag_max_ms。"""
    loop = asyncio.new_event_loop()
    try:
        conn = FakeConn(loop)
        p = make_provider(conn, "ws://unused", noise_reduction="far_field")
        assert p._session()["audio"]["input"]["noise_reduction"] == {"type": "far_field"}
        assert "noise_reduction" not in make_provider(conn, "ws://unused")._session()["audio"]["input"]
        bad = make_provider(conn, "ws://unused", noise_reduction="studio")
        assert bad.noise_reduction == "" and "noise_reduction" not in bad._session()["audio"]["input"]
        g = make_provider(conn, "ws://unused", echo_gate=True, gate_lag_prior_ms=900, gate_lag_max_ms=1500)  # 模型文件不在：能量判据
        assert g.gate is not None and g.gate.start_window == pytest.approx((0.65, 1.15)) and g.gate.lag == pytest.approx(0.9)
        assert g.gate._window() == pytest.approx((0.25, 0.25))   # 锁定前只在先验 ±250 ms 内找回声
        assert not g.lag_prior_remembered
    finally:
        loop.close()


def test_remembered_lag_becomes_next_sessions_prior():
    """会话结束时把锁定的滞后按 device-id 记住；同一台设备下一段会话用它当先验、起始窗围着它开。"""
    from core.providers.s2s.openai_realtime import openai_realtime as mod

    async def go():
        conn = FakeConn(asyncio.get_running_loop())
        conn.device_id = "test-dev-remember"
        mod._LAST_LAG.pop(conn.device_id, None)
        p = make_provider(conn, "ws://unused", echo_gate=True, gate_lag_prior_ms=900)
        p.gate.lag, p.gate.lag_corr = 0.78, 0.7
        await p.close()                                             # 没开过 ws：只收尾、记滞后
        assert mod._LAST_LAG[conn.device_id] == pytest.approx(0.78) and p.stats["gate"]["lag_ms"] == 780
        q = make_provider(conn, "ws://unused", echo_gate=True, gate_lag_prior_ms=900)
        assert q.lag_prior_remembered and q.lag_prior_ms == 780 and q.gate.lag == pytest.approx(0.78)
        assert q.gate.start_window == pytest.approx((0.53, 1.03))
        r = make_provider(conn, "ws://unused", echo_gate=True, gate_lag_prior_ms=900)
        r.gate.lag_corr = None                                      # 没锁定过的会话不覆盖记忆
        await r.close()
        assert mod._LAST_LAG[conn.device_id] == pytest.approx(0.78)
        mod._LAST_LAG.pop(conn.device_id, None)
    asyncio.run(go())


def test_local_barge_counts_only_fresh_pass_frames():
    """本地插话：最近 2N 帧里数门控新判出的人声帧（pass）；保持期（pass-hold）和回声判决记 0 但不清窗；停播即清窗。"""
    class StubGate:
        def __init__(self, labels):
            self.labels = list(labels)
            self.stats, self.lag = {}, 0.2

        def process(self, pcm, t, speaking):
            return pcm, self.labels.pop(0)

    async def go():
        conn = FakeConn(asyncio.get_running_loop())
        p = make_provider(conn, "ws://unused", local_barge_frames=3)
        barges = []

        async def fake_barge(reason):
            barges.append((reason, len(labels) - len(p.gate.labels)))
            p.speaker.speaking = False

        p._barge = fake_barge
        labels = ["pass", "pass-hold", "pass-hold", "pass", "echo", "pass", "pass-hold", "pass", "pass", "pass"]
        p.gate = StubGate(labels)
        p.speaker.speaking = True
        for _ in labels:
            await p.on_pcm(pcm(60))
        # 窗 6 帧：第 6 帧时窗内 pass = 第 1、4、6 帧 = 3 → 打断（保持期与回声判决都不清窗）；之后 speaking=False，窗清空
        assert barges == [("local", 6)] and len(p._barge_marks) == 0
        # 真人测试那种"人声被回声判决打散"的序列（11-01 15:27：13 帧 pass 夹着 echo）：连续计数永远到不了 5，滑窗能
        q = make_provider(conn, "ws://unused", local_barge_frames=5)          # 窗默认 10
        q._barge = fake_barge
        barges.clear()
        labels2 = ["pass", "pass", "echo", "pass", "pass", "echo", "pass", "pass", "echo"]
        q.gate = StubGate(labels2)
        q.speaker.speaking = True
        for _ in labels2:
            await q.on_pcm(pcm(60))
        assert [r for r, _ in barges] == ["local"] and q.local_barge_window == 10
        # 窗外的旧帧会滑出：稀疏的 pass 不触发
        r = make_provider(conn, "ws://unused", local_barge_frames=3, local_barge_window=4)
        r._barge = fake_barge
        barges.clear()
        labels3 = ["pass", "echo", "echo", "echo", "pass", "echo", "echo", "echo", "pass"]
        r.gate = StubGate(labels3)
        r.speaker.speaking = True
        for _ in labels3:
            await r.on_pcm(pcm(60))
        assert barges == []
    asyncio.run(go())


def test_response_audio_reaches_device_with_tts_messages_and_dialogue():
    async def sc(rt, conn, url):
        p = make_provider(conn, url)
        conn.s2s = p
        await p.open()
        await rt.wait_for("session.update")
        await rt.emit({"type": "conversation.item.input_audio_transcription.completed", "transcript": "现在几点"})
        await _stream_response(rt, frames=8)
        await rt.emit({"type": "response.output_audio_transcript.done", "transcript": "现在三点整。"})
        await rt.emit({"type": "response.done", "response": {"status": "completed"}})
        dev = conn.websocket
        await wait_until(lambda: ("tts", "stop") in dev.states(), what="设备收到 tts stop")
        states = dev.states()
        assert states.index(("tts", "start")) < states.index(("tts", "stop"))
        assert ("tts", "sentence_start") in states and ("stt", None) in states
        assert [t["text"] for t in dev.texts() if t.get("type") == "stt"] == ["现在几点"]
        assert [t["text"] for t in dev.texts() if t.get("state") == "sentence_start"] == ["现在三点整。"]
        assert 8 <= len(dev.frames()) <= 9                       # 8 帧 + 可能的补零尾帧
        assert [(m.role, m.content) for m in conn.dialogue.dialogue] == [("user", "现在几点"), ("assistant", "现在三点整。")]
        assert p.stats["turns"] == 1 and p.stats["cancelled"] == 0
        assert conn.client_is_speaking is False
        await p.close()
    run_scenario(sc)


def test_empty_transcript_deletes_item_and_cancels_only_unvoiced_response():
    """空转写：删掉该输入；短开口触发的回复还没出声就取消（后到的音频丢弃）；已出声的回复不取消；长开口不取消。"""
    async def sc(rt, conn, url):
        p = make_provider(conn, url)
        conn.s2s = p
        await p.open()
        await rt.wait_for("session.update")
        # 1) 模型侧判出 0.6 秒的"开口"→ 回复已创建但还没出声 → 空转写到
        await rt.emit({"type": "input_audio_buffer.speech_started", "item_id": "u_empty", "audio_start_ms": 1000})
        await rt.emit({"type": "input_audio_buffer.speech_stopped", "item_id": "u_empty", "audio_end_ms": 1600})
        await rt.emit({"type": "response.created", "response": {"id": "r1"}})
        await rt.emit({"type": "response.output_item.added", "item": {"id": "a1", "type": "message"}})
        await rt.emit({"type": "conversation.item.input_audio_transcription.completed", "item_id": "u_empty", "transcript": " "})
        (d,) = await rt.wait_for("conversation.item.delete")
        assert d["item_id"] == "u_empty"
        await rt.wait_for("response.cancel")
        await rt.emit({"type": "response.output_audio.delta", "delta": b64(60)})     # 取消后才到的音频：丢弃
        await rt.emit({"type": "response.done", "response": {"status": "cancelled", "status_details": {"reason": "client_cancelled"}}})
        await _real_sleep(0.1)
        assert conn.websocket.frames() == [] and ("tts", "start") not in conn.websocket.states()
        assert p.stats["empty_transcripts"] == 1 and p.stats["empty_cancelled"] == 1
        assert [m.content for m in conn.dialogue.dialogue] == []                       # 空转写不进对话历史
        # 2) 回复已经出声：只删项，不取消
        await _stream_response(rt, frames=6, item_id="a2")
        await wait_until(lambda: p.speaker.frames_sent >= 2, what="已出声")
        await rt.emit({"type": "conversation.item.input_audio_transcription.completed", "item_id": "u_empty2", "transcript": ""})
        dels = await rt.wait_for("conversation.item.delete", 2)
        assert dels[1]["item_id"] == "u_empty2"
        await _real_sleep(0.1)
        assert len(rt.of("response.cancel")) == 1 and p.stats["empty_transcripts"] == 2 and p.stats["empty_cancelled"] == 1
        # 3) 开口 2.5 秒的空转写（真提问被设备侧压碎）：只删项，回复还没出声也不取消，让模型回"没听清"
        await rt.emit({"type": "response.done", "response": {"status": "completed"}})
        await wait_until(lambda: p.resp_done, what="上一条回复结束")
        await rt.emit({"type": "input_audio_buffer.speech_started", "item_id": "u_long", "audio_start_ms": 9000})
        await rt.emit({"type": "input_audio_buffer.speech_stopped", "item_id": "u_long", "audio_end_ms": 11500})
        await rt.emit({"type": "response.created", "response": {"id": "r3"}})
        await rt.emit({"type": "conversation.item.input_audio_transcription.completed", "item_id": "u_long", "transcript": ""})
        dels = await rt.wait_for("conversation.item.delete", 3)
        assert dels[2]["item_id"] == "u_long"
        await _real_sleep(0.1)
        assert len(rt.of("response.cancel")) == 1 and p.stats["empty_long"] == 1 and p.stats["empty_cancelled"] == 1
        assert p.suppress is False                                   # 回复照常放给设备
        assert p._turn_seconds("u_long") == pytest.approx(2.5) and p._turn_seconds("nope") is None
        await p.close()
    run_scenario(sc)


def test_server_detected_barge_in_truncates_without_cancel():
    async def sc(rt, conn, url):
        p = make_provider(conn, url)
        conn.s2s = p
        await p.open()
        await rt.wait_for("session.update")
        await _stream_response(rt, frames=20)
        await wait_until(lambda: p.speaker.frames_sent >= 5, what="设备收到预缓冲帧")
        await rt.emit({"type": "input_audio_buffer.speech_started", "item_id": "u1", "audio_start_ms": 1000})
        (tr,) = await rt.wait_for("conversation.item.truncate")
        assert tr["item_id"] == "item_1" and tr["content_index"] == 0 and tr["audio_end_ms"] >= 0
        assert "response.cancel" not in rt.types()                 # 模型侧判的插话它自己已取消
        assert ("tts", "stop") in conn.websocket.states()           # handleAbortMessage 发的
        n_frames = len(conn.websocket.frames())
        await rt.emit({"type": "response.output_audio.delta", "delta": b64(60)})   # 残余音频应被丢弃
        await rt.emit({"type": "response.done", "response": {"status": "cancelled", "status_details": {"reason": "turn_detected"}}})
        await _real_sleep(0.1)
        assert len(conn.websocket.frames()) == n_frames
        assert p.stats["barge_server"] == 1 and p.stats["cancelled"] == 1
        await p.close()
    run_scenario(sc)


def test_second_barge_on_same_response_is_not_cancelled_again():
    """模型侧 speech_started 和设备侧 abort 同一秒到：只打断一次，不重复发 response.cancel / truncate；下一条回复照常可打断。"""
    async def sc(rt, conn, url):
        p = make_provider(conn, url)
        conn.s2s = p
        await p.open()
        await rt.wait_for("session.update")
        await _stream_response(rt, frames=20)
        await wait_until(lambda: p.speaker.frames_sent >= 5, what="设备收到预缓冲帧")
        await rt.emit({"type": "input_audio_buffer.speech_started", "item_id": "u1", "audio_start_ms": 1000})
        await rt.wait_for("conversation.item.truncate")
        await handleAbortMessage(conn)                                # 设备侧 abort 紧跟着到
        await _real_sleep(0.1)
        assert "response.cancel" not in rt.types() and len(rt.of("conversation.item.truncate")) == 1
        assert p.stats["barge_server"] == 1 and p.stats["barge_device"] == 0 and p.stats["barge_dup"] == 1
        await rt.emit({"type": "response.done", "response": {"status": "cancelled", "status_details": {"reason": "turn_detected"}}})
        # 下一条回复：设备 abort 两次 → 只取消一次
        await _stream_response(rt, frames=20, item_id="item_2")
        await wait_until(lambda: p.speaker.frames_sent >= 5, what="第二条回复出声")
        await handleAbortMessage(conn)
        await handleAbortMessage(conn)
        await _real_sleep(0.1)
        assert len(rt.of("response.cancel")) == 1 and len(rt.of("conversation.item.truncate")) == 2
        assert p.stats["barge_device"] == 1 and p.stats["barge_dup"] == 2
        await p.close()
    run_scenario(sc)


def test_device_abort_cancels_response_and_truncates():
    async def sc(rt, conn, url):
        p = make_provider(conn, url)
        conn.s2s = p
        await p.open()
        await rt.wait_for("session.update")
        await _stream_response(rt, frames=20)
        await wait_until(lambda: p.speaker.frames_sent >= 5, what="设备收到预缓冲帧")
        await handleAbortMessage(conn)                                # 设备按键 → abortHandle 钩子 → cancel("device")
        await rt.wait_for("response.cancel")
        await rt.wait_for("conversation.item.truncate")
        assert conn.client_abort is True and p.stats["barge_device"] == 1
        await p.close()
    run_scenario(sc)


def test_tool_call_roundtrip_and_text_entry():
    async def sc(rt, conn, url):
        p = make_provider(conn, url)
        conn.s2s = p
        await p.open()
        await rt.wait_for("session.update")
        await rt.emit({"type": "response.created", "response": {"id": "r"}})
        await rt.emit({"type": "response.output_item.done",
                       "item": {"type": "function_call", "call_id": "call_1", "name": "hass_state",
                                "arguments": '{"entity_id": "light.living"}'}})
        (out,) = await rt.wait_for("conversation.item.create")
        assert out["item"] == {"type": "function_call_output", "call_id": "call_1", "output": "客厅灯已打开"}
        assert conn.func_handler.calls[0]["name"] == "hass_state"
        assert "response.create" not in rt.types()                   # 当前回复还没 done，先记下
        await rt.emit({"type": "response.done", "response": {"status": "completed"}})
        await rt.wait_for("response.create")                         # done 后补发
        # 文字入口
        await p.on_text("把卧室灯关掉")
        items = await rt.wait_for("conversation.item.create", 2)
        assert items[1]["item"]["role"] == "user" and items[1]["item"]["content"][0]["text"] == "把卧室灯关掉"
        await rt.wait_for("response.create", 2)
        assert conn.dialogue.dialogue[-1].content == "把卧室灯关掉"
        # 唤醒词：只让模型短应答，不进对话历史
        await p.on_text(WAKE_TRIGGER)
        creates = await rt.wait_for("response.create", 3)
        assert creates[2]["response"]["instructions"].startswith("用户刚叫醒你")
        assert len(conn.dialogue.dialogue) == 1
        await p.close()
    run_scenario(sc)


def test_open_fails_on_error_first_event_and_attach_falls_back():
    async def sc(rt, conn, url):
        p = make_provider(conn, url)
        with pytest.raises(RuntimeError):
            await p.open()
        # attach_s2s 在执行器线程里调用，失败返回 None，连接走级联
        conn.config["selected_module"]["S2S"] = "OpenAIRealtime"
        conn.config["S2S"] = {"OpenAIRealtime": {"type": "openai_realtime", "api_key": "k", "url": url,
                                                 "connect_timeout_s": 1, "echo_gate": False}}
        got = await asyncio.get_running_loop().run_in_executor(None, attach_s2s, conn)
        assert got is None
        conn.config["selected_module"]["S2S"] = ""
        assert await asyncio.get_running_loop().run_in_executor(None, attach_s2s, conn) is None
    run_scenario(sc, first_event={"type": "error", "error": {"code": "invalid_api_key"}})


def test_attach_succeeds_and_upstream_close_closes_connection():
    async def sc(rt, conn, url):
        conn.config["selected_module"]["S2S"] = "OpenAIRealtime"
        conn.config["S2S"] = {"OpenAIRealtime": {"type": "openai_realtime", "api_key": "k", "url": url,
                                                 "connect_timeout_s": 2, "echo_gate": False, "idle_close_s": 0}}
        p = await asyncio.get_running_loop().run_in_executor(None, attach_s2s, conn)
        assert isinstance(p, S2SProvider)
        conn.s2s = p
        await rt.wait_for("session.update")
        p.after_components_ready()                                   # 执行器线程里的那次同步
        await rt.wait_for("session.update", 2)
        conn.prompt = conn.prompt + "\n下面是我家智能设备列表（位置，设备名，entity_id）\n书房,书房灯,light.study_real\n"  # 上游 HA 插件事后追加
        await rt.emit({"type": "input_audio_buffer.speech_started", "item_id": "u2", "audio_start_ms": 10})
        upd = await rt.wait_for("session.update", 3)                 # 用户一开口就把新提示词同步上去
        assert "light.study_real" in upd[2]["session"]["instructions"]
        await rt.ws.close()                                          # 上游断开
        await wait_until(lambda: conn.closed, what="连接被关闭")
    run_scenario(sc)
