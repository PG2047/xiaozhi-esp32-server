"""GPT-Live provider 的契约测试：假 Live 服务端按脚本发连续音频流（静音是全零），断言分段、tts start/stop 与打断行为。
不出网、不用真 key。运行：在 main/xiaozhi-server 目录下 `pytest tests/core/providers/s2s -q`。
"""
from __future__ import annotations

import asyncio
import base64
import json

import numpy as np
import pytest
from websockets.asyncio.server import serve

from tests.core.providers.s2s.test_openai_realtime import (  # noqa: F401  复用夹具（也带上 _restore_sleep 与 Opus 跳过标记）
    FakeConn, _real_sleep, _restore_sleep, pytestmark, wait_until,
)
from core.providers.s2s.openai_live.openai_live import S2SProvider


class FakeLive:
    def __init__(self):
        self.received = []
        self.ws = None

    async def handler(self, ws):
        self.ws = ws
        async for raw in ws:
            m = json.loads(raw)
            self.received.append(m)
            if m["type"] == "session.start":
                await ws.send(json.dumps({"type": "session.started", "session": {"id": "live_x"}}))

    async def emit(self, ev):
        await self.ws.send(json.dumps(ev))

    async def audio(self, n, amp):
        for _ in range(n):
            d = base64.b64encode((np.ones(1600, np.int16) * amp).tobytes()).decode()  # 100 ms @16k
            await self.ws.send(json.dumps({"type": "session.output_audio.delta", "delta": d}))
            await _real_sleep(0.005)


def run(scenario):
    async def go():
        lv = FakeLive()
        async with serve(lv.handler, "127.0.0.1", 0) as srv:
            url = f"ws://127.0.0.1:{srv.sockets[0].getsockname()[1]}/v1/live/sessions"
            conn = FakeConn(asyncio.get_running_loop())
            p = S2SProvider({"api_key": "k", "url": url, "echo_gate": False, "idle_close_s": 0,
                             "end_silence_ms": 300, "wake_greeting_text": ""}, conn)
            conn.s2s = p
            await p.open()
            await scenario(lv, conn, p)
            await p.close()
    asyncio.run(go())


def test_session_start_and_silence_outside_segment_is_dropped():
    async def sc(lv, conn, p):
        assert lv.received[0]["session"]["model"] == "gpt-live-1"
        assert lv.received[0]["session"]["instructions"] == "你是测试助手"
        await lv.audio(5, 0)
        await _real_sleep(0.1)
        assert not conn.websocket.frames() and not p.speaker.speaking
    run(sc)


def test_speech_then_silence_makes_one_segment_with_tts_stop():
    async def sc(lv, conn, p):
        await lv.audio(6, 2000)
        await wait_until(lambda: p.speaker.speaking, what="开始播放")
        await lv.audio(5, 0)
        await wait_until(lambda: ("tts", "stop") in conn.websocket.states(), timeout=8, what="tts stop")
        assert p.stats["segments"] == 1 and not p.in_seg
    run(sc)


def test_false_device_abort_resumes_after_max_wait():
    async def sc(lv, conn, p):
        p.resume_max_wait_s = 0.0          # 打断后一直没听到用户：等够 resume_max_wait 就恢复
        await lv.audio(4, 2000)
        await wait_until(lambda: p.in_seg, what="进段")
        await p.cancel("device")
        n = len(conn.websocket.frames())
        await lv.audio(4, 2000)          # 模型还在说本段：丢
        await _real_sleep(0.1)
        assert len(conn.websocket.frames()) == n and p.stats["segments"] == 1
        await lv.audio(1, 0)             # 模型停了
        await lv.audio(3, 2000)          # 新的一段
        await wait_until(lambda: p.stats["segments"] == 2, what="第二段")
    run(sc)


def test_after_device_abort_waits_until_user_finished():
    async def sc(lv, conn, p):
        p.resume_after_user_s = 0.2
        loud_up = (np.ones(960, np.int16) * 3000).tobytes()
        quiet_up = (np.ones(960, np.int16) * 10).tobytes()
        await lv.audio(4, 2000)
        await wait_until(lambda: p.in_seg, what="进段")
        await p.on_pcm(loud_up)          # 用户开口
        await p.cancel("device")
        await lv.audio(1, 0)             # 模型短停顿
        await lv.audio(3, 2000)          # 模型接着数（用户还在说）：必须丢
        await _real_sleep(0.1)
        assert p.suppress and p.stats["segments"] == 1
        await _real_sleep(0.45)
        await p.on_pcm(quiet_up)         # 用户说完（上行静了 0.4 秒以上）
        await lv.audio(1, 0)             # 模型静音：可以恢复
        await lv.audio(3, 2000)          # 对用户的回应
        await wait_until(lambda: p.stats["segments"] == 2, what="回应段")
    run(sc)


def test_uplink_only_while_listening_when_configured():
    async def sc(lv, conn, p):
        p.forward_before_listen = False
        await p.on_pcm(b"\x01\x00" * 960)
        conn.client_is_listening = True
        await p.on_pcm(b"\x01\x00" * 960)
        await wait_until(lambda: any(m["type"] == "session.input_audio.append" for m in lv.received), what="上行")
        assert sum(m["type"] == "session.input_audio.append" for m in lv.received) == 1
    run(sc)


def test_wake_audio_forwarded_by_default():
    async def sc(lv, conn, p):
        assert p.forward_before_listen and lv.received[0]["session"]["audio"]["output"]["voice"] == "marin"
        await p.on_pcm(b"\x01\x00" * 960)
        await wait_until(lambda: any(m["type"] == "session.input_audio.append" for m in lv.received), what="上行")
    run(sc)


def test_instructions_suffix_appended():
    async def go():
        lv = FakeLive()
        async with serve(lv.handler, "127.0.0.1", 0) as srv:
            url = f"ws://127.0.0.1:{srv.sockets[0].getsockname()[1]}/v1/live/sessions"
            conn = FakeConn(asyncio.get_running_loop())
            p = S2SProvider({"api_key": "k", "url": url, "echo_gate": False, "idle_close_s": 0,
                             "instructions_suffix": "附和时继续说。", "wake_greeting_text": ""}, conn)
            conn.s2s = p
            await p.open()
            assert lv.received[0]["session"]["instructions"] == "你是测试助手\n\n附和时继续说。"
            await p.close()
    asyncio.run(go())


def test_uplink_trace_logs_onset_then_offset():
    from core.providers.s2s.base import UplinkTrace

    class L:
        def __init__(self):
            self.msgs = []

        def bind(self, **_):
            return self

        def info(self, m):
            self.msgs.append(m)

    lg = L()
    tr = UplinkTrace(lg, "t", "Live", rms=800, quiet_s=0.0)
    loud = (np.ones(960, np.int16) * 3000).tobytes()
    quiet = (np.ones(960, np.int16) * 10).tobytes()
    tr.feed(quiet, False)
    tr.feed(loud, True)
    tr.feed(loud, True)
    tr.feed(quiet, True)
    assert len(lg.msgs) == 2 and "上行开口" in lg.msgs[0] and "playing=True" in lg.msgs[0] and "上行收声" in lg.msgs[1]


def test_delegation_declares_tools_and_returns_function_output():
    async def sc(lv, conn, p):
        d = lv.received[0]["session"]["delegation"]
        assert d["type"] == "responses" and d["responses"]["model"] == "gpt-6-luna"
        assert [t["name"] for t in d["responses"]["tools"]] == ["hass_state"]
        await lv.emit({"type": "session.delegation.created", "delegation": {"id": "item_1", "target": "responses"}})
        await lv.emit({"type": "response.event", "event": {"type": "response.created", "response": {"id": "resp_1"}}})
        await lv.emit({"type": "response.event", "event": {"type": "response.output_item.done", "item": {
            "type": "function_call", "call_id": "call_1", "name": "hass_state", "arguments": "{\"entity_id\": \"light.x\"}"}}})
        await lv.emit({"type": "response.event", "event": {"type": "response.completed", "response": {"id": "resp_1"}}})
        await wait_until(lambda: any(m["type"] == "response.create" for m in lv.received), what="response.create")
        outs = [m for m in lv.received if m["type"] == "response.item.create"]
        assert len(outs) == 1 and outs[0]["item"]["call_id"] == "call_1" and outs[0]["item"]["output"] == "客厅灯已打开"
        assert conn.func_handler.calls and conn.func_handler.calls[0]["name"] == "hass_state"
        # response.create 必须在函数结果之后
        types = [m["type"] for m in lv.received]
        assert types.index("response.item.create") < types.index("response.create")
    run(sc)


def test_session_update_when_tools_change():
    async def sc(lv, conn, p):
        conn.prompt = "你是测试助手（已加设备清单）"
        await p._maybe_refresh()
        await wait_until(lambda: any(m["type"] == "session.update" for m in lv.received), what="session.update")
        upd = [m for m in lv.received if m["type"] == "session.update"][-1]["session"]
        assert "instructions" not in upd and upd["delegation"]["responses"]["tools"]
        assert upd["delegation"]["responses"]["instructions"].startswith("你是测试助手（已加设备清单）")
        n = sum(m["type"] == "session.update" for m in lv.received)
        await p._maybe_refresh()          # 没变化不重复发
        await _real_sleep(0.1)
        assert sum(m["type"] == "session.update" for m in lv.received) == n
    run(sc)


def test_wake_trigger_plays_cached_greeting():
    async def go():
        lv = FakeLive()
        async with serve(lv.handler, "127.0.0.1", 0) as srv:
            url = f"ws://127.0.0.1:{srv.sockets[0].getsockname()[1]}/v1/live/sessions"
            conn = FakeConn(asyncio.get_running_loop())
            p = S2SProvider({"api_key": "k", "url": url, "echo_gate": False, "idle_close_s": 0}, conn)

            async def fake_tts():
                return (np.ones(conn.sample_rate * 3 // 10, np.int16) * 2000).tobytes()   # 0.3 秒，不出网
            p._greeting_pcm = fake_tts
            conn.s2s = p
            await p.open()
            await p.on_text("嘿，你好呀")
            await wait_until(lambda: ("tts", "start") in conn.websocket.states(), what="唤醒应答 tts start")
            await wait_until(lambda: len(conn.websocket.frames()) > 0, what="唤醒应答音频")
            await p.close()
    asyncio.run(go())


def test_trim_silence_keeps_speech_with_padding():
    from core.providers.s2s.openai_live.openai_live import _trim_silence
    rate = 24000
    sil = b"\x00\x00" * (rate // 2)                                  # 0.5 秒静音
    voice = (np.ones(rate // 5, np.int16) * 3000).tobytes()            # 0.2 秒有声
    out = _trim_silence(sil + voice + sil, rate)
    ms = len(out) / 2 / rate * 1000
    assert 200 <= ms <= 300
