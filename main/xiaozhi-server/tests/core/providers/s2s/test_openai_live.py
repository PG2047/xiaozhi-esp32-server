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


def test_goodbye_closes_after_playback():
    async def sc(lv, conn, p):
        await lv.emit({"type": "session.input_transcript.delta", "delta": "好了，再见"})
        await lv.emit({"type": "session.output_transcript.delta", "delta": "再见。"})
        await lv.audio(6, 2000)
        await lv.audio(5, 0)
        await wait_until(lambda: conn.closed, timeout=8, what="告别后关连接")
        assert conn.close_after_chat
    run(sc)


def test_goodbye_word_inside_a_question_does_not_close():
    async def sc(lv, conn, p):
        await lv.emit({"type": "session.input_transcript.delta", "delta": "再见用英语怎么说"})
        await lv.emit({"type": "session.output_transcript.delta", "delta": "再见用英语说是 goodbye，也可以说 bye。"})
        await lv.audio(6, 2000)
        await lv.audio(5, 0)
        await wait_until(lambda: ("tts", "stop") in conn.websocket.states(), timeout=8, what="tts stop")
        await _real_sleep(0.2)
        assert not conn.closed and not conn.close_after_chat
    run(sc)


def test_early_audio_buffered_then_drained_in_order_only_for_willing_provider():
    from core.utils.s2s import drain_early_audio, s2s_buffer_early_audio

    class P:
        def __init__(self, wants):
            self.wants_early_audio = wants
            self.got = []

        async def on_pcm(self, pcm):
            self.got.append(pcm)

    class C:
        config = {"selected_module": {"S2S": "OpenAILive"}}
        conn_from_mqtt_gateway = False
        client_is_listening = True
        s2s = None

        def _decode_opus_packet(self, b):
            return b * 2

    async def go():
        c = C()
        c.client_is_listening = False
        assert s2s_buffer_early_audio(c, b"w")           # listen start 之前（唤醒词音频）：吞掉、不攒
        c.client_is_listening = True
        assert s2s_buffer_early_audio(c, b"a") and s2s_buffer_early_audio(c, b"b")
        c._s2s_attach_done = True
        assert not s2s_buffer_early_audio(c, b"c")      # attach 结束后不再攒
        c.s2s = P(True)
        await drain_early_audio(c)
        assert c.s2s.got == [b"aa", b"bb"]
        await drain_early_audio(c)                       # 只补送一次
        assert c.s2s.got == [b"aa", b"bb"]
        c2 = C()
        s2s_buffer_early_audio(c2, b"x")
        c2._s2s_attach_done = True
        c2.s2s = P(False)
        await drain_early_audio(c2)                      # Realtime 这类不要的：丢掉
        assert c2.s2s.got == []
    asyncio.run(go())


def _greeting_case(after_wake):
    async def go():
        lv = FakeLive()
        async with serve(lv.handler, "127.0.0.1", 0) as srv:
            url = f"ws://127.0.0.1:{srv.sockets[0].getsockname()[1]}/v1/live/sessions"
            conn = FakeConn(asyncio.get_running_loop())
            p = S2SProvider({"api_key": "k", "url": url, "echo_gate": False, "idle_close_s": 0}, conn)

            async def fake_tts():
                return (np.ones(conn.sample_rate // 5, np.int16) * 2000).tobytes()
            p._greeting_pcm = fake_tts
            conn.s2s = p
            await p.open()
            task = asyncio.ensure_future(p.on_text("嘿，你好呀"))
            await _real_sleep(0.05)
            await after_wake(lv, p)
            await task
            await _real_sleep(0.2)
            played = ("tts", "start") in conn.websocket.states()
            await p.close()
            return played
    return asyncio.run(go())


def test_greeting_skipped_when_user_keeps_talking():
    async def speak(lv, p):
        await lv.emit({"type": "session.input_transcript.delta", "delta": "一年有"})   # 模型转写出了人话
        await _real_sleep(0.05)
    assert _greeting_case(speak) is False


def test_greeting_plays_when_only_noise_after_wake():
    async def noise(lv, p):
        await p.on_pcm((np.ones(960, np.int16) * 13000).tobytes())   # 很响但不是人声（11-03 复测：13780 的响声让"我在"被跳过）
    assert _greeting_case(noise) is True


def _run_client(scenario, **over):
    async def go():
        lv = FakeLive()
        async with serve(lv.handler, "127.0.0.1", 0) as srv:
            url = f"ws://127.0.0.1:{srv.sockets[0].getsockname()[1]}/v1/live/sessions"
            conn = FakeConn(asyncio.get_running_loop())
            p = S2SProvider({"api_key": "k", "url": url, "echo_gate": False, "idle_close_s": 0, "wake_greeting_text": "",
                             "delegation_mode": "client", **over}, conn)
            conn.s2s = p
            await p.open()
            await scenario(lv, conn, p)
            await p.close()
    asyncio.run(go())


def test_client_mode_session_start_declares_client_delegation():
    async def sc(lv, conn, p):
        d = lv.received[0]["session"]["delegation"]
        assert d == {"type": "client", "acknowledgement_delay_ms": 800}
    _run_client(sc)


def test_client_delegation_short_tool_result_goes_straight_to_live():
    async def sc(lv, conn, p):
        calls = []

        async def fake(items, tools):
            calls.append([dict(i) for i in items])
            assert [t["name"] for t in tools] == ["hass_state"]
            return {"output": [{"type": "function_call", "call_id": "c1", "name": "hass_state", "arguments": "{\"entity_id\": \"light.x\"}"}]}
        p._responses_call = fake
        await lv.emit({"type": "session.input_transcript.delta", "delta": "把客厅灯打开"})
        await _real_sleep(0.05)
        await lv.emit({"type": "session.delegation.created", "delegation": {"id": "d1", "target": "client"}})
        await wait_until(lambda: any(m["type"] == "session.commentary.append" for m in lv.received), what="commentary")
        m = [m for m in lv.received if m["type"] == "session.commentary.append"][0]
        assert m["delegation_id"] == "d1" and m["content"].endswith("结果：客厅灯已打开") and "hass_state" in m["content"]
        assert len(calls) == 1 and "把客厅灯打开" in calls[0][0]["content"]
        assert conn.func_handler.calls[0]["name"] == "hass_state"
    _run_client(sc)


def test_client_delegation_long_tool_result_gets_summarised():
    async def sc(lv, conn, p):
        n = {"i": 0}

        async def fake(items, tools):
            n["i"] += 1
            if n["i"] == 1:
                return {"output": [{"type": "function_call", "call_id": "c1", "name": "hass_state", "arguments": "{}"}]}
            assert items[-1]["type"] == "function_call_output" and items[-2]["type"] == "function_call"
            return {"output": [{"type": "message", "content": [{"type": "output_text", "text": "客厅灯开着。"}]}]}
        p._responses_call = fake
        p.direct_max_chars = 3            # 让"客厅灯已打开"算长结果，走第二轮
        await lv.emit({"type": "session.delegation.created", "delegation": {"id": "d2", "target": "client"}})
        await wait_until(lambda: any(m["type"] == "session.commentary.append" for m in lv.received), what="commentary")
        m = [m for m in lv.received if m["type"] == "session.commentary.append"][0]
        assert m["content"] == "客厅灯开着。" and n["i"] == 2
    _run_client(sc)


def test_client_delegation_long_tool_result_raw_text_goes_to_thinking_first():
    """总结会删细节（农历的宜忌），原文按 400 字切成 thinking 先发，再发 commentary；总长截到 tool_context_max_chars。"""
    async def sc(lv, conn, p):
        n = {"i": 0}
        raw = "宜: " + "祭祀、" * 300          # 903 字

        async def fake(items, tools):
            n["i"] += 1
            if n["i"] == 1:
                return {"output": [{"type": "function_call", "call_id": "c1", "name": "get_lunar", "arguments": "{}"}]}
            return {"output": [{"type": "message", "content": [{"type": "output_text", "text": "今天是农历九月廿五。"}]}]}

        async def run(name, args):
            return raw
        p._responses_call = fake
        p.tools.run = run
        p.context_max_chars = 850
        await lv.emit({"type": "session.delegation.created", "delegation": {"id": "d3", "target": "client"}})
        await wait_until(lambda: any(m["type"] == "session.commentary.append" for m in lv.received), what="commentary")
        sent = [m for m in lv.received if m["type"] in ("session.thinking.append", "session.commentary.append")]
        assert [m["type"] for m in sent] == ["session.thinking.append"] * 3 + ["session.commentary.append"]
        assert all(m["delegation_id"] == "d3" for m in sent)
        assert [len(m["content"]) for m in sent[:3]] == [400, 400, 50]
        assert "".join(m["content"] for m in sent[:3]) == raw[:850]
        assert sent[3]["content"] == "今天是农历九月廿五。"
    _run_client(sc)


def test_client_delegation_short_tool_result_sends_no_thinking():
    async def sc(lv, conn, p):
        async def fake(items, tools):
            return {"output": [{"type": "function_call", "call_id": "c1", "name": "hass_state", "arguments": "{}"}]}
        p._responses_call = fake
        await lv.emit({"type": "session.delegation.created", "delegation": {"id": "d4", "target": "client"}})
        await wait_until(lambda: any(m["type"] == "session.commentary.append" for m in lv.received), what="commentary")
        assert not any(m["type"] == "session.thinking.append" for m in lv.received)
    _run_client(sc)



class FakeVAD:
    """按帧里的幅度给人声概率：幅度 ≥ 1000 的窗口算人声。"""

    def __call__(self, chunk):
        return 0.9 if float(np.abs(chunk).max()) * 32768 >= 1000 else 0.05


def _frame(amp):
    return (np.ones(960, np.int16) * amp).tobytes()


def test_speech_gate_silences_noise_and_keeps_speech_with_lookahead_and_hold():
    from core.providers.s2s.speech_gate import SpeechGate
    g = SpeechGate(FakeVAD(), threshold=0.5, hold_ms=120, lookahead_ms=120)   # 前瞻 2 帧、保持 2 帧
    seq = [500] * 4 + [3000] * 2 + [500] * 6           # 噪声 4 帧 → 人声 2 帧 → 噪声 6 帧
    outs = [g.process(_frame(a)) for a in seq]
    assert outs[:2] == [b"", b""]                       # 前瞻队列没满
    sent = [o for o in outs[2:]]                        # 对应输入第 0～9 帧
    kept = [o != b"\x00" * len(o) for o in sent]
    # silero 每 512 样本一窗、一帧 960 样本：第 6 帧开头还带着第 5 帧剩下的人声样本，也判成人声。
    # 第 0、1 帧离人声还远 → 静音；第 2、3 帧在人声之前 2 帧以内 → 前瞻保留；第 4～6 帧是人声；第 7、8 帧在保持期；第 9 帧静音
    assert kept == [False, False, True, True, True, True, True, True, True, False]
    assert g.stats == {"pass": 7, "silenced": 3}


def test_live_uses_speech_gate_when_enabled(monkeypatch):
    import core.providers.s2s.openai_live.openai_live as mod
    monkeypatch.setattr(mod, "load_vad", lambda path=None: FakeVAD())

    async def go():
        lv = FakeLive()
        async with serve(lv.handler, "127.0.0.1", 0) as srv:
            url = f"ws://127.0.0.1:{srv.sockets[0].getsockname()[1]}/v1/live/sessions"
            conn = FakeConn(asyncio.get_running_loop())
            p = S2SProvider({"api_key": "k", "url": url, "echo_gate": False, "idle_close_s": 0, "wake_greeting_text": "",
                             "speech_gate": True, "speech_gate_lookahead_ms": 0, "speech_gate_hold_ms": 0}, conn)
            conn.s2s = p
            await p.open()
            assert p.speech_gate is not None
            await p.on_pcm(_frame(500))     # 噪声 → 送全零
            await p.on_pcm(_frame(3000))    # 人声 → 原样
            await wait_until(lambda: sum(m["type"] == "session.input_audio.append" for m in lv.received) == 2, what="两帧上行")
            ups = [base64.b64decode(m["audio"]) for m in lv.received if m["type"] == "session.input_audio.append"]
            assert ups[0] == b"\x00" * 1920 and ups[1] == _frame(3000)
            await p.close()
    asyncio.run(go())


def test_greeting_not_skipped_by_resent_wake_audio_through_speech_gate(monkeypatch):
    """设备连上后会把唤醒词音频再发一遍；人声门把它判成人声也不能跳过"我在"（11-03 12:19 真机）。"""
    import core.providers.s2s.openai_live.openai_live as mod
    monkeypatch.setattr(mod, "load_vad", lambda path=None: FakeVAD())

    async def go():
        lv = FakeLive()
        async with serve(lv.handler, "127.0.0.1", 0) as srv:
            url = f"ws://127.0.0.1:{srv.sockets[0].getsockname()[1]}/v1/live/sessions"
            conn = FakeConn(asyncio.get_running_loop())
            p = S2SProvider({"api_key": "k", "url": url, "echo_gate": False, "idle_close_s": 0,
                             "speech_gate": True, "speech_gate_lookahead_ms": 0}, conn)

            async def fake_tts():
                return (np.ones(conn.sample_rate // 5, np.int16) * 2000).tobytes()
            p._greeting_pcm = fake_tts
            conn.s2s = p
            await p.open()
            task = asyncio.ensure_future(p.on_text("嘿，你好呀"))
            await _real_sleep(0.05)
            await p.on_pcm(_frame(3000))       # 重发的唤醒词：人声门判成人声
            await task
            await _real_sleep(0.2)
            assert ("tts", "start") in conn.websocket.states()
            await p.close()
    asyncio.run(go())


def _client_with_decision(decide, picker_over=None):
    """client 委托 + decisions 选工具：decide(ctx, tools) → (工具名, 置信度)；返回 (commentary, responses 调用记录, 工具调用名)。"""
    rec = {"responses": [], "commentary": None}

    async def sc(lv, conn, p):
        async def fake_dec(ctx, tools):
            return decide(ctx, tools)

        async def fake_resp(items, tools, force=None):
            rec["responses"].append({"force": force, "tools": [t["name"] for t in tools]})
            if force:
                return {"output": [{"type": "function_call", "call_id": "c9", "name": force, "arguments": "{\"entity_id\": \"light.x\"}"}]}
            return {"output": [{"type": "message", "content": [{"type": "output_text", "text": "模型自己答了。"}]}]}
        p._decisions_call = fake_dec
        p._responses_call = fake_resp
        if picker_over:
            picker_over(p)
        await lv.emit({"type": "session.delegation.created", "delegation": {"id": "dd", "target": "client"}})
        await wait_until(lambda: any(m["type"] == "session.commentary.append" for m in lv.received), what="commentary")
        rec["commentary"] = [m for m in lv.received if m["type"] == "session.commentary.append"][0]["content"]
        rec["calls"] = [c["name"] for c in conn.func_handler.calls]
    _run_client(sc, tool_picker="decisions")
    return rec


def test_decisions_no_arg_tool_runs_without_responses_call():
    def over(p):
        p._tool_list = lambda: [{"type": "function", "name": "get_lunar", "description": "农历",
                                 "parameters": {"type": "object", "properties": {}}}]
    rec = _client_with_decision(lambda ctx, tools: ("get_lunar", 0.95), over)
    assert rec["responses"] == [] and rec["calls"] == ["get_lunar"] and rec["commentary"].endswith("结果：客厅灯已打开")


def test_decisions_param_tool_forces_only_that_tool():
    rec = _client_with_decision(lambda ctx, tools: ("hass_state", 0.9))
    assert rec["responses"] == [{"force": "hass_state", "tools": ["hass_state"]}] and rec["calls"] == ["hass_state"]


def test_decisions_low_confidence_or_none_falls_back_to_model():
    for decide in (lambda ctx, tools: ("hass_state", 0.45), lambda ctx, tools: (None, 0.9)):
        rec = _client_with_decision(decide)
        assert rec["responses"] == [{"force": None, "tools": ["hass_state"]}]
        assert rec["commentary"] == "模型自己答了。" and rec["calls"] == []


def test_responses_tools_sent_non_strict(monkeypatch):
    """Responses 的工具不写 strict 会按严格模式出参，可选参数全被填上（11-03 空跑：HA intent 补出 floor:""、
    device_class:["switch"]）。强制填参和退回自选两种请求都要带 strict: false，且不改传进来的工具对象。"""
    import aiohttp
    sent = []

    class Reply:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def json(self):
            return {"output": []}

    class Session(Reply):
        def __init__(self, *a, **kw):
            pass

        def post(self, url, json=None, headers=None):
            sent.append(json)
            return Reply()
    monkeypatch.setattr(aiohttp, "ClientSession", Session)
    tool = {"type": "function", "name": "intent__HassTurnOn", "description": "打开设备",
            "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "floor": {"type": "string"}}}}

    async def go():
        p = S2SProvider({"api_key": "k", "delegation_mode": "client", "tool_picker": "decisions", "echo_gate": False,
                         "wake_greeting_text": ""}, FakeConn(asyncio.get_running_loop()))

        async def fake_dec(ctx, tools):
            return "intent__HassTurnOn", 0.9
        p._decisions_call = fake_dec
        items = [{"role": "user", "content": "用户：打开风扇"}]
        await p._decide_first("用户：打开风扇", items, [tool])      # decisions 选中 → 强制只填这一个工具
        await p._responses_call(items, [tool])                     # 退回小模型自选
    asyncio.run(go())
    assert [b["tool_choice"] for b in sent] == [{"type": "function", "name": "intent__HassTurnOn"}, "auto"]
    assert all(b["tools"] == [dict(tool, strict=False)] for b in sent)
    assert "strict" not in tool


def test_dispatch_tools_drops_lunar_dont_call_hint():
    from core.providers.s2s.openai_live.openai_live import S2SProvider as P
    t = [{"name": "get_lunar", "description": "用于农历。对于'今天农历是多少'、'今天农历日期'这样的基本查询，请直接使用context中的信息，不要调用此工具。"}]
    d = P._dispatch_tools(t)[0]["description"]
    assert "不要调用此工具" not in d and "今天的农历日期也用它查" in d


def test_dispatch_instructions_read_ha_devices_from_conn_prompt():
    """HA 设备清单由 hass_init.append_devices_to_prompt 追加到 conn.prompt（func_handler 初始化时，晚于 Live 会话建立）。
    client 委托的调度指令每次现读 conn.prompt，decisions 选工具和小模型填 entity_id 都看得到清单，Live 侧不用另外带。"""
    async def sc(lv, conn, p):
        assert "light.study_lamp" not in p._dispatch_instructions()
        conn.prompt += "\n下面是我家智能设备列表（位置，设备名，entity_id），可以通过homeassistant控制\n北京书房,书房灯,light.study_lamp\n"
        assert "北京书房,书房灯,light.study_lamp" in p._dispatch_instructions()
        assert p.tools.log_args is True          # log_text 打开时工具日志带参数
    _run_client(sc, log_text=True)


class _HAFuncHandler:
    """带 HA MCP 工具的假 func_handler：GetLiveContext 回 ha_names 测试里的虚构设备，其他调用只记下来。"""

    def __init__(self):
        self.calls = []

    def get_functions(self):
        from tests.core.providers.s2s.test_ha_names import HA_TOOLS
        return [{"type": "function", "function": {k: t[k] for k in ("name", "description", "parameters")}} for t in HA_TOOLS]

    async def handle_llm_function_call(self, conn, data):
        from plugins_func.register import Action, ActionResponse
        from tests.core.providers.s2s.test_ha_names import LIVE_RESULT
        self.calls.append(data)
        if data["name"].endswith("GetLiveContext"):
            return ActionResponse(Action.REQLLM, LIVE_RESULT, None)
        return ActionResponse(Action.REQLLM, "{'response_type': 'action_done'}", None)


def _ha_client(scenario, **over):
    from core.providers.s2s.openai_live import openai_live as mod
    mod._HA_CACHE.update(t=0.0, entries=[])            # 进程内缓存，测试之间清掉

    async def sc(lv, conn, p):
        conn.func_handler = _HAFuncHandler()
        await p._load_ha_names()
        await scenario(lv, conn, p)
    _run_client(sc, **over)


async def _delegate(lv, p, said: str, call: dict, did: str) -> str:
    async def fake(items, tools):
        return {"output": [dict(call, type="function_call", call_id="c1")]}
    p._responses_call = fake
    await lv.emit({"type": "session.input_transcript.delta", "delta": said})
    await _real_sleep(0.05)
    await lv.emit({"type": "session.delegation.created", "delegation": {"id": did, "target": "client"}})
    await wait_until(lambda: any(m["type"] == "session.commentary.append" for m in lv.received), what="commentary")
    return [m for m in lv.received if m["type"] == "session.commentary.append"][0]["content"]


def test_ha_names_go_into_dispatch_instructions_and_are_cached():
    """组件就绪后只读调一次 GetLiveContext：调度指令带上规则和名单（Live 主模型的指令不带）；缓存期内别的连接不再调。"""
    async def sc(lv, conn, p):
        d = p._dispatch_instructions()
        assert "- 甲地1号客厅纱帘 / 客厅纱帘 / 甲地客厅纱帘" in d and "不要把地点或房间拆到 area" in d
        assert "甲地1号客厅纱帘" not in lv.received[0]["session"]["instructions"]
        assert [c["name"] for c in conn.func_handler.calls] == ["homeassistant__GetLiveContext"]
        other = S2SProvider({"api_key": "k", "delegation_mode": "client", "echo_gate": False, "wake_greeting_text": ""}, conn)
        await other._load_ha_names()
        assert other._ha_entries == p._ha_entries and len(conn.func_handler.calls) == 1
    _ha_client(sc)


def test_place_guard_blocks_device_in_another_area_before_running_it():
    """用户说乙地，小模型照抄了甲地设备不带地名的别名：不执行，交给 Live 的说明讲清楚。"""
    async def sc(lv, conn, p):
        content = await _delegate(lv, p, "把乙地的除醛净化器关了",
                                  {"name": "intent__HassTurnOff", "arguments": "{\"name\": \"除醛净化器\"}"}, "g1")
        assert "没有执行" in content and "乙地老家" in content
        assert [c["name"] for c in conn.func_handler.calls] == ["homeassistant__GetLiveContext"]
    _ha_client(sc)


def test_same_area_ha_call_runs_with_device_class_moved_out_of_domain():
    async def sc(lv, conn, p):
        await _delegate(lv, p, "把甲地的窗帘都关上",
                        {"name": "intent__HassTurnOff", "arguments": "{\"area\": \"甲地楼下1号\", \"domain\": [\"curtain\"]}"}, "g2")
        call = conn.func_handler.calls[-1]
        assert call["name"] == "intent__HassTurnOff"
        assert json.loads(call["arguments"]) == {"area": "甲地楼下1号", "device_class": ["curtain"]}
    _ha_client(sc)


def test_ha_names_off_keeps_dispatch_and_calls_as_before():
    async def sc(lv, conn, p):
        assert p._ha_entries == [] and "不要把地点或房间拆到 area" not in p._dispatch_instructions()
        await _delegate(lv, p, "把乙地的除醛净化器关了",
                        {"name": "intent__HassTurnOff", "arguments": "{\"name\": \"除醛净化器\"}"}, "g3")
        assert [c["name"] for c in conn.func_handler.calls] == ["intent__HassTurnOff"]
    _ha_client(sc, ha_names=False)
