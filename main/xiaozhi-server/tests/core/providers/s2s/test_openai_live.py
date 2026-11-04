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


def test_stop_word_in_transcript_while_playing_cuts_like_a_device_abort():
    async def sc(lv, conn, p):
        p.resume_max_wait_s = 0.0
        await lv.audio(4, 2000)
        await wait_until(lambda: p.in_seg, what="进段")
        await lv.emit({"type": "session.input_transcript.delta", "delta": "小智,"})
        await _real_sleep(0.05)
        assert p.in_seg and p.stats["barge_transcript"] == 0     # 只有名字：不截
        await lv.emit({"type": "session.input_transcript.delta", "delta": "停一下"})
        await wait_until(lambda: p.stats["barge_transcript"] == 1, what="停止词截断")
        assert ("tts", "stop") in conn.websocket.states() and p.suppress and not p.in_seg
        assert p.stats["barge_device"] == 0
        n = len(conn.websocket.frames())
        await lv.audio(3, 2000)                                   # 模型还在把那句讲完：丢
        await _real_sleep(0.1)
        assert len(conn.websocket.frames()) == n
    run(sc)


def test_backchannel_negated_stop_and_idle_stop_word_do_not_cut():
    async def sc(lv, conn, p):
        await lv.emit({"type": "session.input_transcript.delta", "delta": "停一下"})   # 设备不在播放
        await _real_sleep(0.05)
        assert p.stats["barge_transcript"] == 0
        await lv.audio(4, 2000)
        await wait_until(lambda: p.in_seg, what="进段")
        for d in ("对对", "嗯,", "闭嘴", "不要停", "别停,接着讲"):   # "闭嘴"：11-03 一声附和"对对"被转写成它
            await lv.emit({"type": "session.input_transcript.delta", "delta": d})
        await _real_sleep(0.1)
        assert p.in_seg and p.stats["barge_transcript"] == 0 and ("tts", "stop") not in conn.websocket.states() and p._stop is None
    run(sc)


def test_stop_words_can_be_turned_off():
    async def sc(lv, conn, p):
        p.stop_words = ()
        await lv.audio(4, 2000)
        await wait_until(lambda: p.in_seg, what="进段")
        await lv.emit({"type": "session.input_transcript.delta", "delta": "停一下"})
        await _real_sleep(0.1)
        assert p.in_seg and p.stats["barge_transcript"] == 0
    run(sc)


LOUD_UP = (np.ones(960, np.int16) * 3000).tobytes()     # 上行 60 ms：用户在说话
QUIET_UP = (np.ones(960, np.int16) * 10).tobytes()


@pytest.fixture(autouse=True)
def _no_ack_tts(monkeypatch):
    """停下应答的 TTS 预取不出网；要测应答的用例自己塞 PCM。"""
    async def none(self):
        return b""
    monkeypatch.setattr(S2SProvider, "_ack_pcm", none)


async def _stop_while_playing(lv, p, said="小智,别说了"):
    p.resume_max_wait_s = 0.0
    p.resume_after_user_s = 0.2
    await lv.audio(4, 2000)
    await wait_until(lambda: p.in_seg, what="进段")
    await lv.emit({"type": "session.output_transcript.delta", "delta": "老乌鸦指了指山顶,说"})
    await lv.emit({"type": "session.input_transcript.delta", "delta": said})
    await wait_until(lambda: p._stop is not None, what="停下")


async def _user_speaks_again(p):
    await _real_sleep(0.65)            # 喊停那句之后静了 0.4 s 以上，也过了进停下后的 0.6 s
    await p.on_pcm(LOUD_UP)            # 用户又开口
    await wait_until(lambda: p._stop is None, what="停下结束")
    await _real_sleep(0.45)
    await p.on_pcm(QUIET_UP)           # 说完（上行静了 0.4 s 以上才算收声）


def _shown(conn):
    return [t.get("text") for t in conn.websocket.texts() if t.get("state") == "sentence_start"]


def test_stop_word_while_playing_holds_until_user_speaks_again():
    async def sc(lv, conn, p):          # 11-03 T1："了指山顶,说"被截，0.9 s 后又放了"那儿有答案。好。"
        await _stop_while_playing(lv, p)
        assert p.stats["barge_transcript"] == 1 and p.suppress and not p.in_seg
        n = len(conn.websocket.frames())
        await lv.audio(3, 0)            # 模型停一下（resume_max_wait 早过了也不放行）
        await lv.audio(3, 2000)         # 把那句讲完、再说"好"：都丢
        await lv.emit({"type": "session.output_transcript.delta", "delta": "那儿有答案。好。"})
        await lv.audio(5, 0)
        await _real_sleep(0.1)
        assert len(conn.websocket.frames()) == n and p.stats["segments"] == 1 and p._stop is not None
        await _user_speaks_again(p)
        await lv.audio(1, 0)
        await lv.audio(3, 2000)         # 对新一句的回答照常放
        await wait_until(lambda: p.stats["segments"] == 2, what="回答段")
        await lv.emit({"type": "session.output_transcript.delta", "delta": "现在九点。"})
        await lv.audio(5, 0)
        await wait_until(lambda: "现在九点。" in _shown(conn), what="回答文字")
        assert not any("答案" in t for t in _shown(conn))       # 停下期间丢掉的话不显示
    run(sc)


def test_stop_word_after_device_abort_holds_instead_of_resuming():
    async def sc(lv, conn, p):          # 11-03 T4：设备 1.37 s 判停，模型没理会"别说了"，又讲了约 20 s
        p.resume_after_user_s = 0.2
        await lv.audio(4, 2000)
        await wait_until(lambda: p.in_seg, what="进段")
        await p.on_pcm(LOUD_UP)
        await p.cancel("device")
        await lv.emit({"type": "session.input_transcript.delta", "delta": "小智,别说了"})
        await wait_until(lambda: p._stop is not None, what="停下")
        assert p.stats["barge_transcript"] == 0     # 设备已经停了，不再截一次
        n = len(conn.websocket.frames())
        await _real_sleep(0.45)
        await p.on_pcm(QUIET_UP)        # 用户说完：原来这里会放行
        await lv.audio(2, 0)
        await lv.audio(4, 2000)         # 模型接着讲故事：丢
        await _real_sleep(0.1)
        assert len(conn.websocket.frames()) == n and p.stats["segments"] == 1
        await _user_speaks_again(p)
        await lv.audio(1, 0)
        await lv.audio(3, 2000)
        await wait_until(lambda: p.stats["segments"] == 2, what="回答段")
    run(sc)


def _false_device_abort(said):
    async def sc(lv, conn, p):          # 误停：转写里没有停止词，照旧接着讲
        p.resume_after_user_s = 0.2
        await lv.audio(4, 2000)
        await wait_until(lambda: p.in_seg, what="进段")
        await p.on_pcm(LOUD_UP)         # 打字 / 附和
        await p.cancel("device")
        if said:
            await lv.emit({"type": "session.input_transcript.delta", "delta": said})
        await _real_sleep(0.45)
        await p.on_pcm(QUIET_UP)
        assert p._stop is None
        await lv.audio(1, 0)
        await lv.audio(3, 2000)         # 模型接着讲
        await wait_until(lambda: p.stats["segments"] == 2, what="接着讲")
    return sc


def test_false_device_abort_by_typing_still_resumes():
    run(_false_device_abort(None))


def test_false_device_abort_by_backchannel_still_resumes():
    run(_false_device_abort("对对"))


def test_stop_word_right_after_a_segment_ends_also_holds():
    async def sc(lv, conn, p):          # 模型听到用户开口先停一下，段已经结束，"停"的转写才到
        await lv.audio(4, 2000)
        await lv.audio(5, 0)
        await wait_until(lambda: ("tts", "stop") in conn.websocket.states() and not p.speaker.speaking, timeout=8,
                         what="段结束、设备播完")
        await lv.emit({"type": "session.input_transcript.delta", "delta": "停一下"})
        await wait_until(lambda: p._stop is not None, what="停下")
        assert p.suppress and p.stats["barge_transcript"] == 0
        await lv.audio(3, 2000)
        await _real_sleep(0.1)
        assert p.stats["segments"] == 1
    run(sc)


def test_stop_plays_cached_ack_once_and_its_echo_does_not_end_the_hold():
    async def sc(lv, conn, p):
        pcm = (np.ones(conn.sample_rate * 3 // 10, np.int16) * 2000).tobytes()   # 0.3 秒
        p.stop_ack_text = "好。"
        p._ack_task = asyncio.ensure_future(asyncio.sleep(0, result=pcm))
        await _stop_while_playing(lv, p)
        n = len(conn.websocket.frames())
        await lv.audio(2, 0)            # 用户已经说完：补一声"好"
        await wait_until(lambda: len(conn.websocket.frames()) > n, what="停下应答")
        await p.on_pcm(LOUD_UP)         # 这声"好"的回声
        await lv.emit({"type": "session.input_transcript.delta", "delta": "停"})   # 同一句里又一个停止词
        await lv.audio(10, 0)
        await _real_sleep(0.3)
        m = len(conn.websocket.frames())
        await lv.audio(10, 0)
        await _real_sleep(0.2)
        assert p._stop is not None and p.stats["stop_holds"] == 1 and len(conn.websocket.frames()) == m
    run(sc)


def test_echo_right_after_the_cut_does_not_end_the_hold():
    async def sc(lv, conn, p):          # 截断时设备还在出声，回声过了 onset_rms 也会记一次"上行开口"
        await _stop_while_playing(lv, p)
        await p.on_pcm(LOUD_UP)
        await _real_sleep(0.05)
        assert p._stop is not None
        await _real_sleep(0.6)
        await p.on_pcm(QUIET_UP)        # 回声过去了
        await p.on_pcm(LOUD_UP)         # 0.6 s 之后再响：算用户开口
        await wait_until(lambda: p._stop is None, what="停下结束")
    run(sc)


def test_ack_is_fetched_again_when_the_first_tts_failed():
    async def sc(lv, conn, p):          # 11-04：重启后第一次取"好。"的 TTS 超时，那个会话就一直没有应答
        pcm = (np.ones(conn.sample_rate * 3 // 10, np.int16) * 2000).tobytes()
        calls = []

        async def flaky():
            calls.append(1)
            return b"" if len(calls) == 1 else pcm
        p.stop_ack_text = "好。"
        p._ack_pcm = flaky
        p._ack_task = asyncio.ensure_future(flaky())
        await _stop_while_playing(lv, p)
        await lv.audio(2, 0)            # 第一次喊停：没有应答可放，再取一次
        await wait_until(lambda: len(calls) == 2, what="重新取 TTS")
        await _user_speaks_again(p)
        await lv.audio(1, 0)
        await lv.audio(3, 2000)
        await wait_until(lambda: p.in_seg, what="回答段")
        p.t_last_input_text -= 2
        n = len(conn.websocket.frames())
        await lv.emit({"type": "session.input_transcript.delta", "delta": "停一下"})   # 第二次喊停
        await wait_until(lambda: p._stop is not None, what="再次停下")
        m = len(conn.websocket.frames())
        await lv.audio(2, 0)
        await wait_until(lambda: len(conn.websocket.frames()) > m, what="这次有应答")
        assert m >= n and len(calls) == 2
    run(sc)


def test_new_user_sentence_ends_the_hold():
    async def sc(lv, conn, p):
        await _stop_while_playing(lv, p)
        p.t_last_input_text -= 2        # 隔了 2 秒，用户又说了一句新的（上行没录到开口）
        await lv.emit({"type": "session.input_transcript.delta", "delta": "讲个笑话吧"})
        await wait_until(lambda: p._stop is None, what="停下结束")
        assert p.stats["stop_holds"] == 1
        await lv.audio(1, 0)
        await lv.audio(3, 2000)
        await wait_until(lambda: p.stats["segments"] == 2, what="回答段")
    run(sc)


def test_question_after_the_stop_word_in_the_same_sentence_gets_answered():
    async def sc(lv, conn, p):          # "小智，等一下，一加一等于几"连着说：停下的同时等这句说完放回答
        await _stop_while_playing(lv, p, said="小智,等一下,")
        assert p.stats["barge_transcript"] == 1
        await lv.emit({"type": "session.input_transcript.delta", "delta": "一加一等于几"})
        await wait_until(lambda: p._stop is None, what="停下结束")
        await lv.audio(1, 0)
        await lv.audio(3, 2000)         # 回答照常放
        await wait_until(lambda: p.stats["segments"] == 2, what="回答段")
    run(sc)


def test_question_in_the_same_fragment_as_the_stop_word_gets_answered():
    async def sc(lv, conn, p):
        p.resume_max_wait_s = 0.0
        await lv.audio(4, 2000)
        await wait_until(lambda: p.in_seg, what="进段")
        await lv.emit({"type": "session.input_transcript.delta", "delta": "等一下,一加一等于几"})
        await wait_until(lambda: p.stats["barge_transcript"] == 1, what="截断旧内容")
        assert p._stop is None and p.stats["stop_holds"] == 1
        await lv.audio(1, 0)
        await lv.audio(3, 2000)
        await wait_until(lambda: p.stats["segments"] == 2, what="回答段")
    run(sc)


def test_question_after_the_stop_word_following_a_device_abort_gets_answered():
    async def sc(lv, conn, p):
        p.resume_after_user_s = 0.2
        await lv.audio(4, 2000)
        await wait_until(lambda: p.in_seg, what="进段")
        await p.on_pcm(LOUD_UP)
        await p.cancel("device")
        await lv.emit({"type": "session.input_transcript.delta", "delta": "小智,停一下,"})
        await wait_until(lambda: p._stop is not None, what="停下")
        await lv.emit({"type": "session.input_transcript.delta", "delta": "现在几点"})
        await wait_until(lambda: p._stop is None, what="停下结束")
        await lv.audio(2, 0)
        await lv.audio(3, 2000)         # 用户还没说完：先丢
        await _real_sleep(0.1)
        assert p.stats["segments"] == 1
        await _real_sleep(0.45)
        await p.on_pcm(QUIET_UP)        # 说完
        await lv.audio(1, 0)
        await lv.audio(3, 2000)         # 回答
        await wait_until(lambda: p.stats["segments"] == 2, what="回答段")
    run(sc)


def test_stop_word_with_only_fillers_still_holds():
    async def sc(lv, conn, p):
        await _stop_while_playing(lv, p, said="小智,停一下")
        await lv.emit({"type": "session.input_transcript.delta", "delta": "吧。"})
        await lv.audio(1, 0)
        await lv.audio(3, 2000)
        await _real_sleep(0.1)
        assert p._stop is not None and p.stats["segments"] == 1
        assert not p._has_more("一下") and not p._has_more("一下好吗") and not p._has_more("下来")
        assert not p._has_more("一点") and not p._has_more("一下,谢谢") and not p._has_more(",小智,停一下")
        assert p._has_more(",一加一等于几") and p._has_more("一下,现在几点") and p._has_more("一下我问你个问题")
    run(sc)


async def _segment_just_ended(lv, conn, p):
    await lv.audio(4, 2000)
    await lv.audio(5, 0)
    await wait_until(lambda: ("tts", "stop") in conn.websocket.states() and not p.speaker.speaking, timeout=8,
                     what="段结束、设备播完")


def test_story_request_heard_as_yao_ting_does_not_stop():
    async def sc(lv, conn, p):          # 11-04 00:35："给我讲一个两分钟的故事，不要停"被转写成"什么故事?要停"
        await _segment_just_ended(lv, conn, p)
        await lv.emit({"type": "session.input_transcript.delta", "delta": "什么故事?要停"})
        await _real_sleep(0.1)
        assert p._stop is None and not p.suppress
        await lv.audio(4, 2000)         # 模型讲新故事：照常放
        await wait_until(lambda: p.stats["segments"] == 2, what="新故事")
        p.t_last_input_text -= 2
        await lv.emit({"type": "session.input_transcript.delta", "delta": "什么故事?要停"})   # 模型在播时也不算
        await _real_sleep(0.1)
        assert p._stop is None and p.in_seg
    run(sc)


def test_bare_ting_right_after_a_segment_does_not_stop_but_ting_yixia_does():
    async def sc(lv, conn, p):
        await _segment_just_ended(lv, conn, p)
        await lv.emit({"type": "session.input_transcript.delta", "delta": "小智,停"})
        await _real_sleep(0.1)
        assert p._stop is None
        p.t_last_input_text -= 2
        await lv.emit({"type": "session.input_transcript.delta", "delta": "小智,停一下"})
        await wait_until(lambda: p._stop is not None, what="停下")
        assert p._stop_word_at("小智,停一下", bare_ting=False) == ("停", 4)
        assert p._stop_word_at("停下来", bare_ting=False)[0] == "停" and p._stop_word_at("小智,停", bare_ting=False) is None
        assert p._stop_word_at("什么故事?要停") is None and p._stop_word_at("小智,停") == ("停", 4)
    run(sc)


def test_bare_ting_while_playing_still_stops():
    async def sc(lv, conn, p):
        await _stop_while_playing(lv, p, said="小智,停")
        assert p.stats["barge_transcript"] == 1
    run(sc)


def test_heard_as_listen_counts_as_stop():
    async def sc(lv, conn, p):          # 11-03 23:37：Mac 17 外放三次"停一下"有两次被转写成"听一下"
        await _stop_while_playing(lv, p, said="小智,听一下")
        assert p.stats["barge_transcript"] == 1
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


def test_dispatch_tools_explains_ha_position_direction_without_touching_input():
    """HA 意图工具的 position 没写哪头是开，小模型把"把纱帘拉上"填成 100（HA 里 100 是全开）：补上方向。
    顶层说明不变（decisions 只看它），已有说明不覆盖，非 HA 工具和没有 position 的工具不动，传进来的工具对象不改。"""
    import copy

    from core.providers.s2s.openai_live.openai_live import S2SProvider as P
    pos = {"type": "integer", "minimum": 0, "maximum": 100}
    tools = [
        {"type": "function", "name": "intent__HassSetPosition", "description": "Sets the position of a device or entity",
         "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "position": dict(pos)},
                        "required": ["position"]}},
        {"type": "function", "name": "cover__HassSetPosition", "description": "Sets the position",
         "parameters": {"type": "object", "properties": {"position": dict(pos, description="已有说明")}}},
        {"type": "function", "name": "play_music", "description": "播放音乐",
         "parameters": {"type": "object", "properties": {"position": {"type": "integer"}}}},
        {"type": "function", "name": "intent__HassTurnOff", "description": "Turns off/closes a device or entity",
         "parameters": {"type": "object", "properties": {"name": {"type": "string"}}}},
    ]
    before = copy.deepcopy(tools)
    out = P._dispatch_tools(tools)
    assert tools == before
    got = out[0]["parameters"]["properties"]["position"]
    assert got == dict(pos, description="位置百分比：0 是全关（拉上、合上），100 是全开（拉开）。")
    assert out[0]["description"] == before[0]["description"]
    assert out[0]["parameters"]["required"] == ["position"] and out[0]["parameters"]["properties"]["name"] == {"type": "string"}
    assert out[1:] == before[1:]


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


async def _delegate(lv, p, said: str | None, call: dict, did: str) -> str:
    """said=None：委托时这句转写还没到（in_text 是空的）。"""
    async def fake(items, tools):
        return {"output": [dict(call, type="function_call", call_id="c1")]}
    p._responses_call = fake
    if said is not None:
        await lv.emit({"type": "session.input_transcript.delta", "delta": said})
        await _real_sleep(0.05)
    await lv.emit({"type": "session.delegation.created", "delegation": {"id": did, "target": "client"}})
    def mine():
        return [m for m in lv.received if m["type"] == "session.commentary.append" and m["delegation_id"] == did]
    await wait_until(lambda: mine(), what="commentary")
    return mine()[0]["content"]


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


def test_place_guard_only_reads_what_user_said_since_last_delegation():
    """上一轮（上次委托之前）提到的地方不算：这一轮没说地点、或者转写还没到，就不拦；
    这期间落进 history 的用户发言（Live 先开口时会把转写刷进去）照样算。"""
    off = "intent__HassTurnOff"

    async def sc(lv, conn, p):
        p.history += [("user", "把乙地的风扇打开"), ("assistant", "好了。")]
        p._guard_mark = len(p.history)                      # 上一次委托在这之后结束
        await _delegate(lv, p, None, {"name": off, "arguments": "{\"name\": \"甲地除醛净化器\"}"}, "h1")
        assert conn.func_handler.calls[-1]["name"] == off
    _ha_client(sc)

    async def sc2(lv, conn, p):
        p.history += [("user", "把乙地的风扇打开"), ("assistant", "好了。")]
        p._guard_mark = len(p.history)
        await _delegate(lv, p, "把甲地的除醛净化器关了", {"name": off, "arguments": "{\"name\": \"除醛净化器\"}"}, "h2")
        assert conn.func_handler.calls[-1]["name"] == off
    _ha_client(sc2)

    async def sc3(lv, conn, p):
        p.history += [("user", "把乙地的除醛净化器关了"), ("assistant", "好，我来关。")]   # 上次委托之后才落进来
        content = await _delegate(lv, p, None, {"name": off, "arguments": "{\"name\": \"除醛净化器\"}"}, "h3")
        assert "没有执行" in content and [c["name"] for c in conn.func_handler.calls] == ["homeassistant__GetLiveContext"]
        assert p._guard_mark == len(p.history)
    _ha_client(sc3)


def test_ha_names_wait_for_tool_handler_init():
    """工具处理器在后台初始化，组件就绪时 HA 的 MCP 工具常常还没挂上（finish_init=False、清单里没有 GetLiveContext）：
    要等它初始化完再取名单（11-03 部署后探针：没等，名单一直没取到）。"""
    from core.providers.s2s.openai_live import openai_live as mod
    mod._HA_CACHE.update(t=0.0, entries=[])

    async def sc(lv, conn, p):
        fh = _HAFuncHandler()
        fh.finish_init = False
        full = fh.get_functions
        fh.get_functions = lambda: full() if fh.finish_init else full()[:1]     # 初始化完之前只有插件工具
        conn.func_handler = fh

        async def init_done():
            await _real_sleep(0.3)
            fh.finish_init = True
        asyncio.ensure_future(init_done())
        await p._load_ha_names()
        assert p._ha_entries and [c["name"] for c in fh.calls] == ["homeassistant__GetLiveContext"]
    _run_client(sc)


def test_area_only_call_hitting_several_devices_asks_unless_user_said_all():
    """没填 name、会一次控制甲地的风扇和窗帘两台：用户没说"都"就不执行、回问；说了"都"照常执行。"""
    off = "intent__HassTurnOff"

    async def sc(lv, conn, p):
        content = await _delegate(lv, p, "把甲地的关了", {"name": off, "arguments": "{\"area\": \"甲地楼下1号\"}"}, "a1")
        assert "没有执行" in content and "2 台" in content
        assert [c["name"] for c in conn.func_handler.calls] == ["homeassistant__GetLiveContext"]
    _ha_client(sc)

    async def sc2(lv, conn, p):
        await _delegate(lv, p, "把甲地的都关了", {"name": off, "arguments": "{\"area\": \"甲地楼下1号\"}"}, "a2")
        assert conn.func_handler.calls[-1]["name"] == off
    _ha_client(sc2)


def _name_client(scenario):
    """设备歧义用的名单（四幅窗帘等，见 test_ha_names._name_entries）。"""
    from tests.core.providers.s2s.test_ha_names import _name_entries

    async def sc(lv, conn, p):
        p._ha_entries = _name_entries()
        await scenario(lv, conn, p)
    _ha_client(sc)


async def _ask_round(lv, p, said: str, name: str, did: str) -> str:
    """用户说完、Live 先开口（转写已落进 history）再委托；交回后 Live 接着说的话也记进 history，和真实对话的顺序一样。"""
    p.history.append(("user", said))
    content = await _delegate(lv, p, None, {"name": "intent__HassTurnOff", "arguments": json.dumps({"name": name}, ensure_ascii=False)}, did)
    p.history.append(("assistant", "你要控制哪一台？"))
    return content


def _ran(conn) -> list[str]:
    return [json.loads(c["arguments"])["name"] for c in conn.func_handler.calls if c["name"] == "intent__HassTurnOff"]


def test_named_call_asks_until_user_tells_the_device_apart():
    """小模型替用户挑了一幅：不执行、回问。用户答了半截（"客厅的"），只问剩下分不开的；再答"布帘"才执行。"""
    async def sc(lv, conn, p):
        content = await _ask_round(lv, p, "把戊地的窗帘关上", "戊地1号卧室布帘", "n1")
        assert "没有执行" in content and "4 台" in content and _ran(conn) == []
        content = await _ask_round(lv, p, "客厅的", "戊地1号客厅布帘", "n2")
        assert "没有执行" in content and "2 台" in content and "戊地1号客厅纱帘" in content and "卧室" not in content
        assert _ran(conn) == []
        await _ask_round(lv, p, "布帘", "戊地1号客厅布帘", "n3")
        assert _ran(conn) == ["戊地1号客厅布帘"]
    _name_client(sc)


def test_answer_with_words_not_in_names_is_not_asked_twice_and_ask_does_not_carry_over():
    """用户答的词名单里没有（"第一个"）：还是那一批设备，不再问，照小模型选的执行。放行一次就用掉了，接着再来一句说不清的照样要问。"""
    async def sc(lv, conn, p):
        assert "没有执行" in await _ask_round(lv, p, "把戊地的窗帘关上", "戊地1号卧室布帘", "o1")
        await _ask_round(lv, p, "第一个", "戊地1号卧室布帘", "o2")
        assert _ran(conn) == ["戊地1号卧室布帘"]
        assert "没有执行" in await _ask_round(lv, p, "再来一个", "戊地1号客厅纱帘", "o3")
        assert _ran(conn) == ["戊地1号卧室布帘"]
    _name_client(sc)


def test_answer_with_device_words_is_still_judged_by_those_words():
    """用户答了"客厅的"，小模型却选了卧室那幅：还是那一批设备，但这句话里有设备的词，不能放行。"""
    async def sc(lv, conn, p):
        assert "4 台" in await _ask_round(lv, p, "把戊地的窗帘关上", "戊地1号卧室布帘", "w1")
        assert "4 台" in await _ask_round(lv, p, "客厅的", "戊地1号卧室布帘", "w2")
        assert _ran(conn) == []
    _name_client(sc)


def test_repeating_the_request_after_an_unanswered_ask_asks_again():
    """回问之后用户没答（没有委托，上次问的设备还记着），过一会儿又说了一遍：照样问，不能当成答过了。"""
    async def sc(lv, conn, p):
        assert "4 台" in await _ask_round(lv, p, "把戊地的窗帘关上", "戊地1号卧室布帘", "r1")
        assert "4 台" in await _ask_round(lv, p, "把戊地的窗帘关上", "戊地1号卧室布帘", "r2")
        assert _ran(conn) == []
    _name_client(sc)


def test_answer_without_device_words_reads_only_the_last_sentence():
    """委托结束后转写才落进 history 时，下一次委托看得到上一句（带"窗帘"）：判设备词只看最后这句，"第一个"照样放行。"""
    off = "intent__HassTurnOff"

    async def sc(lv, conn, p):
        call = {"name": off, "arguments": json.dumps({"name": "戊地1号卧室布帘"}, ensure_ascii=False)}
        assert "4 台" in await _delegate(lv, p, "把戊地的窗帘关上", call, "l1")
        await p._flush_input_text()
        await _delegate(lv, p, "第一个", call, "l2")
        assert _ran(conn) == ["戊地1号卧室布帘"]
    _name_client(sc)


def test_place_guard_still_blocks_other_area_right_after_an_ask():
    """回问之后用户换了个地方：设备歧义看得到上一句，地点守卫不看，否则上一句的地点会把跨房子的误控放过去。"""
    async def sc(lv, conn, p):
        assert "4 台" in await _ask_round(lv, p, "把戊地的窗帘关上", "戊地1号卧室布帘", "p1")
        content = await _ask_round(lv, p, "把己地的大净化器关了", "大净化器", "p2")
        assert "不是同一个地方" in content and "己地老家" in content and _ran(conn) == []
    _name_client(sc)


def test_two_named_calls_in_one_delegation_are_both_asked():
    """小模型一轮填了两幅：同一次委托里不能因为第一个刚问过，就把第二个放过去。"""
    off = "intent__HassTurnOff"

    async def sc(lv, conn, p):
        calls = [{"type": "function_call", "call_id": f"c{i}", "name": off, "arguments": json.dumps({"name": n}, ensure_ascii=False)}
                 for i, n in enumerate(("戊地1号卧室布帘", "戊地1号客厅布帘"))]
        rounds = [{"output": calls}, {"output": [{"type": "message", "content": [{"type": "output_text", "text": "要关哪一幅？"}]}]}]

        async def fake(items, tools):
            return rounds.pop(0)
        p._responses_call = fake
        p.history.append(("user", "把戊地的窗帘关上"))
        await lv.emit({"type": "session.delegation.created", "delegation": {"id": "t1", "target": "client"}})
        await wait_until(lambda: any(m["type"] == "session.commentary.append" for m in lv.received), what="commentary")
        assert _ran(conn) == []
    _name_client(sc)


def test_pronoun_after_naming_the_device_is_not_asked_and_in_list_tools_are_untouched():
    """前一句说过是哪一台（查状态也算），接着说"把它关了"不用问。hass_set_state 不过这些检查。"""
    async def sc(lv, conn, p):
        p.history += [("user", "戊地客厅的纱帘开着吗"), ("assistant", "开着呢。")]
        p._guard_mark = len(p.history)
        await _delegate(lv, p, "把它关了", {"name": "intent__HassTurnOff", "arguments": "{\"name\": \"戊地1号客厅纱帘\"}"}, "q1")
        assert _ran(conn) == ["戊地1号客厅纱帘"]
    _name_client(sc)

    async def sc2(lv, conn, p):
        await _delegate(lv, p, "把灯关了", {"name": "hass_set_state", "arguments": "{\"entity_id\": \"light.x\"}"}, "q2")
        assert conn.func_handler.calls[-1]["name"] == "hass_set_state"
    _name_client(sc2)


def test_close_writes_status_ledger_without_device_identity(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["app.py"])

    async def sc(lv, conn, p):
        conn.config = dict(getattr(conn, "config", None) or {}, log={"status_dir": str(tmp_path)})
        conn.device_name = "书房"
        conn.device_id = "aa:bb:cc:dd:ee:ff"
        p.stats["segments"] = 2
    run(sc)
    row = json.loads((tmp_path / "sessions.jsonl").read_text(encoding="utf-8"))
    assert row["device"] == "书房" and row["model"] == "gpt-live-1"
    assert row["stats"]["segments"] == 2 and row["duration_s"] >= 0
    assert "aa:bb:cc:dd:ee:ff" not in json.dumps(row)


def test_probe_process_does_not_write_status_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["-"])    # docker exec python - 跑的探针

    async def sc(lv, conn, p):
        conn.config = dict(getattr(conn, "config", None) or {}, log={"status_dir": str(tmp_path)})
    run(sc)
    assert not (tmp_path / "sessions.jsonl").exists()


_LIGHTS_PROMPT = ("你是小智\n下面是我家智能设备列表（位置，设备名，entity_id），可以通过homeassistant控制\n"
                  "甲地书房,书房灯,light.a1\n甲地客厅,客厅大灯,light.a2\n甲地1号,小台灯,light.a3\n老家卧室,卧室灯,light.b1\n")


def _light_call(entity: str, call_id: str = "c0", state: str = "turn_off", name: str = "hass_set_state") -> dict:
    return {"type": "function_call", "call_id": call_id, "name": name,
            "arguments": json.dumps({"state": {"type": state}, "entity_id": entity})}


def _after_first_call(said: str | list[str], auto_rounds: list, first: dict | list | None = None, tool: str = "hass_set_state",
                      history: list | None = None, conf: float = 0.95) -> dict:
    """用户说 said（列表是一句一次委托），decisions 选 tool、强制填参填 first（默认只填甲地书房灯一个）；之后小模型每次自选按 auto_rounds
    依次回（多要一次会报错）。返回最后交回 Live 的话、依次执行的调用、小模型每次自选时看到的工具结果数。名单和设备列表用虚构的甲地、老家。"""
    from tests.core.providers.s2s.test_ha_names import entries
    rec = {"auto": []}

    async def sc(lv, conn, p):
        rounds = list(auto_rounds)

        async def fake_dec(ctx, tools):
            return tool, conf

        async def fake_resp(items, tools, force=None):
            if force:
                return {"output": first if isinstance(first, list) else [first or _light_call("light.a1")]}
            rec["auto"].append(sum(1 for it in items if it.get("type") == "function_call_output"))
            return {"output": rounds.pop(0)}
        p._decisions_call = fake_dec
        p._responses_call = fake_resp
        p._tool_list = lambda: [{"type": "function", "name": tool, "description": "设置设备状态",
                                 "parameters": {"type": "object", "properties": {"entity_id": {"type": "string"}, "state": {"type": "object"}}}}]
        p._ha_entries = entries()
        conn.prompt = _LIGHTS_PROMPT
        p.history += history or []
        p._guard_mark = len(p.history)
        for n, s in enumerate([said] if isinstance(said, str) else said):
            await lv.emit({"type": "session.input_transcript.delta", "delta": s})
            await _real_sleep(0.05)
            await lv.emit({"type": "session.delegation.created", "delegation": {"id": f"l{n}", "target": "client"}})
            await wait_until(lambda: any(m["type"] == "session.commentary.append" and m["delegation_id"] == f"l{n}" for m in lv.received),
                             what="commentary")
            await p._flush_input_text()                 # 交回之后 Live 才开口：这句落在委托标记之后，下一次委托的 said 里也有它
            p.history.append(("assistant", "好的。"))
        rec["commentary"] = [m for m in lv.received if m["type"] == "session.commentary.append"][-1]["content"]
        rec["calls"] = [json.loads(c["arguments"]) for c in conn.func_handler.calls]
    _run_client(sc, tool_picker="decisions")
    return rec


def _text_reply(text: str) -> list:
    return [{"type": "message", "content": [{"type": "output_text", "text": text}]}]


def test_all_words_with_a_place_fan_out_to_every_light_there():
    """"把甲地的灯全关了"：强制填参只出第一盏，服务端按设备列表换成甲地的每一盏，动作照小模型填的，结果都给小模型总结。"""
    for said in ("把甲地的灯全关了", "把甲地的灯都关了", "把甲地所有的灯关了", "把甲地的灯全部关掉"):
        rec = _after_first_call(said, [_text_reply("甲地的灯都关了。")])
        assert [c["entity_id"] for c in rec["calls"]] == ["light.a1", "light.a2", "light.a3"], said
        assert {c["state"]["type"] for c in rec["calls"]} == {"turn_off"}
        assert rec["auto"] == [3] and rec["commentary"] == "甲地的灯都关了。"


def test_fan_out_keeps_the_filled_action_and_replaces_a_wrong_first_light():
    """动作（调亮度）跟小模型填的走；小模型第一盏填到别处或者抄错 entity_id，也不执行它，只控制用户说的那一处。"""
    first = {"type": "function_call", "call_id": "c0", "name": "hass_set_state",
             "arguments": json.dumps({"state": {"type": "brightness_value", "input": 30}, "entity_id": "light.b1"})}
    rec = _after_first_call("把甲地的灯都调到百分之三十", [_text_reply("调好了。")], first=first)
    assert [c["entity_id"] for c in rec["calls"]] == ["light.a1", "light.a2", "light.a3"]
    assert all(c["state"] == {"type": "brightness_value", "input": 30} for c in rec["calls"])


def test_fan_out_with_one_light_there_goes_straight_to_live():
    """那一处只有一盏：展开后还是一个调用，照旧直接交回。"""
    rec = _after_first_call("把老家的灯都关了", [], first=_light_call("light.b1"))
    assert [c["entity_id"] for c in rec["calls"]] == ["light.b1"] and rec["auto"] == []
    assert rec["commentary"].startswith("已调用 hass_set_state")


def test_fan_out_takes_the_place_from_earlier_sentences_when_this_one_has_none():
    rec = _after_first_call("都关了吧", [_text_reply("都关了。")],
                            history=[("user", "甲地有哪些灯"), ("assistant", "书房灯、客厅大灯和小台灯。")])
    assert [c["entity_id"] for c in rec["calls"]] == ["light.a1", "light.a2", "light.a3"]


def test_no_fan_out_with_a_named_device_or_without_all_words():
    """点了具体设备（"都"不是全部的意思）、没说"都"：还是一个调用、直接交回，不多走一轮小模型。"""
    for said in ("都这么晚了，把甲地的书房灯关了", "把甲地的灯关了", "把甲地的书房灯亮度调到百分之五十"):
        rec = _after_first_call(said, [])
        assert [c["entity_id"] for c in rec["calls"]] == ["light.a1"] and rec["auto"] == [], said
        assert rec["commentary"].startswith("已调用 hass_set_state") and rec["commentary"].endswith("结果：客厅灯已打开")


def test_all_words_without_a_place_ask_where_and_run_nothing():
    """"把灯全关了"：不替用户控制几处房子的灯，什么都不执行，回问交给 Live，不多走一轮小模型。"""
    rec = _after_first_call("把灯全关了", [])
    assert rec["calls"] == [] and rec["auto"] == []
    assert "没有执行" in rec["commentary"] and "甲地楼下1号、乙地老家" in rec["commentary"] and "问用户要控制哪一处" in rec["commentary"]


def test_several_calls_without_a_place_are_asked_too():
    """置信度低、小模型自己一轮出了几盏：也不执行，它看到"没有执行"后说的话交给 Live。"""
    two = [_light_call("light.a1", "c0"), _light_call("light.b1", "c1")]
    rec = _after_first_call("把灯全关了", [_text_reply("要关哪里的灯？")], first=two)
    assert rec["calls"] == [] and rec["auto"] == [2] and rec["commentary"] == "要关哪里的灯？"


def test_answer_with_a_place_after_asking_controls_that_place():
    rec = _after_first_call(["把灯全关了", "甲地的"], [_text_reply("甲地的灯都关了。")])
    assert [c["entity_id"] for c in rec["calls"]] == ["light.a1", "light.a2", "light.a3"]
    assert rec["commentary"] == "甲地的灯都关了。"


def test_answer_all_after_asking_controls_everything():
    rec = _after_first_call(["把灯全关了", "都关了吧"], [_text_reply("都关了。")])
    assert [c["entity_id"] for c in rec["calls"]] == ["light.a1", "light.a2", "light.a3", "light.b1"]


def test_named_device_after_asking_controls_only_that_one():
    """答了具体设备：上一句的"全关"不算这一句的，照小模型填的执行一个。"""
    rec = _after_first_call(["把灯全关了", "书房灯"], [])
    assert [c["entity_id"] for c in rec["calls"]] == ["light.a1"]
    assert rec["commentary"].startswith("已调用 hass_set_state") and "没有执行" not in rec["commentary"]


def test_vague_answer_after_asking_is_asked_again_and_runs_nothing():
    """没答上来（"嗯"）：小模型一轮出好几个调用也不执行，再问一遍；接着答了地点还能接上。"""
    two = [_light_call("light.a1", "c0"), _light_call("light.b1", "c1")]
    rec = _after_first_call(["把灯全关了", "嗯"], [_text_reply("要关哪里的灯？"), _text_reply("是哪里的灯？")], first=two)
    assert rec["calls"] == [] and rec["commentary"] == "是哪里的灯？"
    rec = _after_first_call(["把灯全关了", "第一个"], [])
    assert rec["calls"] == [] and "没有执行" in rec["commentary"]
    rec = _after_first_call(["把灯全关了", "第一个", "甲地的"], [_text_reply("甲地的灯都关了。")])
    assert [c["entity_id"] for c in rec["calls"]] == ["light.a1", "light.a2", "light.a3"]


def test_asking_only_counts_for_the_next_delegation():
    """问完隔了一次别的委托，再说地点就不算答这个问题了。"""
    rec = _after_first_call(["把灯全关了", "书房灯", "甲地的"], [])
    assert [c["entity_id"] for c in rec["calls"]] == ["light.a1", "light.a1"]


def test_all_words_of_the_previous_sentence_do_not_fan_out_this_one():
    """上一句全关了甲地，这句没说"都"：只照小模型填的执行一个。"""
    rec = _after_first_call(["把甲地的灯全关了", "把它打开"], [_text_reply("都关了。")])
    assert [c["entity_id"] for c in rec["calls"]] == ["light.a1", "light.a2", "light.a3", "light.a1"]


def test_low_confidence_set_state_is_trusted_only_when_it_will_fan_out():
    """会展开的句子：decisions 选了 hass_set_state，置信度低也用它，不退回小模型自选（那里可能只填区域）。"""
    rec = _after_first_call("把甲地的灯全关了", [_text_reply("都关了。")], conf=0.4)
    assert [c["entity_id"] for c in rec["calls"]] == ["light.a1", "light.a2", "light.a3"] and rec["auto"] == [3]
    # 不展开的句子照旧退回小模型自选：自选的结果交给它，第一轮就要自选一次（auto 里的 0）
    for said, ran in (("把甲地的书房灯关了", ["light.a2"]), ("把甲地的灯关了", ["light.a2"]), ("把灯全关了", [])):
        rec = _after_first_call(said, [[_light_call("light.a2")]], conf=0.4)
        assert [c["entity_id"] for c in rec["calls"]] == ran and rec["auto"] == [0], said
    # 选的是别的工具：置信度低还是退回小模型
    rec = _after_first_call("把甲地的灯全关了", [_text_reply("没听清。")], tool="intent__HassTurnOff", conf=0.4)
    assert rec["calls"] == [] and rec["auto"] == [0] and rec["commentary"] == "没听清。"


def test_short_answer_with_low_confidence_after_asking_still_fans_out():
    """回问之后只答地名时 decisions 的置信度常不到 0.6：第一句（没地点）照旧退回小模型自选，第二句（答了地点）信 hass_set_state。"""
    rec = _after_first_call(["把灯全关了", "甲地"], [[_light_call("light.a2")], _text_reply("甲地的灯都关了。")], conf=0.4)
    assert [c["entity_id"] for c in rec["calls"]] == ["light.a1", "light.a2", "light.a3"]
    assert rec["auto"] == [0, 3] and rec["commentary"] == "甲地的灯都关了。"


def test_fan_out_only_in_the_first_round():
    """总结那一轮小模型又调了一次：照它填的执行，不再展开一遍。"""
    rec = _after_first_call("把甲地的灯都关了", [[_light_call("light.a2", "c9")]])
    assert [c["entity_id"] for c in rec["calls"]] == ["light.a1", "light.a2", "light.a3", "light.a2"] and rec["auto"] == [3]


def test_no_fan_out_when_the_model_already_gave_several_calls():
    """小模型一轮出了几个调用：是它自己挑的，照它填的执行。"""
    rec = _after_first_call("把甲地的灯都关了", [_text_reply("关了。")], first=[_light_call("light.a1", "c0"), _light_call("light.a2", "c1")])
    assert [c["entity_id"] for c in rec["calls"]] == ["light.a1", "light.a2"] and rec["auto"] == [2]


def test_no_fan_out_for_other_tools():
    """别的工具（同样是短结果）照旧：说了"都"也是一个调用、直接交回。"""
    rec = _after_first_call("甲地都几点了", [], first=_light_call("light.a1", name="llm__GetDateTime"), tool="llm__GetDateTime")
    assert len(rec["calls"]) == 1 and rec["auto"] == [] and rec["commentary"].startswith("已调用 llm__GetDateTime")
