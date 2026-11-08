"""S2S 读记忆：建会话之前取快照（connection._prepare_memory），只拼进主模型的指令——Live 的 session.start、
Realtime 的 session.update；工具调度的指令不带（它经网关，要求指令不变）。没有快照时指令和以前逐字相同。上游用假的，不出网。"""
from __future__ import annotations

import asyncio
import copy
import json
import time

from websockets.asyncio.server import serve

from tests.core.providers.s2s.test_openai_realtime import (  # noqa: F401  复用夹具（也带上 _restore_sleep 与 Opus 跳过标记）
    FakeConn, _restore_sleep, make_provider, pytestmark, run_scenario,
)
from tests.core.providers.s2s.test_openai_live import FakeLive
from core.providers.s2s.base import MEMORY_HEADER
from core.providers.s2s.openai_live.openai_live import S2SProvider as LiveProvider

MEM = '{"条目": ["最喜欢的水果：芒果"]}'


def _block(text=MEM):
    return f"<memory>\n{MEMORY_HEADER}\n{text}\n</memory>"


# ---------- base.with_memory ----------
def _rt(conn, **over):
    return make_provider(conn, "ws://127.0.0.1:1/v1/realtime", **over)


def test_no_snapshot_leaves_instructions_unchanged():
    conn = FakeConn(None)
    assert _rt(conn).with_memory("你是测试助手") == "你是测试助手"
    conn.memory_snapshot = "   "
    assert _rt(conn).with_memory("你是测试助手") == "你是测试助手"


def test_snapshot_is_appended_when_there_is_no_tag():
    conn = FakeConn(None)
    conn.memory_snapshot = MEM
    assert _rt(conn).with_memory("你是测试助手") == "你是测试助手\n\n" + _block()


def test_snapshot_fills_the_template_tag_in_place():
    conn = FakeConn(None)
    conn.memory_snapshot = MEM
    out = _rt(conn).with_memory("前文\n<memory>\n</memory>\n后文")
    assert out == "前文\n" + _block() + "\n后文"
    assert out.count("<memory>") == 1


def test_backslashes_in_memory_are_kept_verbatim():
    conn = FakeConn(None)
    conn.memory_snapshot = r"路径 C:\new\1 和 \g<0>"
    assert r"C:\new\1 和 \g<0>" in _rt(conn).with_memory("x <memory></memory> y")


def test_long_memory_is_cut_and_zero_turns_it_off():
    conn = FakeConn(None)
    conn.memory_snapshot = "甲" * 50
    p = _rt(conn, memory_max_chars=10)
    assert p.with_memory("x").endswith("甲" * 10 + "…\n</memory>") and p.memory_chars() == 10
    assert _rt(conn, memory_max_chars=0).with_memory("x") == "x"


# ---------- Live：session.start 带，签名和调度指令不带 ----------
def _live(conn, url, **over):
    return LiveProvider({"api_key": "k", "url": url, "echo_gate": False, "idle_close_s": 0,
                         "wake_greeting_text": "", **over}, conn)


def _run_live(check, prompt="你是测试助手", **over):
    async def go():
        lv = FakeLive()
        async with serve(lv.handler, "127.0.0.1", 0) as srv:
            url = f"ws://127.0.0.1:{srv.sockets[0].getsockname()[1]}/v1/live/sessions"
            conn = FakeConn(asyncio.get_running_loop())
            conn.prompt = prompt
            conn.memory_snapshot = MEM
            p = _live(conn, url, **over)
            conn.s2s = p
            await p.open()
            check(lv.received[0]["session"], p)
            await p.close()
    asyncio.run(go())


def test_live_session_start_carries_memory_but_signature_does_not():
    def check(session, p):
        assert session["instructions"] == "你是测试助手\n\n" + _block()
        assert p._session_sig[0] == "你是测试助手"
    _run_live(check)


def test_live_dispatch_instructions_never_carry_memory():
    """模板带了空的 <memory> 标签：主指令里填上记忆，调度指令里那对标签保持为空。"""
    def check(session, p):
        assert "芒果" in session["instructions"]
        d = p._dispatch_instructions()
        assert "芒果" not in d and MEMORY_HEADER not in d
        assert "<memory>\n</memory>" in d
    _run_live(check, prompt="你是测试助手\n<memory>\n</memory>", delegation_mode="client",
              tool_url="http://127.0.0.1:1/v1/responses", tool_api_key="k")


# ---------- Realtime：建会话和之后的同步都带 ----------
def test_realtime_both_session_updates_carry_memory():
    async def sc(rt, conn, url):
        conn.memory_snapshot = MEM
        p = make_provider(conn, url)
        conn.s2s = p
        await p.open()
        (upd,) = await rt.wait_for("session.update")
        assert upd["session"]["instructions"] == "你是测试助手\n\n" + _block()
        conn.prompt = "增强后的提示词\n<memory>\n</memory>"
        await p._refresh_session()
        (_, upd2) = await rt.wait_for("session.update", 2)
        assert upd2["session"]["instructions"] == "增强后的提示词\n" + _block()
        assert p._session_sig[0] == "增强后的提示词\n<memory>\n</memory>"    # 签名是不带记忆的原始指令
        await p.close()
    run_scenario(sc)


# ---------- connection._prepare_memory：在 attach_s2s 之前 ----------
from tests.core.handle.test_s2s_slot import FakeProvider, make_conn  # noqa: E402
import core.connection as connection_mod  # noqa: E402


class FakeMemory:
    def __init__(self, text=MEM, delay=0.0, fail=False):
        self.text, self.delay, self.fail, self.queries, self.role_id = text, delay, fail, 0, None

    def init_memory(self, role_id, llm, **kwargs):
        if self.fail:
            raise KeyError("llm")
        self.role_id = role_id

    def set_llm(self, llm):
        pass

    async def query_memory(self, query):
        self.queries += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.text

    async def save_memory(self, msgs, session_id=None):
        pass


def _init(monkeypatch, memory, s2s="OpenAIRealtime", disabled=False, **block):
    seen = {}

    async def go():
        conn = make_conn(asyncio.get_running_loop())
        conn.memory = memory
        conn.device_id = "dev-a"
        conn.config["selected_module"]["S2S"] = s2s
        conn.config.setdefault("S2S", {}).setdefault("OpenAIRealtime", {}).update(block)
        if disabled:
            conn.memory_disabled = True
        fake = FakeProvider()

        def attach(c):
            seen["snapshot"] = c.memory_snapshot      # open() 时 provider 能看到的
            return fake if s2s else None
        monkeypatch.setattr(connection_mod, "attach_s2s", attach)
        monkeypatch.setattr(conn, "_initialize_tts", lambda: None)
        monkeypatch.setattr(conn, "_initialize_asr", lambda: None)
        monkeypatch.setattr(conn, "_init_prompt_enhancement", lambda: None)
        monkeypatch.setattr(conn, "_inject_tool_call_fewshot", lambda: None)
        t0 = time.monotonic()
        await asyncio.get_running_loop().run_in_executor(None, conn._initialize_components)
        seen["elapsed"] = time.monotonic() - t0
        seen["conn"] = conn
    asyncio.run(go())
    return seen


def test_snapshot_is_ready_before_attach_s2s(monkeypatch):
    mem = FakeMemory()
    seen = _init(monkeypatch, mem)
    assert seen["snapshot"] == MEM
    assert seen["conn"].memory is not mem and seen["conn"].memory.role_id == "dev-a"   # 按连接一份


def test_cascade_connection_does_not_query_at_start(monkeypatch):
    mem = FakeMemory()
    seen = _init(monkeypatch, mem, s2s="")
    assert seen["snapshot"] is None and seen["conn"].memory.queries == 0


def test_registry_memory_false_turns_memory_off_for_the_connection(monkeypatch):
    seen = _init(monkeypatch, FakeMemory(), disabled=True)
    assert seen["snapshot"] is None and seen["conn"].memory is None


def test_slow_memory_does_not_hold_up_the_session(monkeypatch):
    seen = _init(monkeypatch, FakeMemory(delay=2.0), memory_query_timeout_s=0.05)
    assert seen["snapshot"] is None and seen["elapsed"] < 1.5


def test_broken_memory_does_not_break_the_connection(monkeypatch):
    seen = _init(monkeypatch, FakeMemory(fail=True))
    assert seen["snapshot"] is None and seen["conn"].memory is None
    assert isinstance(seen["conn"].s2s, FakeProvider)       # 照样建会话
