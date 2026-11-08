"""连接层的记忆：每条连接一份记忆实例（服务启动时那份是所有连接共用的），关闭时只把真实对话交给总结。"""
from __future__ import annotations

import asyncio
import copy

from tests.conftest import CONFIG
from core.utils.cache.manager import CacheType, cache_manager

cache_manager.set(CacheType.CONFIG, "main_config", CONFIG)

from config.logger import setup_logging  # noqa: E402

setup_logging()

import core.connection as connection_mod  # noqa: E402
from core.connection import ConnectionHandler  # noqa: E402
from core.providers.memory.mem_local_short.mem_local_short import MemoryProvider  # noqa: E402
from core.utils.dialogue import WAKE_GREETING_QUERY, Message  # noqa: E402


class FakeMemory:
    def __init__(self):
        self.role_id, self.saved = None, []

    def init_memory(self, role_id, llm, **kwargs):
        self.role_id = role_id

    def set_llm(self, llm):
        pass

    async def save_memory(self, msgs, session_id=None):
        self.saved.append(list(msgs))

    async def query_memory(self, query):
        return ""


def _conn(memory, device_id="dev-a"):
    conn = ConnectionHandler(copy.deepcopy(CONFIG), None, None, None, memory, None)
    conn.device_id = device_id
    conn.session_id = "s"
    return conn


def test_each_connection_gets_its_own_memory_instance(tmp_path):
    shared = MemoryProvider({}, None)
    shared.memory_path = str(tmp_path / "m.yaml")
    a, b = _conn(shared, "dev-a"), _conn(shared, "dev-b")
    a._initialize_memory()
    b._initialize_memory()
    assert a.memory is not shared and b.memory is not shared and a.memory is not b.memory
    assert (a.memory.role_id, b.memory.role_id) == ("dev-a", "dev-b")
    assert shared.role_id is None          # 共用的那份没被改


def _run_save_and_close(conn, monkeypatch):
    real_thread = connection_mod.threading.Thread

    class SyncThread:                       # 照样开线程（它自己建事件循环），但等它跑完再往下，好断言
        def __init__(self, target, daemon=None):
            self.t = real_thread(target=target, daemon=daemon)

        def start(self):
            self.t.start()
            self.t.join(5)

    async def no_close(ws=None):
        pass

    monkeypatch.setattr(connection_mod.threading, "Thread", SyncThread)
    conn.close = no_close
    asyncio.run(conn._save_and_close(None))


def test_save_and_close_skips_wake_only_session(monkeypatch):
    mem = FakeMemory()
    conn = _conn(mem)
    conn.dialogue.put(Message(role="user", content="拜拜", is_temporary=True))
    conn.dialogue.put(Message(role="assistant", content="再见，下次再聊~", is_temporary=True))
    conn.dialogue.put(Message(role="user", content=WAKE_GREETING_QUERY))
    conn.dialogue.put(Message(role="assistant", content="你好呀"))
    _run_save_and_close(conn, monkeypatch)
    assert mem.saved == []


def test_save_and_close_before_components_does_nothing(monkeypatch):
    """鉴权后立刻断开：组件没初始化、对话为空，不调保存（这时 memory 还是共用的那份）。"""
    mem = FakeMemory()
    conn = _conn(mem)
    _run_save_and_close(conn, monkeypatch)
    assert mem.saved == []


def test_save_and_close_passes_only_real_turns(monkeypatch):
    mem = FakeMemory()
    conn = _conn(mem)
    conn.dialogue.put(Message(role="system", content="你是小智"))
    conn.dialogue.put(Message(role="user", content="给我讲个故事吧", is_temporary=True))
    conn.dialogue.put(Message(role="user", content="记住，我最喜欢的水果是芒果"))
    conn.dialogue.put(Message(role="assistant", content="记住啦"))
    _run_save_and_close(conn, monkeypatch)
    assert len(mem.saved) == 1
    assert [(m.role, m.content) for m in mem.saved[0]] == [("user", "记住，我最喜欢的水果是芒果"), ("assistant", "记住啦")]
