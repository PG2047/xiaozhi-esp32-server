"""mem_local_short 的文件读写：两台设备的记忆存在同一个 YAML 里，并发保存不能互相冲掉；
总结以文件里最新的记忆为"历史"；写文件是原子的；日志不出现设备身份。LLM 用假的，不出网。"""
from __future__ import annotations

import asyncio
import threading
import time

import yaml

from tests.conftest import CONFIG
from core.utils.cache.manager import CacheType, cache_manager

cache_manager.set(CacheType.CONFIG, "main_config", CONFIG)

from core.providers.memory.mem_local_short.mem_local_short import MemoryProvider  # noqa: E402
from core.utils.dialogue import Message  # noqa: E402


class FakeLLM:
    model_name = "fake"
    api_key = "test-key"

    def __init__(self, reply=lambda user: '{"k": "v"}', delay=0.0):
        self.reply, self.delay, self.inputs = reply, delay, []

    def response_no_stream(self, system_prompt, user_prompt, **kwargs):
        self.inputs.append(user_prompt)
        time.sleep(self.delay)
        return self.reply(user_prompt)


def _provider(path, role, llm=None):
    p = MemoryProvider({}, None)
    p.memory_path = str(path)
    p.init_memory(role, llm or FakeLLM())
    return p


def _talk():
    return [Message(role="user", content="我最喜欢的水果是芒果"), Message(role="assistant", content="记住啦")]


def _read(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def test_save_keeps_other_devices(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text(yaml.dump({"dev-b": '{"b": 1}'}, allow_unicode=True), encoding="utf-8")
    p = _provider(path, "dev-a", FakeLLM(lambda u: '{"a": 1}'))
    asyncio.run(p.save_memory(_talk(), "s"))
    assert _read(path) == {"dev-a": '{"a": 1}', "dev-b": '{"b": 1}'}


def test_concurrent_saves_of_two_devices_both_survive(tmp_path):
    """没有锁时两边都读到空文件、各写各的，后写的冲掉先写的。"""
    path = tmp_path / "m.yaml"
    pa = _provider(path, "dev-a", FakeLLM(lambda u: '{"who": "a"}', delay=0.2))
    pb = _provider(path, "dev-b", FakeLLM(lambda u: '{"who": "b"}', delay=0.2))
    ts = [threading.Thread(target=lambda p=p: asyncio.run(p.save_memory(_talk(), "s"))) for p in (pa, pb)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(5)
    assert _read(path) == {"dev-a": '{"who": "a"}', "dev-b": '{"who": "b"}'}


def test_summary_uses_latest_memory_in_file_as_history(tmp_path):
    """同一台设备两段会话挨得很近：后一段连上时读到的是旧记忆，保存时要以文件里最新的为历史。"""
    path = tmp_path / "m.yaml"
    path.write_text(yaml.dump({"dev-a": '{"v": "old"}'}), encoding="utf-8")
    llm = FakeLLM()
    p = _provider(path, "dev-a", llm)
    assert p.short_memory == '{"v": "old"}'
    path.write_text(yaml.dump({"dev-a": '{"v": "newer"}'}), encoding="utf-8")
    asyncio.run(p.save_memory(_talk(), "s"))
    assert '{"v": "newer"}' in llm.inputs[0] and "old" not in llm.inputs[0]


def test_write_is_atomic_and_leaves_no_temp_file(tmp_path):
    path = tmp_path / "m.yaml"
    p = _provider(path, "dev-a")
    asyncio.run(p.save_memory(_talk(), "s"))
    assert _read(path) == {"dev-a": '{"k": "v"}'}
    assert not (tmp_path / "m.yaml.tmp").exists()


def test_bad_summary_keeps_the_file_untouched(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text(yaml.dump({"dev-a": '{"v": "old"}'}), encoding="utf-8")
    p = _provider(path, "dev-a", FakeLLM(lambda u: "不是 JSON"))
    asyncio.run(p.save_memory(_talk(), "s"))
    assert _read(path) == {"dev-a": '{"v": "old"}'}


def test_device_without_memory_starts_empty(tmp_path):
    """同一个实例换了设备：文件里没有这台的记忆就从空开始，不沿用上一台的。"""
    path = tmp_path / "m.yaml"
    path.write_text(yaml.dump({"dev-a": '{"v": "a"}'}), encoding="utf-8")
    p = _provider(path, "dev-a")
    assert p.short_memory == '{"v": "a"}'
    p.init_memory("dev-c", FakeLLM())
    assert p.short_memory == ""


def test_success_log_has_no_device_id(tmp_path):
    from loguru import logger

    seen = []
    hid = logger.add(lambda m: seen.append(str(m)), level="INFO", format="{message}")
    try:
        p = _provider(tmp_path / "m.yaml", "aa:bb:cc:dd:ee:ff")
        asyncio.run(p.save_memory(_talk(), "s"))
    finally:
        logger.remove(hid)
    assert any("Save memory successful" in s for s in seen)
    assert not any("aa:bb:cc:dd:ee:ff" in s for s in seen)


def test_summary_prompt_from_config_is_used(tmp_path):
    seen = []

    class RecordingLLM(FakeLLM):
        def response_no_stream(self, system_prompt, user_prompt, **kwargs):
            seen.append(system_prompt)
            return '{"k": "v"}'

    p = MemoryProvider({"summary_prompt": "  只记用户本人的稳定信息，只输出 JSON。  "}, None)
    p.memory_path = str(tmp_path / "m.yaml")
    p.init_memory("dev-a", RecordingLLM())
    asyncio.run(p.save_memory(_talk(), "s"))
    assert seen == ["只记用户本人的稳定信息，只输出 JSON。"]


def test_default_summary_prompt_when_not_configured():
    from core.providers.memory.mem_local_short.mem_local_short import short_term_memory_prompt

    assert MemoryProvider({}, None).summary_prompt == short_term_memory_prompt
    assert MemoryProvider({"summary_prompt": ""}, None).summary_prompt == short_term_memory_prompt


def test_loading_does_not_wait_for_a_running_summary(tmp_path):
    """新连接在建会话之前读记忆；另一条连接正在总结（持锁等几秒的请求）时，读不能跟着排队，不然唤醒应答要多等这几秒。"""
    path = tmp_path / "m.yaml"
    path.write_text(yaml.dump({"dev-b": '{"b": 1}'}), encoding="utf-8")
    slow = _provider(path, "dev-a", FakeLLM(delay=1.0))
    t = threading.Thread(target=lambda: asyncio.run(slow.save_memory(_talk(), "s")))
    t.start()
    time.sleep(0.1)                         # 让它先拿到锁、进入总结
    t0 = time.monotonic()
    other = _provider(path, "dev-b")
    took = time.monotonic() - t0
    t.join(5)
    assert other.short_memory == '{"b": 1}' and took < 0.5


def test_two_summaries_of_one_device_run_one_after_another(tmp_path):
    """同一台设备两段会话挨得很近、两次总结同时在跑：后一次要以前一次的结果为历史，
    不能两次都从旧记忆出发、后写的把先写的冲掉（总结全程持锁管的就是这个）。"""
    path = tmp_path / "m.yaml"
    replies, lock = iter(['{"第": 1}', '{"第": 2}']), threading.Lock()

    def reply(_user):
        with lock:
            return next(replies)

    llm = FakeLLM(reply, delay=0.2)
    p1, p2 = _provider(path, "dev-a", llm), _provider(path, "dev-a", llm)
    ts = [threading.Thread(target=lambda p=p: asyncio.run(p.save_memory(_talk(), "s"))) for p in (p1, p2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(5)
    assert len(llm.inputs) == 2 and '{"第": 1}' in llm.inputs[1]
    assert _read(path) == {"dev-a": '{"第": 2}'}
