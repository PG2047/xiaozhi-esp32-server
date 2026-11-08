"""Tests for the Message/Dialogue classes used to build LLM context."""
import pytest

from core.utils.dialogue import Dialogue, Message


def test_message_assigns_default_uniq_id():
    m = Message(role="user", content="hi")
    assert m.uniq_id is not None
    assert len(m.uniq_id) > 0


def test_message_preserves_explicit_uniq_id():
    m = Message(role="user", content="hi", uniq_id="abc")
    assert m.uniq_id == "abc"


def test_message_default_is_not_temporary():
    m = Message(role="user", content="hi")
    assert m.is_temporary is False


def test_message_tool_call_id_default_is_none():
    m = Message(role="user", content="hi")
    assert m.tool_call_id is None


def test_dialogue_starts_empty():
    d = Dialogue()
    assert d.dialogue == []


def test_dialogue_put_appends_message():
    d = Dialogue()
    d.put(Message(role="user", content="hi"))
    d.put(Message(role="assistant", content="hello"))
    assert len(d.dialogue) == 2


def test_dialogue_get_llm_dialogue_includes_user_and_assistant():
    d = Dialogue()
    d.put(Message(role="user", content="hi"))
    d.put(Message(role="assistant", content="hello"))
    msgs = d.get_llm_dialogue()
    assert {"role": "user", "content": "hi"} in msgs
    assert {"role": "assistant", "content": "hello"} in msgs


def test_dialogue_update_system_message_replaces_existing():
    d = Dialogue()
    d.update_system_message("you are a helper")
    d.update_system_message("you are a coder")
    system_msgs = [m for m in d.dialogue if m.role == "system"]
    assert len(system_msgs) == 1
    assert system_msgs[0].content == "you are a coder"


def test_dialogue_update_system_message_creates_when_missing():
    d = Dialogue()
    d.update_system_message("you are a coder")
    assert len(d.dialogue) == 1
    assert d.dialogue[0].role == "system"


@pytest.mark.asyncio
async def test_get_llm_dialogue_with_memory_substitutes_into_system_message():
    d = Dialogue()
    d.update_system_message("Memory so far: <memory></memory>")
    msgs = d.get_llm_dialogue_with_memory(memory_str="user likes cats")
    system = next(m for m in msgs if m["role"] == "system")
    assert "user likes cats" in system["content"]
    assert "<memory></memory>" not in system["content"]


@pytest.mark.asyncio
async def test_get_llm_dialogue_separates_temporary_few_shot_from_actual_history():
    d = Dialogue()
    d.put(Message(role="user", content="real user msg", is_temporary=False))
    d.put(Message(role="user", content="fewshot example", is_temporary=True))
    msgs = d.get_llm_dialogue()
    contents = [m["content"] for m in msgs if m["role"] == "user"]
    assert "real user msg" in contents
    assert "fewshot example" in contents

# ---------- dialogue_for_memory：交给记忆总结的消息 ----------
from core.utils.dialogue import WAKE_GREETING_QUERY, dialogue_for_memory  # noqa: E402


def _fewshot():
    return [
        Message(role="user", content="给我讲个故事吧", is_temporary=True),
        Message(role="tool", content="已直接回复", is_temporary=True, tool_call_id="x"),
        Message(role="user", content="拜拜", is_temporary=True),
        Message(role="assistant", content="再见，下次再聊~", is_temporary=True),
    ]


def test_memory_skips_session_with_only_fewshot_and_wake_reply():
    msgs = [Message(role="system", content="你是小智")] + _fewshot()
    msgs.append(Message(role="assistant", content="我在这里哦！"))   # 唤醒应答缓存命中时只有这一条
    assert dialogue_for_memory(msgs) == []


def test_memory_skips_session_with_only_the_wake_greeting():
    msgs = _fewshot() + [Message(role="user", content=WAKE_GREETING_QUERY),
                         Message(role="assistant", content="你好呀，有什么可以帮你？")]
    assert dialogue_for_memory(msgs) == []


def test_memory_ignores_configured_wake_words():
    msgs = [Message(role="user", content="你好小智"), Message(role="assistant", content="我在")]
    assert dialogue_for_memory(msgs, ["你好小智"]) == []


def test_memory_keeps_only_real_turns():
    real_u = Message(role="user", content="记住，我最喜欢的水果是芒果")
    real_a = Message(role="assistant", content="记住啦")
    msgs = [Message(role="system", content="你是小智")] + _fewshot() + [
        Message(role="user", content=WAKE_GREETING_QUERY), Message(role="assistant", content="你好呀"), real_u, real_a]
    out = dialogue_for_memory(msgs)
    assert real_u in out and real_a in out
    assert all(not m.is_temporary and m.role != "system" for m in out)
    assert all(m.content != WAKE_GREETING_QUERY for m in out)


def test_memory_needs_a_reply():
    assert dialogue_for_memory([Message(role="user", content="记住我喜欢芒果")]) == []
