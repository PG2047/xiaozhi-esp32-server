"""状态页台账：写出来的内容都可以上页面（没有设备标识、没有用户原话），没配目录就不写，出错不影响会话。"""
from __future__ import annotations

import json
import stat

from loguru import logger

from core.utils.status_ledger import USER_TEXT_PREFIXES, add_error_sink, append_session, is_server_process, scrub_message

MAC = "aa:bb:cc:dd:ee:ff"
UUID = "11111111-2222-4333-8444-555555555555"


def test_scrub_masks_identifiers_and_flattens_lines():
    out = scrub_message(f"连接 {MAC} {UUID} 来自 10.1.2.3\n第二行 Bearer abc.def sk-abcdefgh12345")
    assert MAC not in out and UUID not in out and "10.1.2.3" not in out
    assert "abc.def" not in out and "sk-abcdefgh12345" not in out
    assert "\n" not in out
    assert "<mac>" in out and "<uuid>" in out and "<ip>" in out


def test_scrub_drops_noise_and_cuts_long_lines():
    assert scrub_message("声纹识别URL未配置，声纹识别将被禁用") is None
    assert len(scrub_message("x" * 1000)) == 200


def test_scrub_removes_user_text_from_known_templates():
    for prefix in USER_TEXT_PREFIXES:
        out = scrub_message(f"{prefix} 帮我查一下明天的日程: boom")
        assert "日程" not in out and "boom" not in out
        assert out.startswith(prefix)


def test_append_session_writes_one_json_line(tmp_path):
    stats = {"segments": 3, "barge_device": 1, "barge_transcript": 0, "seg_after_input_ms": [100, 200]}
    for _ in range(2):
        append_session(str(tmp_path), device="书房", brain="openai_live", model="gpt-live-1",
                       started_at=100.0, ended_at=161.26, stats=stats)
    lines = (tmp_path / "sessions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    row = json.loads(lines[0])
    assert row == {"device": "书房", "brain": "openai_live", "model": "gpt-live-1", "started_at": 100.0,
                   "ended_at": 161.26, "duration_s": 61.3, "stats": stats}


def test_append_session_is_a_noop_without_dir_and_never_raises(tmp_path):
    append_session(None, device="d", brain="b", model="m", started_at=1.0, stats={})
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    append_session(str(blocked), device="d", brain="b", model="m", started_at=1.0, stats={})


def test_error_sink_writes_scrubbed_warnings_only(tmp_path):
    sink = add_error_sink(logger, str(tmp_path))
    try:
        log = logger.bind(tag="core.demo")
        log.info(f"用户说：不该出现 {MAC}")
        log.warning(f"设备 {MAC} 掉线")
        log.warning("声纹识别URL未配置，声纹识别将被禁用")
        log.error("LLM 处理出错 帮我查日程: boom")
        try:
            raise ValueError("堆栈里的内容")
        except ValueError:
            log.exception("工具失败")
        logger.complete()
    finally:
        logger.remove(sink)
    raw = (tmp_path / "errors.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in raw.splitlines()]
    assert [r["message"] for r in rows] == ["设备 <mac> 掉线", "LLM 处理出错（内容已省略）", "工具失败"]
    assert rows[0]["level"] == "WARNING" and rows[0]["module"] == "core.demo" and rows[0]["at"] > 0
    assert "用户说" not in raw and MAC not in raw and "日程" not in raw and "堆栈" not in raw
    assert json.loads((tmp_path / "server.json").read_text())["started_at"] > 0
    assert stat.S_IMODE((tmp_path / "errors.jsonl").stat().st_mode) & 0o044 == 0o044  # 非 root 的状态页进程要读得到


def test_only_the_server_process_counts(monkeypatch):
    for argv0, expected in (("app.py", True), ("/opt/xiaozhi-esp32-server/app.py", True), ("-", False), ("-c", False),
                            ("probe.py", False), ("", False)):
        monkeypatch.setattr("sys.argv", [argv0])
        assert is_server_process() is expected
