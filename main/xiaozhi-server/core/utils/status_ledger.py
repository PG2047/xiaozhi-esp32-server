"""状态页的台账：status_dir 目录下只放可以上页面的内容。

状态页的网页进程只读这个目录，碰不到 docker、登记表和原始日志，所以这里写什么，页面最多就能漏什么：

    sessions.jsonl   每段 S2S 会话结束写一行：设备名、大脑、模型、开始 / 结束时间、stats（只有计数，没有转写）
    errors.jsonl     WARNING 以上的日志，写之前遮掉设备标识、删掉带用户原话的尾巴
    server.json      server 进程的启动时间

目录由 config["log"]["status_dir"] 决定，没配就什么都不写（上游默认行为不变）。
任何异常都只吞掉：写台账失败不能影响会话。
"""
from __future__ import annotations

import json
import os
import re
import time

# 这些格式串把用户原话、工具参数或设备消息拼进了日志，只留前面的固定文字。加新的 error / warning 时带上用户文本的话，这里也要加一条。
USER_TEXT_PREFIXES = (
    "LLM 处理出错",                 # core/connection.py：{query}
    "收到未知类型消息",              # core/handle/textMessageProcessor.py：{message}
    "解析到错误的消息",              # 同上
    "加入TTS上报队列失败",           # core/handle/reportHandle.py：{text}
    "无法解析函数参数",              # core/providers/tools/unified_tool_handler.py：{arguments}
    "audio_play_priority_thread",  # core/providers/tts/base.py：{text}
    "Live 后台",                    # core/providers/s2s/openai_live：后台返回的 JSON 可能带工具参数
)
# 每段会话都会打一条、没有信息量的告警，不进错误列表
NOISE = ("声纹识别URL未配置",)
MAX_MESSAGE_CHARS = 200

_MAC = re.compile(r"\b(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}\b", re.I)
_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
_IPV4 = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_BEARER = re.compile(r"\bbearer\s+\S+", re.I)
_API_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}")


def scrub_message(message: str) -> str | None:
    """日志正文 → 可以上页面的一行。返回 None = 不进错误列表。"""
    text = str(message)
    if any(n in text for n in NOISE):
        return None
    for prefix in USER_TEXT_PREFIXES:
        if text.startswith(prefix):
            return f"{prefix}（内容已省略）"
    text = _MAC.sub("<mac>", text)
    text = _UUID.sub("<uuid>", text)
    text = _IPV4.sub("<ip>", text)
    text = _BEARER.sub("<token>", text)
    text = _API_KEY.sub("<key>", text)
    text = " ".join(text.split())
    return text[:MAX_MESSAGE_CHARS]


def append_session(status_dir, *, device, brain, model, started_at, stats, ended_at=None) -> None:
    """写一行会话记录。stats 是 provider 的计数器，只放数字和数字列表，不放转写。"""
    if not status_dir:
        return
    try:
        ended_at = ended_at if ended_at is not None else time.time()
        line = json.dumps(
            {
                "device": device,
                "brain": brain,
                "model": model,
                "started_at": round(started_at, 3),
                "ended_at": round(ended_at, 3),
                "duration_s": round(ended_at - started_at, 1),
                "stats": stats,
            },
            ensure_ascii=False,
            default=str,
        )
        os.makedirs(status_dir, exist_ok=True)
        with open(os.path.join(status_dir, "sessions.jsonl"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:  # noqa: BLE001  写台账失败不能影响会话
        pass


def _error_filter(record) -> bool:
    message = scrub_message(record["message"])
    if message is None:
        return False
    record["extra"]["status_line"] = json.dumps(
        {
            "at": round(record["time"].timestamp(), 3),
            "level": record["level"].name,
            "module": record["extra"].get("tag") or record["name"],
            "message": message,
        },
        ensure_ascii=False,
    )
    return True


def _error_format(_record) -> str:
    return "{extra[status_line]}\n"  # 用函数形式：loguru 不会再追加 {exception}，异常堆栈不进这个文件


def add_error_sink(logger, status_dir: str):
    """给 loguru 加一路 WARNING 以上的输出（后台线程写，不阻塞事件循环），并记下启动时间。返回 handler id。"""
    os.makedirs(status_dir, exist_ok=True)
    _write_server_start(status_dir)
    return logger.add(
        os.path.join(status_dir, "errors.jsonl"),
        level="WARNING",
        format=_error_format,
        filter=_error_filter,
        rotation="1 MB",
        retention=1,
        encoding="utf-8",
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )


def _write_server_start(status_dir: str) -> None:
    try:
        path = os.path.join(status_dir, "server.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"started_at": round(time.time(), 3)}, f)
        os.replace(tmp, path)
    except OSError:
        pass
