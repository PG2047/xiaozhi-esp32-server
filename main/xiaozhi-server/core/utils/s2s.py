"""S2S provider 的创建与挂接（与 core/utils/llm.py 的 create_instance 同一套路）。"""
import asyncio
import importlib
import os
import sys

from config.logger import setup_logging
from core.utils.status_ledger import append_session, is_server_process

TAG = __name__
logger = setup_logging()


def create_instance(class_name, *args, **kwargs):
    """按 type 名加载 core/providers/s2s/<type>/<type>.py 里的 S2SProvider。"""
    if os.path.exists(os.path.join("core", "providers", "s2s", class_name, f"{class_name}.py")):
        lib_name = f"core.providers.s2s.{class_name}.{class_name}"
        if lib_name not in sys.modules:
            sys.modules[lib_name] = importlib.import_module(f"{lib_name}")
        return sys.modules[lib_name].S2SProvider(*args, **kwargs)

    raise ValueError(f"不支持的S2S类型: {class_name}，请检查该配置的type是否设置正确")


def s2s_selected(conn) -> bool:
    """这条连接是否走 S2S（provider 可能还没建成）。打开失败就不算了：之后走级联，转写的文字要交给大模型。"""
    if getattr(conn, "_s2s_failed", False):
        return False
    return bool((conn.config.get("selected_module") or {}).get("S2S"))


async def s2s_on_text(conn, text: str) -> None:
    """文字入口：provider 建成了就直接给它；还没建成（唤醒词 detect 常比它早到一秒）就先记着，建成后回放。"""
    if conn.s2s is not None:
        await conn.s2s.on_text(text)
        return
    pending = getattr(conn, "_s2s_pending_texts", None)
    if pending is None:
        pending = conn._s2s_pending_texts = []
    pending.append(text)


async def drain_pending_texts(conn) -> None:
    pending = getattr(conn, "_s2s_pending_texts", None)
    if not pending or conn.s2s is None:
        return
    texts, pending[:] = list(pending), []
    for t in texts:
        await conn.s2s.on_text(t)


async def drain_pending_to_cascade(conn) -> None:
    """S2S 打开失败：之前记下的文字（唤醒词那句常比失败早到）交给级联，不然唤醒后没有回应。"""
    from core.handle.receiveAudioHandle import startToChat   # 那个模块 import 了本文件
    pending = getattr(conn, "_s2s_pending_texts", None)
    if not pending:
        return
    texts, pending[:] = list(pending), []
    for t in texts:
        await startToChat(conn, t)


EARLY_AUDIO_MAX = 60   # 约 3.6 秒（60 ms 一包）


def s2s_buffer_early_audio(conn, packet: bytes) -> bool:
    """选了 S2S、provider 还没建成（唤醒后约 1 秒）时先攒设备的 Opus 包；返回 True = 已攒，调用方别再往下走。
    建成之后由 drain_early_audio 按原顺序补送给愿意要的 provider（Live：wants_early_audio），
    这样"你好小智，一年有几个月"一口气说完时问题不丢（11-03 R22）；不要的（Realtime）照旧丢掉。attach 结束后不再攒。"""
    if getattr(conn, "_s2s_attach_done", False) or getattr(conn, "conn_from_mqtt_gateway", False) or not s2s_selected(conn):
        return False
    if not getattr(conn, "client_is_listening", False):
        return True   # listen start 之前是设备补发的唤醒词音频：照旧丢掉，不然模型听到"你好小智"会自己再打一遍招呼（11-03 R23）
    buf = getattr(conn, "_s2s_early_audio", None)
    if buf is None:
        buf = conn._s2s_early_audio = []
    if len(buf) < EARLY_AUDIO_MAX:
        buf.append(packet)
    return True


async def drain_early_audio(conn) -> None:
    """在连接的事件循环上、处理建成后第一个音频包之前调用：先把攒下的包解码交给 provider，保证顺序。"""
    buf = getattr(conn, "_s2s_early_audio", None)
    if not buf:
        return
    conn._s2s_early_audio = []
    if conn.s2s is None or not getattr(conn.s2s, "wants_early_audio", False):
        return
    for packet in buf:
        pcm = conn._decode_opus_packet(packet)
        if pcm:
            await conn.s2s.on_pcm(pcm)


def attach_s2s(conn):
    """在 ConnectionHandler._initialize_components（执行器线程）里调用。

    selected_module.S2S 没填 → None（原版级联）。填了就创建 provider 并在连接的事件循环上打开上游会话，
    整段限时 connect_timeout_s + 1 秒；任何失败都返回 None 并记一条警告，连接照常走级联，设备无感。
    """
    try:
        provider = _attach_s2s(conn)
        if provider is None and s2s_selected(conn):
            conn._s2s_failed = True    # 选了但没建成：之后的文字走级联（见 s2s_selected）
        return provider
    finally:
        conn._s2s_attach_done = True   # 之后的音频直达 provider（或走级联），不再攒


def _attach_s2s(conn):
    selected = (conn.config.get("selected_module") or {}).get("S2S")
    if not selected:
        return None
    cfg = (conn.config.get("S2S") or {}).get(selected)
    if not isinstance(cfg, dict):
        conn.logger.bind(tag=TAG).warning(f"selected_module.S2S={selected} 但没有对应的 S2S 配置块，回退级联")
        return None
    s2s_type = cfg.get("type", selected)
    timeout = float(cfg.get("connect_timeout_s", 4)) + 1.0
    provider = None
    fut = None
    try:
        provider = create_instance(s2s_type, cfg, conn)
        fut = asyncio.run_coroutine_threadsafe(provider.open(), conn.loop)
        fut.result(timeout=timeout)
    except Exception as e:  # noqa: BLE001
        conn.logger.bind(tag=TAG).warning(f"S2S {selected} 打开失败，回退原版级联: {type(e).__name__}: {e}")
        conn._s2s_failed = True    # 先于 provider.close()：它据此不写台账，这条连接由 record_cascade_session 记一行
        if fut is not None:
            fut.cancel()  # 超时的那次 open() 可能还在跑：取消掉，别让它事后建成一个没人管的会话
        if provider is not None:
            try:
                asyncio.run_coroutine_threadsafe(provider.close(), conn.loop)
            except Exception:  # noqa: BLE001
                pass
        return None
    conn.logger.bind(tag=TAG).info(f"S2S {selected} 已接管本连接（不建 TTS / ASR）")
    # 兜底：组件初始化后面某步失败、after_components_ready 没被调到时，建成前排队的文字也要送出去
    # 这里在执行器线程：loop.call_later 不是线程安全的，要经 call_soon_threadsafe 排进事件循环再定时
    conn.loop.call_soon_threadsafe(conn.loop.call_later, 3.0, lambda: asyncio.ensure_future(drain_pending_texts(conn)))
    return provider


def record_cascade_session(conn) -> None:
    """走级联的连接在关闭时记一行状态页台账（S2S 的由 provider 自己记）。

    只在关闭时数一次对话历史，不碰语音路径；close() 可能走多次，只记一次。
    stats 不用 segments / turns：monitoring-stack 的余额探针靠这两个键证明 OpenAI key 还能用，级联不用那把 key。
    """
    if getattr(conn, "_cascade_recorded", False) or conn.s2s is not None or s2s_selected(conn):
        return                 # s2s_selected 为真 = 选了 S2S 但还没建成就断开了，这条连接没有对话
    conn._cascade_recorded = True
    try:
        started_ms = getattr(conn, "first_activity_time", 0)
        status_dir = (conn.config.get("log") or {}).get("status_dir")
        if not status_dir or not started_ms or not is_server_process():
            return
        selected = conn.config.get("selected_module") or {}
        messages = getattr(conn.dialogue, "dialogue", [])
        fallback_from = None
        if getattr(conn, "_s2s_failed", False):
            block = (conn.config.get("S2S") or {}).get(selected.get("S2S")) or {}
            fallback_from = block.get("type") or selected.get("S2S")
        llm = (conn.config.get("LLM") or {}).get(selected.get("LLM")) or {}
        append_session(
            status_dir,
            device=getattr(conn, "device_name", None),
            brain="cascade",
            model=llm.get("model_name") or selected.get("LLM"),
            started_at=started_ms / 1000,
            stats={
                "replies": sum(1 for m in messages if m.role == "assistant" and m.content),
                "input_utts": sum(1 for m in messages if m.role == "user"),
            },
            fallback_from=fallback_from,
        )
    except Exception:  # noqa: BLE001  写台账失败不能影响连接关闭
        pass
