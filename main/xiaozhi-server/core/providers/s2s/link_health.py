"""到 S2S 上游这条连接的健康度：只记录，不改任何行为。

为什么要记：11-04 19:40 真机上三段响亮的话，转写依次晚了 19、14、11 s，约 1.9 倍速追上，现有日志分不清卡在路上还是 Live 内部。
Tokyo 正常时的基线（11-06 直连 15 s）：下行约 100 ms 一条、最大 142 ms，ping 约 100 ms，上行写缓冲 0.2 ms。

  rx_gap_ms     两条下行消息之间隔多久，不含处理时间。Live 下行是连续流（静音也发），隔久了就是 Live 或下行路上停了
  rx_handle_ms  这边处理一条下行消息花多久。它长说明卡在服务端自己
  send_ms       上行一条消息写进发送缓冲花多久。只有缓冲满了才变长，所以它短不能说明路上没卡
  ping_ms       WebSocket ping 往返，和音频走同一条 TCP，但不经过模型

怎么读：ping 也晚，是 Tokyo 到 OpenAI 的路上（上行、下行都算）；ping 准时而下行断了，是 Live 内部；
都正常还是没转写，是 Live 那边没把它当成话，或者本来就是噪声（没存音频，分不清）。
"""
from __future__ import annotations

KEYS = ("rx_gap_ms", "rx_handle_ms", "send_ms", "ping_ms")


class LinkHealth:
    """时刻都由调用方传入（monotonic 秒），方便测试。session 是整个会话的最大值，window 是 reset_window 之后的最大值。"""

    def __init__(self):
        self.session = dict.fromkeys(KEYS, 0)
        self.window = dict.fromkeys(KEYS, 0)
        self.t_idle = None     # 上一条下行处理完的时刻；还没收到过或正在处理时是 None
        self.t_rx = None       # 正在处理的这条下行是什么时候到的
        self.t_ping = None     # 还没回 pong 的那个 ping 是什么时候发的

    def _note(self, key: str, seconds: float) -> None:
        ms = round(seconds * 1000)
        self.session[key] = max(self.session[key], ms)
        self.window[key] = max(self.window[key], ms)

    def rx(self, now: float) -> None:
        if self.t_idle is not None:
            self._note("rx_gap_ms", now - self.t_idle)
        self.t_idle, self.t_rx = None, now

    def rx_done(self, now: float) -> None:
        if self.t_rx is not None:
            self._note("rx_handle_ms", now - self.t_rx)
        self.t_idle, self.t_rx = now, None

    def sent(self, seconds: float) -> None:
        self._note("send_ms", seconds)

    def ping_sent(self, now: float) -> None:
        self.t_ping = now

    def pong(self, now: float) -> None:
        if self.t_ping is not None:
            self._note("ping_ms", now - self.t_ping)
        self.t_ping = None

    def reset_window(self) -> None:
        self.window = dict.fromkeys(KEYS, 0)

    def _with_open(self, values: dict, now: float) -> dict:
        """把还没结束的那次也算上：正卡着的时候，下行还没来、pong 还没回，只看已结束的会以为一切正常。"""
        out = dict(values)
        if self.t_idle is not None:
            out["rx_gap_ms"] = max(out["rx_gap_ms"], round((now - self.t_idle) * 1000))
        if self.t_ping is not None:
            out["ping_ms"] = max(out["ping_ms"], round((now - self.t_ping) * 1000))
        return out

    def window_view(self, now: float) -> dict:
        return self._with_open(self.window, now)

    def session_view(self, now: float) -> dict:
        return self._with_open(self.session, now)
