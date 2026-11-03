"""上行人声门：送给模型之前，把 silero 判为"不是人声"的帧换成静音（键盘声、碰撞声、杂音）。

为什么要它（11-03 v9 真人复测）：GPT-Live 自己判轮次。用户在设备旁敲键盘时，键盘声一直送进模型，模型就一直以为"用户还在出声"，
迟迟不接话（"不对，我想听个笑话"8 s 后才开讲），还会把前后两句合成一句（"小智停一下接着讲吧"）。
人声照常送；不是人声的帧送全零，保持音频时间线连续。

做法：固定的前瞻延迟（lookahead_ms，默认 120 ms）。每帧晚 lookahead 才送出，送不送看它前后有没有人声：
  - 前瞻：后面 lookahead 内出现人声，这一帧照送，人声开头（silero 起判要一两个窗口）不会被剪掉；
  - 保持：最后一次人声之后 hold_ms（默认 600 ms）内照送，词和词之间的停顿不会被挖空。
不用"开门时补发预录"：那样每开一次门，模型收到的音频就比实际时间多一截，越积越多就成了延迟。
"""
from __future__ import annotations

from collections import deque

import numpy as np

WIN = 512          # silero 每次 512 样本（16 kHz，32 ms）


class SpeechGate:
    def __init__(self, vad, threshold: float = 0.5, hold_ms: int = 600, lookahead_ms: int = 120, frame_ms: int = 60):
        self.vad = vad
        self.threshold = float(threshold)
        self.frame_ms = int(frame_ms)
        self.lookahead = max(0, round(int(lookahead_ms) / self.frame_ms))
        self.hold = max(0, round(int(hold_ms) / self.frame_ms))
        self._carry = np.zeros(0, dtype=np.float32)
        self._last_prob = 0.0
        self._queue: deque[tuple[bytes, bool]] = deque()   # (帧, 这帧是否人声)，长度 = lookahead + 1
        self._since_speech = 10 ** 6                        # 已送出的帧里，距离最近一次人声隔了几帧
        self.speech_frames = 0
        self.silenced_frames = 0
        self.t_last_speech = None                           # 最近一次判出人声的时刻（time.monotonic，调用方填）

    def _is_speech(self, pcm: bytes) -> bool:
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        buf = np.concatenate([self._carry, x])
        probs = []
        while len(buf) >= WIN:
            probs.append(self.vad(buf[:WIN]))
            buf = buf[WIN:]
        self._carry = buf
        if probs:
            self._last_prob = max(probs)
        return self._last_prob >= self.threshold

    def process(self, pcm: bytes, now: float | None = None) -> bytes:
        """送进一帧（60 ms），返回 lookahead 之前那一帧该送的内容（原样或全零）；队列没填满时返回 b""。"""
        speech = self._is_speech(pcm)
        if speech:
            self.t_last_speech = now
        self._queue.append((pcm, speech))
        if len(self._queue) <= self.lookahead:
            return b""
        frame, frame_speech = self._queue.popleft()
        ahead = any(s for _, s in self._queue)
        if frame_speech:
            self._since_speech = 0
        else:
            self._since_speech += 1
        if frame_speech or ahead or self._since_speech <= self.hold:
            self.speech_frames += 1
            return frame
        self.silenced_frames += 1
        return b"\x00" * len(frame)

    @property
    def stats(self) -> dict:
        return {"pass": self.speech_frames, "silenced": self.silenced_frames}
