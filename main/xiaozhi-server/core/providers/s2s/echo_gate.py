"""一级残余回声门控：S2S provider 在把设备上行送给模型之前，用它压掉设备端 AEC 剩下的回声。

为什么需要（2026-10-20 ESP32-S3 真机实测）：设备端 AEC 之后，播放期间上行仍有与播放内容相关的残余：安静时 RMS 中位 25，
播放时峰值可到 1200，包络与下行相关、滞后稳定；模型侧 VAD 会把它当成用户开口，模型播到 0.5 秒就被自己打断、
再接着回答自己的回声。门控打开后同样的场景 0 次误触发。

做法：provider 知道自己发了什么、什么时候发的：
  1. 参考 = 本段第一帧发送时刻起、按设备实际播放排列的下行 PCM（断流处补零）；
  2. 滞后先验 200 ms，0～1200 ms 内按 50 ms 对数包络互相关复估（插话多时跳过）；
  3. 回声回传比 g = 播放期间"上行功率 / 参考尾窗最大功率"的 90% 分位，但不超过中位数的 g_mult 倍：比值池里混着用户插话的
     子帧（插话时上行 / 参考比残余回声高得多），90% 分位会被它们顶到"人声电平"，之后门槛就按人声电平定、把人声也当回声
     （11-01 XVF3800：播放 1045 帧只放行 4 帧，音量 25 的插话过不去）；中位数要一半以上的池子是人声才会被顶起来。
     倍数默认 8（合成信号里 4 倍在锁定前的宽窗阶段会漏回声，8 倍不漏；人声子帧只能把 90% 分位顶高，顶不动中位数）；
  4. 每 20 ms 子帧判：上行功率 > α·g·P_ref + (2×底噪)²，且 silero 判为人声 → 整帧放行（判决 pass）并保持 300 ms
     （保持期内的帧判决 pass-hold，同样放行但不算新证据）；否则整帧换成底噪水平的舒适噪声。
     设备不在播放、或这一帧对应的时间窗里设备没放东西时直通。
滞后：锁定前参考窗覆盖 start_window 全程（provider 传先验 ±250 ms；11-01 实测窗开到 80 ms～1.5 s 时取的是 1.4 秒里的最大功率，
     门槛高到音量 25 的插话都过不去），先验只决定"候选离当前值近不近"；估计值离当前值 100 ms 内且相关 ≥ 0.5 直接采用，
     远的要相关 ≥ 0.7 或连续两次落在同一处才换（见 _accept_lag）。每段播放结束记一行诊断（g、各判决数、电平）。
silero 直接用上游自带的 onnx 模型（models/snakers4_silero-vad），不装 torch；模型文件不在时只用能量判据并记一条警告。
"""
from __future__ import annotations

import collections
import math
import os
from pathlib import Path

import numpy as np

from config.logger import setup_logging

TAG = __name__
logger = setup_logging()

RATE = 16000
SUB = RATE * 20 // 1000  # 20 ms 子帧 = 320 样本
ENV_S = 0.02
DEFAULT_VAD_MODEL = Path("models/snakers4_silero-vad/src/silero_vad/data/silero_vad.onnx")

_SESSIONS: dict[str, object] = {}  # 模型只加载一次，各会话各自保存状态


class SileroVAD:
    """silero VAD 的 onnx 推理（16 kHz、每次 512 样本 + 64 样本上下文），与上游 core/providers/vad/silero.py 同一模型。"""

    def __init__(self, path: Path):
        key = str(path)
        if key not in _SESSIONS:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            _SESSIONS[key] = ort.InferenceSession(key, sess_options=opts, providers=["CPUExecutionProvider"])
        self.sess = _SESSIONS[key]
        self.reset_states()

    def reset_states(self) -> None:
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, 64), dtype=np.float32)

    def __call__(self, chunk: np.ndarray) -> float:
        """chunk：512 个 float32 样本（−1～1）。返回人声概率。"""
        x = np.concatenate([self.context, chunk.reshape(1, -1).astype(np.float32)], axis=1)
        out, self.state = self.sess.run(None, {"input": x, "state": self.state, "sr": np.array(16000, dtype=np.int64)})
        self.context = x[:, -64:]
        return float(np.asarray(out).reshape(-1)[0])


def load_vad(path: str | os.PathLike | None = None):
    p = Path(path or DEFAULT_VAD_MODEL)
    if not p.exists():
        logger.bind(tag=TAG).warning(f"silero VAD 模型不存在（{p}），回声门控只用能量判据")
        return None
    return SileroVAD(p)


class EchoGate:
    def __init__(self, alpha=2.0, floor_rms=25.0, g_quantile=0.90, g_prior=1e-2, g_mult=8.0, lag_prior=0.20,
                 lag_min=0.0, lag_max=1.20, lag_step=0.02, lag_every=5.0, hold_s=0.30,
                 tail_back=0.12, tail_fwd=0.04, vad_threshold=0.5, vad=None, log_fn=None, seed=0,
                 start_window: tuple[float, float] | None = (0.08, 0.65)):
        """默认值来自真机：Seoul 公网入口下滞后 160～520 ms（相对本段第一帧发送时刻；Tokyo 入口 11-01 两台设备锁定在
        940～1180 ms，按部署传 lag_prior / lag_max），残余回声相对参考 −8～−50 dB 起伏，上行到达抖动中位 31 ms、最大 143 ms
        （所以音频时钟按帧序号重建）。vad=None 表示只用能量判据。
        start_window=(最短, 最长) 秒：还没估出滞后之前，参考窗放宽到覆盖这段回声延迟（会话开头先验常常不准，
        窄窗盖不住，唤醒应答的回声漏给模型，模型去回答自己的回声）；最长值应取到 lag_max。"""
        self.alpha = alpha
        self.floor_rms = floor_rms
        self.g_quantile = g_quantile
        self.g_prior = g_prior
        self.g_mult = g_mult     # g ≤ g_mult × 比值中位数；0 = 只用分位数
        self.lag = lag_prior
        self.lag_corr = None
        self._lag_cand = None  # 离当前值远、相关 0.5～0.7 的候选：下一次估计仍落在它附近才采用
        self.lag_min, self.lag_max, self.lag_step, self.lag_every = lag_min, lag_max, lag_step, lag_every
        self.hold_s = hold_s
        self.tail_back, self.tail_fwd = tail_back, tail_fwd
        self.start_window = start_window
        self.vad_threshold = vad_threshold
        self.vad = vad
        self.t_audio = None
        self.log = log_fn or (lambda m: logger.bind(tag=TAG).info(m))
        self.rng = np.random.default_rng(seed)
        self.seg_t0 = None
        self.seg_buf = bytearray()
        self.seg_end = None
        self.ratios = collections.deque(maxlen=250)
        self.up_env = collections.deque(maxlen=400)
        self.t_last_lag = -1e9
        self.g = None
        self.hold_until = -1.0
        self.recent = collections.deque(maxlen=100)
        self._seg = None   # 本段诊断：各判决计数与电平，on_tts_stop 时记一行
        self.stats = {"frames": 0, "bypass": 0, "pass": 0, "echo": 0, "lag_est": 0, "lag_reject": 0, "noref": 0,
                      "lag_skip_barge": 0}

    # ---------- 事件 ----------
    @property
    def seg(self) -> np.ndarray:
        return np.frombuffer(self.seg_buf, dtype=np.int16)  # 视图，不拷贝；别长期持有

    def on_tts_start(self, t: float) -> None:
        self.seg_t0 = None
        self.seg_buf = bytearray()
        self.seg_end = None
        self.up_env.clear()
        self._flush_seg(t)   # 上一段的尾窗还没过完就接上了新回复：先把上一段的诊断记掉
        self._seg = {"t0": t, "pass": 0, "hold": 0, "echo": 0, "noref": 0, "up_pass": [], "up_echo": [], "ref": []}
        if self.vad is not None:
            self.vad.reset_states()

    def on_downlink(self, pcm_bytes: bytes, t_send: float) -> None:
        """设备收到一帧就排进播放队列：本帧开播时刻 = max(发送时刻, 上一帧播完)。断流时中间补零。pcm 是 16 kHz。"""
        if self.seg_t0 is None:
            self.seg_t0 = t_send
        end_t = self.seg_t0 + len(self.seg_buf) / 2 / RATE
        gap = t_send - end_t
        if gap > 0.03:
            self.seg_buf.extend(b"\x00\x00" * int(gap * RATE))
        self.seg_buf.extend(pcm_bytes)

    def on_tts_stop(self, t: float) -> None:
        self.seg_end = t

    def _flush_seg(self, t: float) -> None:
        """一段播放（含播完后的尾窗）过完，记一行诊断：g、各判决数、放行 / 回声帧的电平。在尾窗过完后的第一帧直通时调用。"""
        s, self._seg = self._seg, None
        if s is None:
            return
        p50 = lambda xs: f"{math.sqrt(float(np.median(xs))):.0f}" if xs else "-"   # 功率中位数 → RMS
        ratios = np.array(self.ratios) if self.ratios else None
        self.log(f"echo gate seg {t - s['t0']:.1f}s lag={self.lag * 1000:.0f}ms corr={self.lag_corr if self.lag_corr is None else round(self.lag_corr, 2)} "
                 f"g={self.g if self.g is None else round(self.g, 4)}"
                 f"{'' if ratios is None else f' ratio_p50={np.quantile(ratios, 0.5):.4f}'} "
                 f"pass={s['pass']} hold={s['hold']} echo={s['echo']} noref={s['noref']} "
                 f"up_rms pass_max={math.sqrt(max(s['up_pass'])) if s['up_pass'] else 0:.0f} echo_p50={p50(s['up_echo'])} ref_p50={p50(s['ref'])}")

    # ---------- 内部 ----------
    def _window(self) -> tuple[float, float]:
        if self.lag_corr is None and self.start_window:
            lo, hi = self.start_window
            return max(self.tail_back, hi - self.lag), max(self.tail_fwd, self.lag - lo)
        return self.tail_back, self.tail_fwd

    def _active(self, t: float, speaking: bool) -> bool:
        if speaking and self.seg_t0 is not None:
            return True
        return self.seg_end is not None and self.seg_t0 is not None and t - self.seg_end < self.lag + self._window()[0] + 0.18

    def _ref_sub(self, t_sub: float):
        if self.seg_t0 is None:
            return None
        idx = int(round((t_sub - self.lag - self.seg_t0) * RATE))
        if idx < 0 or idx + SUB > len(self.seg_buf) // 2:
            return None
        return np.frombuffer(self.seg_buf, dtype=np.int16, count=SUB, offset=idx * 2)

    def _ref_power(self, t_sub: float):
        best = None
        tail_back, tail_fwd = self._window()
        off = -tail_back
        while off <= tail_fwd + 1e-9:
            r = self._ref_sub(t_sub + off)
            if r is not None:
                p = float(np.mean(r.astype(np.float64) ** 2))
                best = p if best is None else max(best, p)
            off += ENV_S
        return best

    def _estimate_lag(self, t: float) -> None:
        # 重试节奏：锁定前触发周期是 2.4 秒、锁定后 lag_every；"t - 周期 + 1" = 1 秒后再试（原来固定按 lag_every 算，
        # 锁定前 t - 5 + 1 立刻又满足 ≥ 2.4，变成每帧都估一次：11-01 日志一秒 13 行 rejected，白烧 CPU）
        period = 2.4 if self.lag_corr is None else self.lag_every
        if len(self.up_env) < 100 or len(self.seg_buf) < 2 * RATE:
            self.t_last_lag = t - period + 0.5
            return
        if self.recent and sum(self.recent) / len(self.recent) > 0.3:
            self.stats["lag_skip_barge"] += 1
            self.t_last_lag = t - period + 1.0
            return
        self.t_last_lag = t
        ups = np.array(self.up_env, dtype=np.float64)
        W = 0.05
        seg = self.seg.astype(np.float64)
        n_env = int(len(seg) / RATE / W)
        wlen = int(RATE * W)
        ref_env = np.sqrt(np.mean(seg[:n_env * wlen].reshape(n_env, wlen) ** 2, axis=1))
        del seg
        quiet = ups[:, 1] <= np.median(ups[:, 1])
        up_log = np.log10(ups[:, 1] + 1.0)
        ref_log = np.log10(ref_env + 1.0)
        best_c, best_lag = -1.0, None
        for lag in np.arange(self.lag_min, self.lag_max + 1e-9, self.lag_step):
            k = np.floor((ups[:, 0] - lag - self.seg_t0) / W + 1e-6).astype(int)
            m = (k >= 0) & (k < n_env) & quiet
            if m.sum() < 30:
                continue
            x, y = up_log[m], ref_log[k[m]]
            if x.std() < 1e-6 or y.std() < 1e-6:
                continue
            c = float(np.corrcoef(x, y)[0, 1])
            if c > best_c:
                best_c, best_lag = c, float(lag)
        self.stats["lag_est"] += 1
        if best_lag is not None and self._accept_lag(best_lag, best_c):
            old = self.lag
            self.lag, self.lag_corr = best_lag, best_c
            self.log(f"echo gate lag {old * 1000:.0f} -> {best_lag * 1000:.0f} ms (corr {best_c:.2f})")
        else:
            self.stats["lag_reject"] += 1
            if best_lag is not None and best_c >= 0.25:
                self.log(f"echo gate lag candidate {best_lag * 1000:.0f} ms (corr {best_c:.2f}) rejected, keep {self.lag * 1000:.0f} ms")
            self.t_last_lag = t - period + 1.0

    def _accept_lag(self, lag: float, corr: float) -> bool:
        """滞后估计的接受规则（near = 候选与当前值相差 ≤ 100 ms）：
        - corr ≥ 0.7，或 corr ≥ 0.5 且 near：直接采用；
        - corr ≥ 0.5 但远：先记为候选，下一次估计仍落在它 100 ms 内才采用。几十个候选里挑最大值，0.5 出头的相关在噪声里
          也凑得出来（11-01 真机：一次 0.51 把滞后从 220 换到 80 ms，之后整段会话门控对真正的回声视而不见）；
        - 0.25 ≤ corr < 0.5 且 near：只在锁定过之后当作微调接受。锁定前不接受，免得弱相关把先验钉死
          （11-01 真机：0.44 把 200 换成 140 ms，而真实约 1000 ms，之后再没估出来过）。"""
        near = abs(lag - self.lag) <= 0.10
        if corr >= 0.7 or (corr >= 0.5 and near):
            ok = True
        elif corr >= 0.5:
            ok = self._lag_cand is not None and abs(lag - self._lag_cand) <= 0.10
            if not ok:
                self._lag_cand = lag
        else:
            ok = corr >= 0.25 and near and self.lag_corr is not None
        if ok:
            self._lag_cand = None
        return ok

    def _vad_prob(self, a: np.ndarray) -> float:
        if self.vad is None:
            return 1.0
        x = a.astype(np.float32) / 32768.0
        probs = []
        for start in (0, max(0, len(x) - 512)):
            chunk = x[start:start + 512]
            if len(chunk) < 512:
                chunk = np.pad(chunk, (0, 512 - len(chunk)))
            probs.append(self.vad(chunk))
        return max(probs)

    # ---------- 主入口 ----------
    def process(self, pcm_bytes: bytes, t_recv: float, speaking: bool) -> tuple[bytes, str]:
        """返回 (送给模型的 PCM, 判决)。判决首词：bypass / noref / pass / echo。t_recv 是这 60 ms 帧到达的时刻。"""
        self.stats["frames"] += 1
        a = np.frombuffer(pcm_bytes, dtype=np.int16)
        if self.t_audio is None:
            self.t_audio = t_recv
        else:
            self.t_audio += 0.06
            if t_recv - self.t_audio > 0.2 or self.t_audio - t_recv > 0.1:
                self.t_audio = t_recv
        t_frame = self.t_audio
        if not self._active(t_frame, speaking):
            self.stats["bypass"] += 1
            self._flush_seg(t_frame)
            return pcm_bytes, "bypass"
        floor_p = (2.0 * self.floor_rms) ** 2
        g = self.g if self.g is not None else self.g_prior
        any_pass = False
        n_ref = 0
        last_ref = None
        for i in range(len(a) // SUB):
            sub = a[i * SUB:(i + 1) * SUB].astype(np.float64)
            t_sub = t_frame - 0.06 + i * ENV_S + 0.01
            p_up = float(np.mean(sub ** 2))
            self.up_env.append((t_sub, math.sqrt(p_up)))
            p_ref = self._ref_power(t_sub)
            if p_ref is None:
                continue
            n_ref += 1
            last_ref = p_ref
            if p_ref > 1e3:
                self.ratios.append(p_up / p_ref)
            if p_up > self.alpha * g * p_ref + floor_p:
                any_pass = True
        seg = self._seg
        if n_ref == 0:
            self.stats["noref"] += 1
            self.recent.append(0)
            if seg is not None:
                seg["noref"] += 1
            return pcm_bytes, "noref"
        if seg is not None and last_ref is not None:
            seg["ref"].append(last_ref)
        if len(self.ratios) >= 50:
            arr = np.array(self.ratios)
            g_q = float(np.quantile(arr, self.g_quantile))
            self.g = min(g_q, self.g_mult * float(np.median(arr))) if self.g_mult else g_q
        if (self.lag_corr is None and t_frame - self.t_last_lag >= 2.4) or t_frame - self.t_last_lag >= self.lag_every:
            self._estimate_lag(t_frame)
        vad_p = self._vad_prob(a) if any_pass or t_frame < self.hold_until else 0.0
        fresh = any_pass and vad_p >= self.vad_threshold
        if fresh:
            self.hold_until = t_frame + self.hold_s
        p_frame = float(np.mean(a.astype(np.float64) ** 2))
        if fresh or t_frame < self.hold_until:
            self.stats["pass"] += 1
            self.recent.append(1)
            if seg is not None:
                seg["pass" if fresh else "hold"] += 1
                seg["up_pass"].append(p_frame)
            return pcm_bytes, "pass" if fresh else "pass-hold"
        self.stats["echo"] += 1
        self.recent.append(0)
        if seg is not None:
            seg["echo"] += 1
            seg["up_echo"].append(p_frame)
        noise = self.rng.normal(0.0, self.floor_rms, len(a)).astype(np.int16)
        return noise.tobytes(), "echo"
