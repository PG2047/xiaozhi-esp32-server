"""残余回声门控：合成信号自检（参考延迟衰减 + 底噪 + 插话），以及上游自带的 silero onnx 推理能跑。"""
from __future__ import annotations

import numpy as np
import pytest

from core.providers.s2s.echo_gate import RATE, EchoGate, load_vad

FR = 960  # 60 ms


def _speechlike(seconds: float, seed: int, amp: float = 6000.0) -> np.ndarray:
    """有音节起伏的带限信号（4 Hz 调制的 200～3000 Hz 噪声），能量包络像说话。"""
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    x = rng.normal(0, 1, n)
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / RATE)
    spec[(f < 200) | (f > 3000)] = 0
    x = np.fft.irfft(spec, n)
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * np.arange(n) / RATE) * np.sin(2 * np.pi * 0.7 * np.arange(n) / RATE)
    x = x / (np.abs(x).max() + 1e-9) * amp * env
    return x


def _run(lag=0.10, echo_db=-28.0, barge_gain=0.8, barge_at=2.0, barge_len=1.5, log_fn=None, g_mult=None):
    # 真机：TTS 下行 RMS 约 3000～5000，设备 AEC 后残余相对参考约 −28 dB（中位）
    ref = _speechlike(5.0, seed=1, amp=20000.0).astype(np.int16)
    spk = _speechlike(barge_len, seed=2, amp=9000.0)
    att = 10 ** (echo_db / 20)
    rng = np.random.default_rng(3)
    n = len(ref) + RATE
    up = rng.normal(0, 25, n)
    d = int(lag * RATE)
    up[d:d + len(ref)] += ref.astype(float) * att
    b0 = int(barge_at * RATE)
    up[b0:b0 + len(spk)] += spk * barge_gain
    up = np.clip(up, -32768, 32767).astype(np.int16)

    gate = EchoGate(vad=None, lag_prior=0.08, lag_max=0.30, start_window=None, log_fn=log_fn,
                    **({"g_mult": g_mult} if g_mult is not None else {}))  # 能量判据（合成信号不是真人声）
    t = 100.0
    gate.on_tts_start(t)
    nf = len(ref) // FR
    for k in range(nf):  # 段首 3 帧一次送出，之后每 60 ms 一帧（与节拍器一致）
        gate.on_downlink(ref[k * FR:(k + 1) * FR].tobytes(), t + (0 if k < 3 else (k - 2) * 0.06))
    gate.on_tts_stop(t + nf * 0.06)
    out, dec = [], []
    for k in range(n // FR):
        o, info = gate.process(up[k * FR:(k + 1) * FR].tobytes(), t + (k + 1) * 0.06, speaking=k * 0.06 < nf * 0.06)
        out.append(np.frombuffer(o, np.int16))
        dec.append(info)
    return gate, up, np.concatenate(out), dec, b0, len(spk)


def _rms(x):
    return float(np.sqrt(np.mean(x.astype(float) ** 2)))


def test_echo_suppressed_and_barge_passes():
    gate, up, out, dec, b0, nb = _run()
    echo_only = slice(int(0.5 * RATE), int(1.9 * RATE))
    barge = slice(b0, b0 + nb)
    assert _rms(up[echo_only]) > 100          # 原始残余回声明显高于底噪
    assert _rms(out[echo_only]) < 25 * 1.5     # 门控后压到底噪附近
    assert _rms(out[barge]) > 0.8 * _rms(up[barge])  # 插话基本原样放行
    first = next(k for k in range(b0 // FR, len(dec)) if dec[k] == "pass")
    assert (first - b0 // FR) * 60 <= 120      # 插话判决延迟 ≤ 120 ms
    assert dec[first] == "pass" and "pass-hold" in dec[first:]   # 新判出的人声标 pass；音节间隙 / 结束后的 300 ms 保持期标 pass-hold
    for k, d in enumerate(dec):
        if d == "pass-hold":
            assert "pass" in dec[max(0, k - 5):k]                 # 保持帧只跟在 300 ms 内的新判人声帧之后


def test_bypass_when_not_playing():
    gate = EchoGate(vad=None)
    pcm = (np.ones(FR, np.int16) * 500).tobytes()
    o, info = gate.process(pcm, 1.0, speaking=False)
    assert info == "bypass" and o == pcm


def test_noref_passthrough_during_underrun():
    gate = EchoGate(vad=None)
    gate.on_tts_start(10.0)
    gate.on_downlink(bytes(FR * 2), 10.0)       # 只播了 60 ms
    pcm = (np.ones(FR, np.int16) * 3000).tobytes()
    o, info = gate.process(pcm, 14.0, speaking=True)  # 4 秒后设备什么都没在放
    assert info == "noref" and o == pcm


def test_lag_estimate_converges():
    gate, *_ = _run(lag=0.18, barge_gain=0.0)
    assert abs(gate.lag - 0.18) <= 0.04


def test_lag_estimate_converges_from_far_prior():
    """先验 80 ms、真实 250 ms（离当前值超过 100 ms）：要么相关 ≥ 0.7 一次采用，要么连续两次一致后采用；6 秒内都该收敛。"""
    gate, *_ = _run(lag=0.25, barge_gain=0.0)
    assert abs(gate.lag - 0.25) <= 0.04 and gate.lag_corr is not None


def test_lag_retry_pacing_before_and_after_lock():
    """估不出 / 被拒后 1 秒再试：锁定前触发周期是 2.4 秒，重试时间不能按 5 秒的周期倒推（否则下一帧就又估一次）。"""
    gate = EchoGate(vad=None)
    gate._estimate_lag(100.0)                                       # 数据不够
    assert 100.06 - gate.t_last_lag < 2.4 and gate.t_last_lag == pytest.approx(100.0 - 2.4 + 0.5)
    gate.lag_corr = 0.7
    gate._estimate_lag(200.0)
    assert gate.t_last_lag == pytest.approx(200.0 - gate.lag_every + 0.5)


def test_g_capped_by_median_so_barge_frames_do_not_raise_threshold():
    """比值池里混着插话子帧时 90% 分位被顶高几十倍；g 以中位数 × g_mult 封顶后仍压得住 −28 dB 的回声。"""
    capped, *_ = _run()
    plain, *_ = _run(g_mult=0)
    # 同一段信号：纯分位数被 1.5 秒插话顶到约 0.07（−11.6 dB，比真实残余 −28 dB 高 16 dB）；封顶后约 0.01，仍高于真实残余比 0.0016
    assert plain.g > 0.03 and capped.g < 0.02 and capped.g > 0.0016 and capped.g_mult == 8.0


def test_segment_diagnostic_line():
    logs = []
    gate, *_ = _run(log_fn=logs.append)
    seg = [l for l in logs if l.startswith("echo gate seg")]
    assert len(seg) == 1 and "g=" in seg[0] and "pass=" in seg[0] and "echo=" in seg[0] and "ref_p50=" in seg[0]
    assert "ratio_echo_p90=" in seg[0] and "vad_reject=" in seg[0] and "echo_max=" in seg[0] and "idle_p90=" in seg[0]
    assert "pass=0" not in seg[0]                                   # 插话段里有放行帧


def test_lag_accept_rule():
    """11-01 真机两种误锁：0.44 的弱相关把先验钉死（200→140）、一次 0.51 把滞后换到 80 ms。"""
    gate = EchoGate(vad=None, lag_prior=0.9)
    assert gate._accept_lag(0.70, 0.52)                 # 锁定前：先验 ±250 ms 内（与起始窗一致）、corr ≥ 0.5 就采用
    assert not gate._accept_lag(0.14, 0.44)             # 锁定前、远、弱：拒
    assert not gate._accept_lag(0.84, 0.44)             # 锁定前、近、弱：也拒（0.25 档只在锁定后当微调）
    assert not gate._accept_lag(0.08, 0.51)             # 远、0.5 出头：先记为候选
    assert gate._lag_cand == 0.08
    assert not gate._accept_lag(0.40, 0.55)             # 下一次落在别处：换候选，仍不采用
    assert gate._lag_cand == 0.40
    assert gate._accept_lag(0.46, 0.52)                 # 连续两次落在同一处（≤ 100 ms）：采用
    assert gate._lag_cand is None
    gate.lag, gate.lag_corr = 0.46, 0.52
    assert gate._accept_lag(0.52, 0.30)                 # 锁定后、近、0.25 档：微调
    assert not gate._accept_lag(0.80, 0.45)             # 锁定后、远、弱：拒
    assert gate._accept_lag(1.00, 0.75)                 # 相关 ≥ 0.7：远也直接采用
    gate.lag = 1.00
    assert gate._accept_lag(0.95, 0.5)                  # 近且 ≥ 0.5：直接采用


def test_silero_onnx_runs_with_upstream_model():
    vad = load_vad()  # 默认路径 = 上游自带的 models/snakers4_silero-vad
    if vad is None:
        pytest.skip("上游 silero onnx 模型不在当前目录下")
    p_silence = vad(np.zeros(512, np.float32))
    assert 0.0 <= p_silence < 0.5
    vad.reset_states()
    assert 0.0 <= vad(np.random.default_rng(0).normal(0, 0.3, 512).astype(np.float32)) <= 1.0


def _short_echo_run(start_window, lag=0.5, echo_db=-24.0):
    """像唤醒应答那样短（1.2 秒）的一段播放，残余回声晚 lag 秒到；返回 (原始回声 RMS, 门控后 RMS, 是否估出过滞后)。"""
    ref = _speechlike(1.2, seed=5, amp=20000.0).astype(np.int16)
    rng = np.random.default_rng(6)
    n = len(ref) + RATE
    up = rng.normal(0, 25, n)
    d = int(lag * RATE)
    up[d:d + len(ref)] += ref.astype(float) * 10 ** (echo_db / 20)
    up = np.clip(up, -32768, 32767).astype(np.int16)
    gate = EchoGate(vad=None, lag_prior=0.2, lag_max=1.2, start_window=start_window)
    t = 100.0
    gate.on_tts_start(t)
    nf = len(ref) // FR
    for k in range(nf):
        gate.on_downlink(ref[k * FR:(k + 1) * FR].tobytes(), t + (0 if k < 3 else (k - 2) * 0.06))
    gate.on_tts_stop(t + nf * 0.06)
    out = []
    for k in range(n // FR):
        o, _ = gate.process(up[k * FR:(k + 1) * FR].tobytes(), t + (k + 1) * 0.06, speaking=k * 0.06 < nf * 0.06)
        out.append(np.frombuffer(o, np.int16))
    out = np.concatenate(out)
    echo = slice(d, d + len(ref))
    return _rms(up[echo]), _rms(out[echo]), gate.lag_corr is not None


def test_start_window_suppresses_echo_before_first_lag_estimate():
    """会话开头先验 200 ms、实际滞后 500 ms 时，窄窗把唤醒应答的回声漏给模型；起始窗（80～650 ms）压得住。"""
    raw, narrow, est1 = _short_echo_run(None)
    _, wide, est2 = _short_echo_run((0.08, 0.65))
    assert not est1 and not est2                    # 这么短还估不出滞后：全靠先验和窗口
    assert raw > 100 and narrow > 2 * 25             # 窄窗：回声明显漏过
    assert wide < 25 * 1.5                           # 起始窗：压到底噪附近


def test_start_window_only_applies_before_estimate():
    gate = EchoGate(vad=None, lag_prior=0.2, start_window=(0.08, 0.65))
    assert gate._window() == pytest.approx((0.45, 0.12))
    gate.lag_corr = 0.7                              # 估出过滞后：回到正常窗口
    assert gate._window() == (gate.tail_back, gate.tail_fwd)
