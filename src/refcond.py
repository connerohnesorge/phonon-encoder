"""The runtime's reference preparation, mirrored in Python: `ptts::utils::condense_speech`
(80 Hz high-pass, pauses cut), the 10 s cap, then loudness to -18 LUFS (`normalize_loudness`)."""
import numpy as np
import pyloudnorm
import scipy.signal as ss

SR = 24000
SPEECH_HIGHPASS_HZ = 80.0
SILENCE_BELOW_DB = 30.0
SPEECH_MARGIN_S = 0.15
MAX_S = 10.0
_meter = pyloudnorm.Meter(SR)


def condense_speech(pcm, sr=SR):
    w0 = 2 * np.pi * SPEECH_HIGHPASS_HZ / sr
    alpha = np.sin(w0) / np.sqrt(2)
    a0, cos = 1 + alpha, np.cos(w0)
    b = np.array([(1 + cos) / 2, -(1 + cos), (1 + cos) / 2]) / a0
    a = np.array([1.0, -2 * cos / a0, (1 - alpha) / a0])
    x = ss.lfilter(b, a, pcm.astype(np.float64)).astype(np.float32)
    frame = sr // 100
    n = len(x) // frame
    if n == 0:
        return x
    f = x[: n * frame].reshape(n, frame)
    e = 10 * np.log10((f.astype(np.float32) ** 2).sum(1) / frame + 1e-12)
    loud = np.sort(e)[int(round((n - 1) * 0.99))]
    m = int(round(SPEECH_MARGIN_S * 100))
    on = e > loud - SILENCE_BELOW_DB
    keep = np.convolve(on, np.ones(2 * m + 1), "same") > 0
    kept = f[keep].reshape(-1)
    return x if len(kept) < sr // 2 else kept


def normalize_loudness(pcm, target=-18.0):
    pcm = pcm - pcm.mean()
    if np.sqrt((pcm.astype(np.float64) ** 2).mean()) < 2e-3:
        return pcm.astype(np.float32)
    L = _meter.integrated_loudness(pcm.astype(np.float64))
    if not np.isfinite(L):
        return pcm.astype(np.float32)
    return (pcm * 10 ** ((target - L) / 20)).astype(np.float32)


def prepare_reference(pcm):
    """What `Synth::add_voice_from_pcm` hands the speaker encoder."""
    x = condense_speech(pcm)[: int(MAX_S * SR)]
    return normalize_loudness(x)
