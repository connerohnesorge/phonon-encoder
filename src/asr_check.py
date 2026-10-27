"""Whisper transcripts of generated clips, with word error rate against the intended text."""
import glob, os, sys, re
import numpy as np, torch
from transformers import pipeline
sys.path.insert(0, os.path.dirname(__file__))
from verifiers import load16
from eval_clone import TEXTS

def norm(t):
    return re.sub(r"[^a-z ]", "", t.lower()).split()

def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d.copy(), i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))

if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", "2")))
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
    ws = []
    for p in sorted(glob.glob(f"{sys.argv[1]}/*_clone*.wav")):
        i = int(re.search(r"clone(\d+)", p).group(1))
        hyp = asr(load16(p))["text"]
        ws.append(wer(TEXTS[i], hyp))
        print(f"{os.path.basename(p)}  WER {ws[-1]:.2f}  {hyp.strip()}")
    print(f"mean WER {np.mean(ws):.3f} over {len(ws)} clips")
