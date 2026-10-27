"""Decodes the parquet shards into 24 kHz int16 wavs plus a metadata jsonl.

Training pool: LibriTTS-R train-clean-100 and the VCTK speakers not held out.
Eval pool: the held-out VCTK speakers, LibriSpeech test-clean and LJSpeech.
"""
import glob, io, json, os, sys
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import librosa

W = os.environ.get("PHONON_WORK", "/home/user/work")
OUT = f"{W}/audio"
SR = 24000
EVAL_VCTK = {"p226", "p265", "p304", "p294"}
MAX_PER_SPK_TRAIN = 60


def decode(b):
    a, sr = sf.read(io.BytesIO(b), dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    if sr != SR:
        a = librosa.resample(a, orig_sr=sr, target_sr=SR, res_type="soxr_hq")
    return a


def job(args):
    path, kind = args
    pf = pq.ParquetFile(path)
    out = []
    per_spk = {}
    for batch in pf.iter_batches(batch_size=64):
        for r in batch.to_pylist():
            if kind == "ltr":
                spk, text, uid = "ltr_" + str(r["speaker_id"]), r["text_normalized"], r["id"]
                split = "train"
            elif kind == "vctk":
                spk, text, uid = "vctk_" + r["speaker_id"], r["text"], r["text_id"] + "_" + os.path.basename(r["file"])
                split = "eval" if r["speaker_id"] in EVAL_VCTK else "train"
            elif kind == "ls":
                spk, text, uid = "ls_" + str(r["speaker_id"]), r["text"].lower().capitalize(), r["id"]
                split = "eval"
            else:
                spk, text, uid = "lj", r["normalized_text"], os.path.basename(r["file"])
                split = "eval"
            if text is None:
                continue
            n = per_spk.get(spk, 0)
            if split == "train" and n >= MAX_PER_SPK_TRAIN:
                continue
            a = decode(r["audio"]["bytes"])
            dur = len(a) / SR
            if dur < 2.0 or dur > 16.0:
                continue
            per_spk[spk] = n + 1
            d = f"{OUT}/{split}/{spk}"
            os.makedirs(d, exist_ok=True)
            fn = f"{d}/{uid.replace('/', '_')}.wav"
            peak = np.abs(a).max()
            if peak > 1.0:
                a = a / peak
            sf.write(fn, a, SR, subtype="PCM_16")
            out.append({"split": split, "spk": spk, "text": text, "path": fn, "dur": round(dur, 3), "src": kind})
    return out


if __name__ == "__main__":
    jobs = [(f, "ltr") for f in sorted(glob.glob(f"{W}/data/ltr/*.parquet"))]
    jobs += [(f, "vctk") for f in sorted(glob.glob(f"{W}/data/vctk/*.parquet"))]
    jobs += [(f, "ls") for f in glob.glob(f"{W}/data/ls_test/*.parquet")]
    jobs += [(f, "lj") for f in glob.glob(f"{W}/data/lj/*.parquet")]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 3) as p:
        res = p.map(job, jobs, chunksize=1)
    with open(f"{OUT}/meta.jsonl", "w") as f:
        for rows in res:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print("done", sum(len(r) for r in res))
