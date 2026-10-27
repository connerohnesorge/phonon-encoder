"""More training speakers, streamed one parquet shard at a time (download, extract, delete).

Adds the VCTK shards not yet used and LibriTTS-R train-clean-360 shards, at most MAX_PER_SPK
utterances per speaker, to audio/train and audio/meta_more.jsonl. The held-out eval speakers
(prep_data.EVAL_VCTK) are skipped.
Usage: prep_more.py N_LTR360_SHARDS
"""
import json, os, subprocess, sys, time, urllib.request

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_data import EVAL_VCTK, OUT, SR, W, decode

MAX_PER_SPK = 20
HF = "https://huggingface.co/datasets"


def listing(repo, folder):
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main/{folder}"
    return [x["path"] for x in json.load(urllib.request.urlopen(url))]


def extract(path, kind, out):
    per_spk = {}
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=64):
        for r in batch.to_pylist():
            if kind == "vctk":
                if r["speaker_id"] in EVAL_VCTK:
                    continue
                spk, text, uid = "vctk_" + r["speaker_id"], r["text"], r["text_id"] + "_" + os.path.basename(r["file"])
            else:
                spk, text, uid = "ltr_" + str(r["speaker_id"]), r["text_normalized"], r["id"]
            if text is None or per_spk.get(spk, 0) >= MAX_PER_SPK:
                continue
            a = decode(r["audio"]["bytes"])
            dur = len(a) / SR
            if dur < 2.0 or dur > 16.0:
                continue
            per_spk[spk] = per_spk.get(spk, 0) + 1
            d = f"{OUT}/train/{spk}"
            os.makedirs(d, exist_ok=True)
            fn = f"{d}/{uid.replace('/', '_')}.wav"
            peak = np.abs(a).max()
            sf.write(fn, a / peak if peak > 1.0 else a, SR, subtype="PCM_16")
            out.write(json.dumps({"split": "train", "spk": spk, "text": text, "path": fn, "dur": round(dur, 3), "src": kind}) + "\n")
            out.flush()
    return len(per_spk)


if __name__ == "__main__":
    n360 = int(sys.argv[1])
    done_path = f"{OUT}/more_done.txt"
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    have_vctk = {f"data/train-{i:05d}" for i in (0, 3, 6, 9, 12, 15, 18, 21, 24)}
    jobs = [("sanchit-gandhi/vctk", p, "vctk") for p in listing("sanchit-gandhi/vctk", "data")
            if p[: len("data/train-00000")] not in have_vctk]
    jobs += [("mythicinfinity/libritts_r", p, "ltr") for p in listing("mythicinfinity/libritts_r", "data/train.clean.360")[:n360]]
    tmp = f"{W}/data/tmp.parquet"
    os.makedirs(f"{W}/data", exist_ok=True)
    with open(f"{OUT}/meta_more.jsonl", "a") as out:
        for repo, p, kind in jobs:
            if p in done:
                continue
            for attempt in range(5):
                r = subprocess.run(["curl", "-sL", "--retry", "3", "-o", tmp, f"{HF}/{repo}/resolve/main/{p}"])
                if r.returncode == 0:
                    break
                time.sleep(2 ** (attempt + 1))
            else:
                raise RuntimeError(f"download failed: {p}")
            n = extract(tmp, kind, out)
            os.remove(tmp)
            with open(done_path, "a") as f:
                f.write(p + "\n")
            print(f"{p}: {n} speakers", flush=True)
    print("done", flush=True)
