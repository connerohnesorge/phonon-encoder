"""End-to-end cloning evaluation through the Rust runtime.

For each held-out speaker: a ~10 s reference made of raw recordings goes to `ptts --voice ref.wav`
(the runtime encodes it with the checkpoint's speaker encoder), which speaks fixed sentences.
Each clone is scored against held-out real recordings of the same speaker, never the reference
itself, by three verifiers, next to the speaker's real same-speaker ceiling and cross-speaker floor.

Usage: eval_clone.py CKPT_DIR OUT_DIR [--baseline-voice NAME]
"""
import argparse, itertools, json, os, subprocess, sys
from collections import defaultdict

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from verifiers import Verifiers, load16

W = os.environ.get("PHONON_WORK", "/home/user/work")
PTTS = f"{W}/xn-ptts/target/release/ptts"
EVAL = ["vctk_p226", "vctk_p265", "vctk_p294", "vctk_p304", "ls_1089", "ls_121", "ls_237", "lj"]
if os.environ.get("EVAL_SPK"):
    EVAL = os.environ["EVAL_SPK"].split(",")
TEXTS = [
    "The birch canoe slid on the smooth planks, and the boat drifted toward the far shore.",
    "Glue the sheet to the dark blue background before the paint begins to dry.",
    "It is easy to tell the depth of a well when you drop a stone into it.",
    "These days a chicken leg is a rare dish, but the soup was warm and filling.",
    "The juice of lemons makes fine punch, especially on a hot afternoon in July.",
    "Four hours of steady work faced us, so we started early and kept a quiet pace.",
]
TEXTS = TEXTS[: int(os.environ.get("N_TEXTS", len(TEXTS)))]
N_HELD = 8
REF_S = 10.0


def speaker_sets():
    meta = [json.loads(l) for l in open(f"{W}/audio/meta.jsonl")]
    by = defaultdict(list)
    for m in meta:
        if m["split"] == "eval" and m["spk"] in EVAL:
            by[m["spk"]].append(m)
    sets = {}
    for s in EVAL:
        utts = sorted(by[s], key=lambda m: m["path"])
        ref, n = [], 0.0
        for u in utts:
            if n >= REF_S + (4.0 if os.environ.get("REF_TRIM") else 0.0):
                break
            ref.append(u)
            n += u["dur"]
        rest = [u for u in utts[len(ref):] if u["dur"] >= 3.0][:N_HELD]
        sets[s] = (ref, rest)
    return sets


def trim(a, db=40.0, pad=0.1):
    """Leading and trailing silence: 10 ms frames more than `db` below the loudest one."""
    f = a[: len(a) // 240 * 240].reshape(-1, 240)
    e = 10 * np.log10((f**2).mean(1) + 1e-10)
    on = np.flatnonzero(e > e.max() - db)
    if len(on) == 0:
        return a
    p = int(pad * 100)
    return a[max(0, on[0] - p) * 240 : (on[-1] + 1 + p) * 240]


def write_ref(refs, path):
    clips = [sf.read(u["path"], dtype="float32")[0] for u in refs]
    if os.environ.get("REF_TRIM"):
        clips = [trim(c) for c in clips]
    a = np.concatenate(clips)[: int(REF_S * 24000)]
    sf.write(path, a, 24000, subtype="PCM_16")


def synth(ckpt, voice, text, out, seed):
    cmd = [PTTS, "--model", ckpt, "--lang", "en", "--voice", voice, "--seed", str(seed), text, "-o", out]
    subprocess.run(cmd, check=True, capture_output=True)


def cos(a, b):
    return float(np.dot(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--baseline-voice", default=None, help="speak with this fixed voice instead of cloning")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sets = speaker_sets()
    V = Verifiers()
    emb = {}  # (kind, spk, i) -> {verifier: vec}
    for s, (ref, held) in sets.items():
        ref_path = f"{args.out}/{s}_ref.wav"
        write_ref(ref, ref_path)
        for i, u in enumerate(held):
            emb[("real", s, i)] = V.embed(load16(u["path"]))
        voice = args.baseline_voice or ref_path
        for i, t in enumerate(TEXTS):
            p = f"{args.out}/{s}_clone{i}.wav"
            synth(args.ckpt, voice, t, p, seed=i)
            emb[("clone", s, i)] = V.embed(load16(p))
        print(f"{s}: ref {sum(u['dur'] for u in ref):.1f}s from {len(ref)} recordings, {len(held)} held-out", flush=True)

    report = {}
    for v in Verifiers.names:
        R = {s: [emb[("real", s, i)][v] for i in range(len(sets[s][1]))] for s in EVAL}
        C = {s: [emb[("clone", s, i)][v] for i in range(len(TEXTS))] for s in EVAL}
        rows = {}
        for s in EVAL:
            ceiling = np.mean([cos(a, b) for a, b in itertools.combinations(R[s], 2)])
            floor = np.mean([cos(a, b) for o in EVAL if o != s for a in R[s] for b in R[o]])
            sim = {o: np.mean([cos(c, r) for c in C[s] for r in R[o]]) for o in EVAL}
            own = sim[s]
            best_other = max((sim[o], o) for o in EVAL if o != s)
            rows[s] = {
                "ceiling": ceiling, "floor": floor, "own": own, "ratio": own / ceiling,
                "best_other": best_other[0], "best_other_spk": best_other[1],
                "beats_all_others": bool(own > best_other[0]),
                "pass": bool(own / ceiling >= 0.75 and own > floor and own > best_other[0]),
            }
        report[v] = rows
    json.dump(report, open(f"{args.out}/report.json", "w"), indent=1, default=float)
    for v, rows in report.items():
        print(f"\n[{v}]  speaker      ceiling  floor   clone  ratio  best-other(spk)        pass")
        for s, r in rows.items():
            print(f"  {s:12s} {r['ceiling']:.3f}   {r['floor']:.3f}   {r['own']:.3f}  {r['ratio']:.2f}   "
                  f"{r['best_other']:.3f} ({r['best_other_spk']:10s})  {'PASS' if r['pass'] else 'fail'}")
        print(f"  passes: {sum(r['pass'] for r in rows.values())}/{len(rows)}")


if __name__ == "__main__":
    main()
