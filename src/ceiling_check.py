"""Re-scores an eval directory with the same-speaker ceiling taken over *different takes* only.

VCTK ships every utterance from two microphones (`..._mic1`, `..._mic2`): one performance,
recorded twice. eval_clone.py's held-out sets contain both copies, so its ceiling includes
pairs of the same take, which are not "two different recordings of the speaker". Here those
pairs are dropped from the ceiling; floor, clone similarity and best-other are unchanged.

Usage: ceiling_check.py EVAL_DIR
"""
import itertools, os, re, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_clone import EVAL, TEXTS, speaker_sets
from verifiers import Verifiers, load16


def take(path):
    return re.sub(r"_mic\d", "", os.path.basename(path))


def main():
    out = sys.argv[1]
    sets, V = speaker_sets(), Verifiers()
    R, T, C = {}, {}, {}
    for s in EVAL:
        held = sets[s][1]
        R[s] = [V.embed(load16(u["path"])) for u in held]
        T[s] = [take(u["path"]) for u in held]
        C[s] = [V.embed(load16(f"{out}/{s}_clone{i}.wav")) for i in range(len(TEXTS))]
    for v in Verifiers.names:
        print(f"[{v}]  speaker      ceil(all) ceil(takes) floor   clone  ratio(all) ratio(takes)  best-other  pass(takes)")
        n = 0
        for s in EVAL:
            pairs = list(itertools.combinations(range(len(R[s])), 2))
            c_all = np.mean([R[s][i][v] @ R[s][j][v] for i, j in pairs])
            c_tk = np.mean([R[s][i][v] @ R[s][j][v] for i, j in pairs if T[s][i] != T[s][j]])
            floor = np.mean([a[v] @ b[v] for o in EVAL if o != s for a in R[s] for b in R[o]])
            sim = {o: np.mean([c[v] @ r[v] for c in C[s] for r in R[o]]) for o in EVAL}
            best = max(sim[o] for o in EVAL if o != s)
            ok = sim[s] / c_tk >= 0.75 and sim[s] > floor and sim[s] > best
            n += ok
            print(f"  {s:<11} {c_all:9.3f} {c_tk:11.3f} {floor:6.3f} {sim[s]:6.3f} {sim[s] / c_all:10.2f} "
                  f"{sim[s] / c_tk:12.2f} {best:10.3f}  {'PASS' if ok else 'fail'}")
        print(f"  passes: {n}/{len(EVAL)}\n")


if __name__ == "__main__":
    main()
