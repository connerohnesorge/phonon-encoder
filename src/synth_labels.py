"""Recovers which catalog voice and text each synthetic item used, by replaying gen_synth's
seeded `random` calls (generation consumes no Python randomness)."""
import glob, json, random, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from gen_synth import texts, W

def labels(shard, n_items, batch):
    rnd = random.Random(shard + 1234)
    voices = sorted(glob.glob(f"{W}/voices/voices/*.safetensors"))
    T = texts()
    out = []
    while len(out) < n_items:
        for _ in range(batch):
            v = rnd.choice(voices)
            t = rnd.choice(T)
            out.append({"voice": os.path.basename(v)[:-12], "text": t})
    return out[:n_items]

if __name__ == "__main__":
    lab = {f"shard{s}": labels(s, 800, 8) for s in (0, 1)}
    json.dump(lab, open(f"{W}/synth/labels.json", "w"))
    print({k: len(v) for k, v in lab.items()}, lab["shard0"][:2])
