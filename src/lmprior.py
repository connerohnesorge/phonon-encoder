"""Reference latents that are both faithful and likely under the frozen LM.

The codec-encoder stand-in's latents for real speech decode to the recording only roughly, and
latents optimised for the decoder alone fall off the LM's manifold (the LM then pulls generation
back to a generic voice). Here a reference segment's latents start at E_main and minimise
    mel(decoder(z), x) + LAM * LM flow-matching loss(z | voice prompt, transcript),
so the LM is given audio context it models well that still sounds like the reference.
"""
import os

import numpy as np
import torch

from train_emain import MelLoss, load_decoder
from train_spk import flow_loss
from train_spk_raw import chat, pad

LAM = float(os.environ.get("LM_PRIOR", "0"))
LM_STEPS = int(os.environ.get("LM_PRIOR_STEPS", "200"))
_dec = _mel = None


def refine_lm(z0, x, text, prompt, model, proc, lam=None, steps=None):
    """z0 [T, 32] (numpy, codec-latent scale), x audio at 24 kHz padded to T frames -> refined z0."""
    global _dec, _mel
    if _dec is None:
        _dec, _mel = load_decoder(), MelLoss()
        for p in _dec.parameters():
            p.requires_grad_(False)
    lam = LAM if lam is None else lam
    steps = LM_STEPS if steps is None else steps
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32))[None]
    inputs = chat(proc, [(np.asarray(z0), text)], [len(prompt)])
    lat, mask = pad([prompt.detach()])
    z = torch.from_numpy(np.asarray(z0, dtype=np.float32)).clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=0.02)
    gen = torch.Generator().manual_seed(0)
    for _ in range(steps):
        loss = _mel(_dec.decode(z[None]).audio_values[:, 0, : xt.shape[-1]], xt)
        if lam:
            inputs["labels"] = z[None]
            lf, _ = flow_loss(model, inputs, lat, mask, n_noise=4, gen=gen)
            loss = loss + lam * lf
        opt.zero_grad()
        loss.backward()
        opt.step()
    return z.detach().numpy()
