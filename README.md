# phonon-encoder

Zero-shot voice cloning from raw reference audio for [Phonon](https://github.com/gradium-ai/xn-ptts)
31m (`gradium/phonon-31m-en`), running in the unmodified Rust runtime path
`ptts --voice reference.wav`.

## Why this is needed

Phonon conditions its flow LM on `speaker_wavs`: 125 frames (10 s at 12.5 Hz) of 512-d latents
from a "speaker Mimi" encoder, projected by the checkpoint's `speaker_wavs.output_proj`. The
runtime already has the audio path (`MimiEnc`, `create_voice --input x.wav`), but the public
checkpoint ships **no speaker encoder** (no `mimi_speaker.*` tensors), and no **codec encoder**
either (only the decoder). Cloning is offered only through the Gradium cloud API.

Checked and ruled out:

- Kyutai's public Mimi encoder is not the speaker encoder: per-channel statistics against the
  catalog voices correlate at r = 0.07, and voice identification from its latents is at chance.
- Pocket TTS's public checkpoint has the same codec architecture, but its encoder weights are
  zeroed, and its decoder differs from Phonon's.

## Method

Everything trains on CPU from public data. No speaker-verification model is used in training.

1. **Codec encoder stand-in** (`train_emain.py`): Kyutai's Mimi encoder (frozen SEANet front) plus a
   512→32 projection, trained to produce Phonon's 32-d codec latents. Two signals: 1,600 synthetic
   (audio, latent) pairs that Phonon itself generated in catalog voices (exact targets, R² 0.94 on
   held-out pairs), and real speech round-tripped through Phonon's frozen released decoder
   (multi-resolution mel loss). It supplies teacher-forcing targets for real speech.
2. **Speaker encoder** (`train_spk.py`): Phonon's `speaker_mimi` architecture (identical to
   Kyutai's Mimi encoder), initialised from Kyutai's weights, SEANet front frozen, transformer and
   12.5 Hz downsampler trained, three phases:
   - *Phase 1*, likelihood: with the prompt encoded from reference audio A of a speaker, the frozen
     flow LM's own flow-matching and end-of-speech loss on the codec latents of a different
     utterance B of that speaker. 264 LibriTTS-R / VCTK training speakers.
   - *Phase 2*, distillation from released voices: for Phonon-generated speech in 92 catalog
     voices, the encoder hears other utterances of the voice. The frozen LM's hidden states under
     that prompt must match its hidden states under the voice's released `speaker_wavs`. Ten
     catalog voices are held out for validation. Half of each batch is still phase 1.
   - *Phase 3*, channel robustness: the same distillation, with the student's reference passed
     through simulated recording channels (room reverb, EQ, band-limiting, high-pass, noise floor),
     one consistent channel per voice and variant. The teacher target stays clean.
3. **Export** (`export_speaker.py`): the encoder is written into the checkpoint as
   `mimi_speaker.*` under the Rust loader's tensor names. `ptts --voice ref.wav` and
   `create_voice --input ref.wav` then clone with no new Rust code for the encoder. PyTorch and
   Rust embeddings agree to 0.26% relative error (`parity.py`; the gap is the loudness meter).
4. **Reference conditioning** (`patches/xn-ptts-condense-speech.patch`, against xn-ptts
   `ac0e152`): `ptts::utils::condense_speech` high-passes the reference at 80 Hz and cuts
   pauses (10 ms frames more than 30 dB below the 99th-percentile frame, keeping 150 ms around
   speech) before the 10 s cap. It is wired into `Synth::add_voice_from_pcm` and `create_voice`.
   Without it, sentence-by-sentence corpora such as VCTK, with long pauses and rumble, produce
   prompts that make the LM loop ("I love you. I love you…"). Two unit tests are included.

## Evaluation

Eval voices are speaker-disjoint from all training and fitting: VCTK p226, p265, p294 and p304,
LibriSpeech test-clean 1089, 121 and 237 (16 kHz audiobook recordings), and LJSpeech (single
speaker, 22 kHz). That is 8 voices, 4 of them non-VCTK. Their speakers' `speaker_wavs` are never
used. Each reference is the speaker's first recordings, raw, capped at 10 s. Each clone is
`ptts --voice ref.wav` on 6 fixed sentences, scored against 8 *held-out* real recordings, never
against the reference.

Per speaker and verifier:

- **ceiling**: mean cosine between pairs of the speaker's real recordings.
- **floor**: mean cosine between the speaker's real recordings and the other speakers'.
- **ratio**: clone-vs-real cosine divided by the ceiling.
- **pass** requires ratio ≥ 0.75, clone above the floor, and the clone closer to its own speaker
  than to any other speaker's real recordings.

Verifiers: SpeechBrain ECAPA-TDNN (primary), WavLM-Base-Plus-SV and Resemblyzer, none used in
training.

| voice | ECAPA ratio | WavLM ratio | Resemblyzer ratio | codec ceiling (ECAPA) |
|---|---|---|---|---|
| vctk_p226 | 0.21 ✓ | 0.91 ✓ **PASS** | 0.72 ✓ | 0.75 |
| vctk_p265 | 0.22 | 0.89 ✓ **PASS** | 0.69 ✓ | 0.62 |
| vctk_p294 | 0.35 ✓ | 0.94 | 0.88 ✓ **PASS** | 0.62 |
| vctk_p304 | 0.12 | 0.85 | 0.71 | 0.63 |
| ls_1089 | 0.19 | 0.94 ✓ **PASS** | 0.73 ✓ | 0.62 |
| ls_121 | 0.47 ✓ | 0.87 ✓ **PASS** | 0.79 ✓ **PASS** | 0.76 |
| ls_237 | 0.30 | 0.92 | 0.79 ✓ **PASS** | 0.57 |
| lj | 0.36 ✓ | 0.87 | 0.73 ✓ | 0.68 |
| **passes** | **0/8** | **4/8** | **3/8** | |

✓ = the clone is closer to its own speaker than to every other reference. Every clone is above
the floor on all three verifiers.

| verifier | mean ratio, cloned | mean ratio, no cloning (default voice) | codec ceiling | clone ÷ codec ceiling |
|---|---|---|---|---|
| ECAPA | 0.28 | 0.12 | 0.66 | 0.42 |
| WavLM-SV | 0.90 | 0.71 | 0.98 | 0.92 |
| Resemblyzer | 0.76 | 0.64 | 0.93 | 0.82 |

Intelligibility (Whisper base.en): 47 of 48 clones read the sentence, median WER 0.0. One p304
clip falls into a repetition loop.

**The ECAPA target is not met.** The codec ceiling column explains most of the gap. Round-tripping
a speaker's real recordings through Phonon's own released decoder keeps only 0.57 to 0.76 of the
ECAPA ceiling (`codec_ceiling.py`, using the rebuilt codec encoder). Any clone is rendered by that
decoder, so ≥ 0.75 on ECAPA is at or beyond what Phonon 31m can produce for 6 of the 8 voices,
however good the encoder. The encoder reaches 0.42 of that codec bound on ECAPA. On held-out
*catalog* voices (Phonon-rendered references), clones reach 0.74 of the true voice's own
consistency, so the remaining loss is mostly the gap between real recordings and Phonon's
acoustic domain.

Raw numbers: `results/`.

## Follow-up: more data (O1) and an unfrozen front end (O2)

Both were tried after the results above and **both made end-to-end cloning worse**, so the
shipped weights stay the phase-3 encoder.

- **O1**, more data: 667 training speakers instead of 264 (348 more from LibriTTS-R
  train-clean-360 and 81 more from VCTK, eval speakers excluded), and 3,200 teacher utterances
  instead of 1,600 (`prep_more.py`, `gen_synth.py`).
- **O2**, front end unfrozen: the encoder runs on raw audio during training
  (`train_spk_raw.py`), with the SEANet tail trainable (`model.9` onwards), a fresh random
  recording channel per synthetic reference, and every reference passed through `refcond.py`,
  a Python mirror of the runtime's conditioning (it matches Rust within 0.6%).
- **O1 only** is an ablation: the same training but with SEANet frozen, isolating O1.

Both continued from the phase-3 encoder, with batch 6 and learning rate 5e-5.

| run | steps | ECAPA mean ratio | WavLM | Resemblyzer | ECAPA beats every other voice | passes (ECAPA / WavLM / Resemblyzer) | intelligible (WER ≤ 0.3) |
|---|---|---|---|---|---|---|---|
| phase 3 (shipped) | — | **0.28** | **0.90** | **0.76** | 4/8 | 0 / 4 / 3 | **47/48** |
| O1 + O2 | 1000 | 0.22 | 0.83 | 0.74 | 4/8 | 0 / 3 / 3 | 36/48 |
| O1 only | 600 | 0.25 | 0.87 | 0.75 | 4/8 | 0 / 4 / 3 | 39/48 |

Teacher-forced validation *improved* in both runs: held-out-voice distillation error fell from
0.0237 to 0.0190 (O1+O2) and 0.0184 (O1 only), and real-speaker loss from 0.628 to 0.613 and
0.612 (`results/validation_*.txt`). Free-running generation still got worse, mostly on VCTK
voices, which fall back into repetition loops. So the teacher-forced objectives stop predicting
end-to-end quality once the encoder moves further from the phase-3 solution, and unfreezing the
front end makes that worse. Raw numbers: `results/eval_o1_o2.json`, `results/eval_o1_only.json`.

## Use

```sh
# speaker encoder weights -> a checkpoint folder with mimi_speaker.* (needs the Phonon 31m
# checkpoint in $PHONON_WORK/model and Kyutai's Mimi tokenizer file in $PHONON_WORK/kmimi)
python src/export_speaker.py weights/speaker_encoder_head.fp16.safetensors ckpt
# xn-ptts with patches/xn-ptts-condense-speech.patch applied
ptts --model ckpt --lang en --voice reference.wav "Text to say." -o out.wav
```

Clone a voice only with the speaker's consent.
