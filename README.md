---
library_name: mlx-audio
pipeline_tag: text-to-speech
license: mit
language:
- es
base_model:
- ResembleAI/Chatterbox-Multilingual-es-mx-latam
tags:
- mlx
- chatterbox-v3
- spanish
- mixed-precision
---

# Chatterbox LatAm — tested MLX bundle

Portable native Apple Silicon package containing the retained FP32 baseline,
FP16 T3 / FP32 decoder candidate, and local oMLX Q8/Q6/Q5 T3 variants. The runner
applies the verified final-vocoder activation correction and loads mixed
per-layer quantization correctly.

- Public model weights: https://huggingface.co/groxaxo/Chatterbox-LatAm-MLX
- Public runner and reports: https://github.com/groxaxo/chatterbox-latam-mlx-private

## Results actually measured

Thirty Spanish passages, three seeds, one existing voice reference: 90 clips per
model and 270 total. All 270 clips had zero samples at the decoder's ±0.99 hard
limit. Local Whisper-medium transcription uses consistent numeric-format
normalization; literal scores and raw transcripts remain in `evaluation/`.

| Variant | Weights | Normalized ASR WER | Generation / audio time | Peak MLX allocation |
|---|---:|---:|---:|---:|
| FP32 | 2.711 GB | 1.54% | 1.063 | 4.84 GB |
| FP16T3 | 1.639 GB | 1.67% | 1.761 | 3.77 GB |
| oQ8 | 1.264 GB | 1.54% | 0.733 | 3.37 GB |

Q8 is the practical default for this runner. FP16 did not demonstrate a
recognition improvement. Both paired confidence intervals include zero; this
does not establish equivalent perceptual quality. ASR is not MOS, pronunciation
assessment, or speaker similarity. One voice and three seeds were tested.
Timings are one sequential run under current machine load. MLX allocation is
not process RSS. Decimal GB units are used.

The tested FP16-weight runner retains FP32 initial transformer context and KV
cache, so its timing represents that mixed execution path. A fully FP16
activation/cache optimization has not been qualified.

The older three-text test also covered oQ6 and oQ5. It is retained separately as
small-sample evidence and uses Whisper-small; its percentages are not directly
comparable with the larger test.

The packaged runner was also checked from a different working directory: all
five variants passed native Metal loading/forward checks, and fresh Q8 and FP16
speech samples passed the saturation guard. See `validation/portable-runtime.json`.

## Install and test

Use native arm64 Python 3.10 or newer, preferably 3.12. Run:

```bash
bash ./Install-and-Test-Mac.command
```

The installer creates `.venv` inside this checkout and installs the pinned
runtime. From a GitHub clone, it fetches Q8 and the tokenizer if absent from
the public Hugging Face repository; no login is required. The initial
Q8/tokenizer weights are about 1.76 GB. A full model-bundle download contains
all five variants and the pinned tokenizer, about 8.33 GB of weights.

The install check verifies checksums and a finite native Metal transformer
forward. Generate a real speech sample separately; an unconditioned forward is
not an audio-quality certificate.

## Generate Spanish speech

Supply an authorized clean mono reference recording:

```bash
.venv/bin/python run.py generate \
  --model oQ8 \
  --ref-audio /path/to/reference.wav \
  --text '¡Hola! ¿Cómo estás? El niño vio un pingüino y sonrió.' \
  --output outputs/spanish.wav \
  --report outputs/spanish.json
```

Choose `oQ8`, `FP16T3`, `FP32`, `oQ6`, or `oQ5`. Generation is local and uses
only the included speech tokenizer. Existing outputs are not overwritten.
Spanish is the default; use `--language en` for English or another supported
Chatterbox language code when appropriate.
The runner rejects nonfinite or hard-limited output. The package contains no
original voice-reference recording or generated voice-waveform corpus; hashes,
settings, metrics and transcripts preserve the benchmark audit trail.

To use the corrected runner with Codex, Claude Code or OpenCode's Talk skill,
see [Talk integration](docs/talk-integration.md). The included adapter uses Q8
by default and performs synthesis locally with the saturation guard enabled.

To fetch every variant into a GitHub clone:

```bash
.venv/bin/python download_models.py --all
.venv/bin/python run.py verify --all
```

The downloader accepts `--revision COMMIT` to pin a model-repository revision.

Portable conversion helpers are in `tools/`. `convert_fp16.py` accepts
`--source` and `--output`. Run `quantize_omlx.py` in the environment that owns
your oMLX checkout, supplying `--source`, `--output`, `--level 5|6|8`, and
`--omlx-path`. Existing output directories are refused. The original published
weights and reports are the audited artifacts; a new conversion must be tested
again before making quality claims.

## Precision and provenance

Original LatAm source revision: `27e595bf2fe7be0533ca299d9afafcde08b7cca7`.
MLX-Audio runtime: `dad2087f790dfe96b07d5aac1405db4b578b411f`.
Local oMLX quantizer: `70f44cfd652d2013dc0f9776dca7518e32101b00`.

Only 210 T3 attention/MLP matrices are quantized. Voice encoder, conditioning,
embeddings, heads, decoder and vocoder stay at source FP32 precision in all
oQ variants. oQ8 has 210 Q8 matrices; oQ6 has 210 Q6 matrices; oQ5 has 196 Q5
and 14 Q6 matrices. This is the structural oMLX policy, without speech-activation
calibration. The FP16 candidate casts all 292 stored T3 tensors to FP16 and
preserves the other 2,195 tensors exactly at FP32. Per-model reports record
hashes, serialization checks and conversion policy.

`runtime_fix.py` changes only the final vocoder LeakyReLU slope from 0.1 to
0.01, matching the official V3 implementation. Intermediate slopes stay 0.1.
The identical-input diagnostic eliminated measured hard-limit saturation.
Installed library files are not modified. See `docs/vocoder-fix.md`.

`SHA256SUMS.json` covers all packaged files. The runtime source hashes and
dependency versions are pinned in `runtime-lock.json`. The included speech
tokenizer is verified against `speech_tokenizer.lock.json`. These are integrity
checks, not cryptographic author signatures or perceptual-quality guarantees.

Original Resemble AI MIT notice is retained in `LICENSE`; dependency provenance
and notices are in `NOTICE.md`. The generic oMLX LLM server is not used to serve
this speech architecture. No persistent server or autostart is installed.
