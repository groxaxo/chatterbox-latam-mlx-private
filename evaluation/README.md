# Evaluation evidence

The expanded comparison used 30 Spanish passages, seeds 42/43/44, one existing
Carina Spanish reference, native Metal generation and cached Whisper-medium
transcription on CPU. Each model generated 90 clips. `summary.json` contains
both literal and number-normalized WER, per-seed counts, paired passage-cluster
bootstrap intervals, timings, memory, and the mixed FP16 context diagnostic.
`number-normalized-transcripts.json` retains both score definitions and the
raw recognizer transcripts. `scoring.py` implements the common normalization
and semantic checks. Generation results contain hashes of the original WAVs.

The original reference recording and generated waveform corpus are not included
in either repository. `REFERENCE_NOT_INCLUDED` marks the original reference
path; its digest identifies the voice used for the measured experiment.
`local-waveform-evidence/` paths are logical IDs for locally retained evidence,
not missing model dependencies. Reproduce new speech with your own authorized
reference recording; the particular benchmark cannot be recreated without
providing the original matching reference locally.

The previous three-text Whisper-small comparison is kept in `small-test/`.
It is a different corpus and recognizer, with substantially less evidence.
