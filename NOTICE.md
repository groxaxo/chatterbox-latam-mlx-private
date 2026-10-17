# Provenance and notices

Model weights derive from ResembleAI/Chatterbox-Multilingual-es-mx-latam at
27e595bf2fe7be0533ca299d9afafcde08b7cca7, with shared ResembleAI/chatterbox
assets at 5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18. The original Resemble AI
MIT license is retained verbatim in LICENSE.

The package includes mlx-community/S3TokenizerV2 weights and config at
e0c9886f0e1c35ae85b1f27277416fb19fc72bec. Their exact upstream file hashes are
recorded in speech_tokenizer.lock.json. These files were dependencies of the
tested local runtime; the original public Q8 release did not redistribute them.

MLX-Audio is installed separately at commit
dad2087f790dfe96b07d5aac1405db4b578b411f, under its own license and third-party
notices. The upstream V3 HiFi-GAN implementation contains Apache-2.0 notices;
this package does not vendor that upstream source. runtime_fix.py patches the
installed class in memory to match the final activation's original semantics.

oMLX conversion provenance is recorded in each quantization report. Its
Apache-2.0 implementation remains an external dependency, not a vendored copy.
The original voice reference and generated waveform corpus remain local.
Evaluation reports include hashes and synthetic reference texts/transcripts.
No credentials or client configuration are part of this package.
