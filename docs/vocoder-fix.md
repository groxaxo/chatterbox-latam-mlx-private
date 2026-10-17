# Final V3 vocoder activation

Official V3 source:
https://huggingface.co/spaces/ResembleAI/Chatterbox-Multilingual-TTS-es-mx-latam/blob/main/chatterbox/src/chatterbox/models/s3gen/hifigan.py

The final PyTorch `F.leaky_relu(x)` uses the default slope 0.01. The pinned MLX
implementation used its intermediate slope 0.1 at that final position as well.
`runtime_fix.py` changes only that final activation to 0.01, preserving all
intermediate 0.1 slopes. On identical FP32 decoder inputs, measured RMS fell
from 0.344 to 0.069 and the fraction at the ±0.99 decoder hard limit fell from
7.23% to zero. See evaluation/small-test/vocoder-diagnostic.json.

The expanded 270-clip test found zero samples at the hard limit in every clip.
This is a saturation result; it does not establish human perceptual quality.
