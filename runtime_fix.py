"""Match the official V3 vocoder's final F.leaky_relu(x) default (0.01).

Upstream source:
https://huggingface.co/spaces/ResembleAI/Chatterbox-Multilingual-TTS-es-mx-latam/blob/main/chatterbox/src/chatterbox/models/s3gen/hifigan.py
The intermediate upsampling activations intentionally remain at slope 0.1.
Only this process is patched; installed packages and published files are intact.
"""
import inspect
import textwrap
from mlx_audio.tts.models.chatterbox.s3gen import hifigan

def apply():
    source = textwrap.dedent(inspect.getsource(hifigan.HiFTGenerator.decode))
    old = 'x = nn.leaky_relu(x, negative_slope=self.lrelu_slope)\n    # conv_post'
    assert source.count(old) == 1, 'Pinned runtime source changed'
    source = source.replace(old, 'x = nn.leaky_relu(x, negative_slope=0.01)\n    # conv_post')
    namespace = dict(vars(hifigan))
    exec(compile(source, '<v3-final-activation-parity>', 'exec'), namespace)
    hifigan.HiFTGenerator.decode = namespace['decode']
