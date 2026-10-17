"""Test the public Talk adapter boundary without requiring a GPU or recording."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TalkAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'run.py').touch()
        (self.root/'reference.wav').touch()
        (self.root/'english.wav').touch()
        python = self.root/'fake_python'
        python.write_text(f'''#!{sys.executable}
import json, os, sys, wave
from pathlib import Path
args = sys.argv[1:]
Path(os.environ['CALL_LOG']).write_text(json.dumps({{'args':args, 'offline':os.environ.get('HF_HUB_OFFLINE')}}))
if os.environ.get('FAKE_FAIL') == '1': sys.exit(7)
output = args[args.index('--output')+1]
with wave.open(output, 'wb') as wav:
    wav.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
    wav.writeframes(b'\\x01\\x00'*100)
''')
        python.chmod(0o755)
        self.env = dict(os.environ, CHATTERBOX_LATAM_HOME=str(self.root),
                        CHATTERBOX_LATAM_PYTHON=str(python), CHATTERBOX_LATAM_REF_AUDIO=str(self.root/'reference.wav'),
                        CHATTERBOX_LATAM_REF_AUDIO_EN=str(self.root/'english.wav'),
                        CALL_LOG=str(self.root/'call.json'), TTS_NO_PLAY='1')
        for name in ['TTS_OUTPUT','CHATTERBOX_LATAM_REPORT','CHATTERBOX_LATAM_MODEL']:
            self.env.pop(name, None)
        self.outputs = []
        self.addCleanup(lambda: [p.unlink(missing_ok=True) for p in self.outputs])

    def generate(self, language='es'):
        result = subprocess.run(['bash', str(ROOT/'tools/talk_tts.sh'), 'Hello, prueba.', language],
                                env=self.env, capture_output=True, text=True)
        if result.returncode == 0:
            self.outputs.append(Path(result.stdout.strip()))
        return result

    def test_q8_language_reference_and_offline_generation_contract(self):
        result = self.generate('en-US')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(Path(result.stdout.strip()).is_file())
        call = json.loads((self.root/'call.json').read_text())
        args = call['args']
        self.assertEqual(args[args.index('--language')+1], 'en')
        self.assertEqual(args[args.index('--model')+1], 'oQ8')
        self.assertEqual(args[args.index('--ref-audio')+1], str(self.root/'english.wav'))
        self.assertEqual(call['offline'], '1')

    def test_each_request_returns_its_own_output(self):
        first, second = self.generate(), self.generate()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertNotEqual(first.stdout, second.stdout)

    def test_model_failure_is_not_replaced_with_another_provider(self):
        self.env['FAKE_FAIL'] = '1'
        result = self.generate()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, '')

    def test_existing_explicit_output_is_preserved_and_rejected(self):
        output = self.root/'existing.wav'
        output.write_bytes(b'original')
        self.env['TTS_OUTPUT'] = str(output)
        result = self.generate()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(output.read_bytes(), b'original')


if __name__ == '__main__':
    unittest.main()
