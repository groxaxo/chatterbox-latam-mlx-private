# Use the fixed runner with Talk

Install on native Apple Silicon macOS:

```bash
git clone https://github.com/groxaxo/chatterbox-latam-mlx-private.git \
  "$HOME/.local/share/chatterbox-latam-mlx"
cd "$HOME/.local/share/chatterbox-latam-mlx"
bash ./Install-and-Test-Mac.command
```

Configure your existing Talk installation with a clean mono voice reference
that you are authorized to use:

```bash
export CHATTERBOX_LATAM_HOME="$HOME/.local/share/chatterbox-latam-mlx"
export CHATTERBOX_LATAM_REF_AUDIO="/path/to/reference.wav"
export TALK_TTS_ENGINE=chatterbox-latam-mlx
export TTS_ENGINE=chatterbox-latam-mlx
export TTS_SH="$CHATTERBOX_LATAM_HOME/tools/talk_tts.sh"
export TTS_LANG_SH="/path/to/talk/tts_lang.sh"
```

Put these exports in your shell configuration for persistence. Give the same
variables to a desktop client's Talk process if it does not inherit your shell.
Codex, Claude Code and OpenCode can use the same Talk checkout and runner.

Test synthesis without opening the microphone or playing audio:

```bash
bash /path/to/talk/talk.sh render 'Hola, esta es una prueba.' /tmp/talk-check.wav es
```

The adapter implements Talk's `TTS_SH` contract: text and optional language are
positional arguments, `TTS_NO_PLAY=1` returns one generated WAV path, and errors
return a nonzero status. It forwards the detected language to the multilingual
runner. `CHATTERBOX_LATAM_REF_AUDIO_EN` optionally supplies an English reference;
otherwise the common reference is used. `CHATTERBOX_LATAM_MODEL` selects a
downloaded variant; Q8 (`oQ8`) is the default.

Each request uses a separate temporary output. An explicit `TTS_OUTPUT` must
name a new file. `CHATTERBOX_LATAM_REPORT` optionally saves the runner's JSON
generation report. The adapter applies the fixed runner and saturation guard
on every request. Synthesis does not start a persistent server.
