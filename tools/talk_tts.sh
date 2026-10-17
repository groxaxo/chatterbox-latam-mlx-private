#!/bin/bash
# Talk's TTS_SH interface: text, optional language; TTS_NO_PLAY=1 returns a WAV.
set -euo pipefail

RUNNER_HOME="${CHATTERBOX_LATAM_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
RUNNER_PYTHON="${CHATTERBOX_LATAM_PYTHON:-$RUNNER_HOME/.venv/bin/python}"
MODEL="${CHATTERBOX_LATAM_MODEL:-oQ8}"
TEXT="${1:?Speech text is required}"
LANGUAGE="${2:-es}"
LANGUAGE="$(printf '%s' "$LANGUAGE" | tr '[:upper:]' '[:lower:]' | cut -d- -f1)"
REFERENCE="${CHATTERBOX_LATAM_REF_AUDIO:-$HOME/voices/qwen3-mlx-carina-es.wav}"
if [ "$LANGUAGE" = en ]; then
    REFERENCE="${CHATTERBOX_LATAM_REF_AUDIO_EN:-$REFERENCE}"
fi
[ -x "$RUNNER_PYTHON" ] || { echo 'Chatterbox LatAm: run Install-and-Test-Mac.command first.' >&2; exit 1; }
[ -f "$RUNNER_HOME/run.py" ] || { echo 'Chatterbox LatAm: runner not found.' >&2; exit 1; }
[ -f "$REFERENCE" ] || { echo 'Chatterbox LatAm: set CHATTERBOX_LATAM_REF_AUDIO to your mono reference recording.' >&2; exit 1; }

BASE="$(mktemp "${TMPDIR:-/tmp}/talk-latam.XXXXXX")"
rm -f "$BASE"
WAVE="$BASE.wav"
KEEP=0
trap 'if [ "$KEEP" != 1 ]; then rm -f "$WAVE"; fi' EXIT
ARGS=(generate --model "$MODEL" --language "$LANGUAGE" --ref-audio "$REFERENCE"
      --text "$TEXT" --seed "${CHATTERBOX_LATAM_SEED:-42}" --output "$WAVE")
if [ -n "${CHATTERBOX_LATAM_REPORT:-}" ]; then
    ARGS+=(--report "$CHATTERBOX_LATAM_REPORT")
fi
echo "[tts] Chatterbox LatAm MLX $MODEL lang=$LANGUAGE; corrected vocoder; local synthesis" >&2
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$RUNNER_PYTHON" "$RUNNER_HOME/run.py" "${ARGS[@]}" >/dev/null
[ -s "$WAVE" ] || { echo 'Chatterbox LatAm: no audio produced.' >&2; exit 1; }
if [ -n "${TTS_OUTPUT:-}" ]; then
    [ ! -e "$TTS_OUTPUT" ] || { echo 'Chatterbox LatAm: output already exists.' >&2; exit 1; }
    mv "$WAVE" "$TTS_OUTPUT"
    WAVE="$TTS_OUTPUT"
fi
if [ "${TTS_NO_PLAY:-0}" = 1 ]; then
    KEEP=1
    printf '%s\n' "$WAVE"
else
    case "$(uname -s)" in
        Darwin) afplay "$WAVE" ;;
        *) ffplay -nodisp -autoexit -loglevel quiet "$WAVE" ;;
    esac
fi
