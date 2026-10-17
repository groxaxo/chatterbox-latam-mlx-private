#!/bin/bash
set -euo pipefail
HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  echo 'Native Apple Silicon macOS is required.' >&2
  exit 1
fi
PY=''
for candidate in /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 /opt/homebrew/bin/python3 python3; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import platform,sys; assert platform.machine()=="arm64" and sys.version_info >= (3,10)' >/dev/null 2>&1; then
    PY="$candidate"
    break
  fi
done
if [ -z "$PY" ]; then
  echo 'Install native arm64 Python 3.10 or newer, then rerun.' >&2
  exit 1
fi
if [ ! -e "$HERE/.venv" ]; then
  "$PY" -m venv "$HERE/.venv"
fi
"$HERE/.venv/bin/python" -m pip install -r "$HERE/requirements-mac.txt"
if [ ! -f "$HERE/models/oQ8/model.safetensors" ] || [ ! -f "$HERE/speech-tokenizer/model.safetensors" ]; then
  "$HERE/.venv/bin/python" "$HERE/download_models.py" --model oQ8
fi
"$HERE/.venv/bin/python" "$HERE/run.py" verify --model oQ8
"$HERE/.venv/bin/python" "$HERE/run.py" smoke --model oQ8 --report "$HERE/outputs/install-smoke-$(date +%Y%m%d-%H%M%S).json"
echo 'Verified Q8 weights, pinned speech tokenizer, and a native Metal transformer forward.'
echo 'Generate speech with run.py generate and your own --ref-audio recording.'
