#!/usr/bin/env python3
"""Fetch public model assets into a fresh code checkout without authentication."""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

REPO='groxaxo/Chatterbox-LatAm-MLX'
ROOT=Path(__file__).resolve().parent
VARIANTS=['oQ8','FP16T3','FP32','oQ6','oQ5']
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model',choices=VARIANTS,default='oQ8')
parser.add_argument('--all',action='store_true')
parser.add_argument('--revision',help='Optional pinned model-repository commit')
args=parser.parse_args()
models=VARIANTS if args.all else [args.model]
patterns=['speech-tokenizer/*']+[f'models/{v}/*' for v in models]
snapshot_download(REPO,revision=args.revision,allow_patterns=patterns,local_dir=ROOT,token=False)
from run import verify
print(verify(args.model,args.all))
