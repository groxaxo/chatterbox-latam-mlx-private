#!/usr/bin/env python3
"""Portable T3-only FP16 conversion; preserve every other tensor exactly."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import mlx.core as mx

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',required=True,type=Path)
parser.add_argument('--output',required=True,type=Path)
args=parser.parse_args()
if args.output.exists():raise FileExistsError(args.output)
weights=mx.load(str(args.source/'model.safetensors'))
if any(v.dtype!=mx.float32 for v in weights.values()):
    raise ValueError('Expected the unquantized source FP32 baseline')
converted={k:v.astype(mx.float16) if k.startswith('t3.') else v for k,v in weights.items()}
mx.eval(converted)
if not all(bool(mx.all(mx.isfinite(v))) for v in converted.values()):raise ValueError('Nonfinite cast')
args.output.mkdir(parents=True)
mx.save_safetensors(str(args.output/'model.safetensors'),converted,metadata={'format':'mlx'})
saved=mx.load(str(args.output/'model.safetensors'))
assert set(saved)==set(converted)
for key,expected in converted.items():
    assert saved[key].dtype==expected.dtype and saved[key].shape==expected.shape and bool(mx.array_equal(saved[key],expected)),key
for name in ['config.json','tokenizer.json','Cangjie5_TC.json']:
    shutil.copyfile(args.source/name,args.output/name)
report={'policy':'All stored t3.* FP32 to FP16; all other tensors unchanged','source_tensors':len(weights),'t3_fp16_tensors':sum(k.startswith('t3.') for k in weights),'bytes':(args.output/'model.safetensors').stat().st_size,'roundtrip_verified':True}
(args.output/'precision-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
