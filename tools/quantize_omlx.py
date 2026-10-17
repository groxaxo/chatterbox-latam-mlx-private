#!/usr/bin/env python3
"""Apply an installed oMLX structural policy to only 210 Chatterbox T3 matrices.

Use the Python environment that owns the oMLX checkout. Every other tensor is
preserved exactly; this tool does not perform activation calibration.
"""
import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',required=True,type=Path)
parser.add_argument('--output',required=True,type=Path)
parser.add_argument('--level',choices=[5,6,8],required=True,type=int)
parser.add_argument('--omlx-path',required=True,type=Path)
args=parser.parse_args()
if args.output.exists():raise FileExistsError(args.output)
sys.path.insert(0,str(args.omlx_path.resolve()))
import mlx.core as mx
from omlx.oq import _get_predicate_bits,_quantize_chunked
import omlx.oq

projections=('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj')
source=mx.load(str(args.source/'model.safetensors'))
if any(v.dtype!=mx.float32 for v in source.values()):raise ValueError('Use the unquantized FP32 baseline')
keys={f't3.tfmr.model.layers.{i}.{p}.weight' for i in range(30) for p in projections}
assert len(keys)==210 and keys.issubset(source)
weights=dict(source)
policy={}
for key in sorted(keys):
    bits,group,mode=_get_predicate_bits(key.removeprefix('t3.tfmr.'),{'num_hidden_layers':30,'hidden_size':1024},args.level,64)
    assert bits and mode=='affine'
    q,s,b=_quantize_chunked(source[key],group,bits,mode)
    mx.eval(q,s,b)
    weights[key]=q
    weights[key[:-7]+'.scales']=s
    weights[key[:-7]+'.biases']=b
    policy[key[:-7]]={'bits':bits,'group_size':group,'mode':mode}
args.output.mkdir(parents=True)
mx.save_safetensors(str(args.output/'model.safetensors'),weights,metadata={'format':'mlx'})
saved=mx.load(str(args.output/'model.safetensors'))
assert set(saved)==set(weights)
for key,expected in weights.items():
    assert saved[key].dtype==expected.dtype and saved[key].shape==expected.shape and bool(mx.array_equal(saved[key],expected)),key
    if key in source and key not in keys:assert bool(mx.array_equal(saved[key],source[key])),key
config=json.loads((args.source/'config.json').read_text())
quant={'bits':args.level,'group_size':64,'mode':'affine'}
config['quantization']=quant
config['quantization_config']={**quant,**policy}
(args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
for name in ['tokenizer.json','Cangjie5_TC.json']:
    shutil.copyfile(args.source/name,args.output/name)
report={'level':args.level,'policy':'oMLX structural T3-only; all non-backbone tensors exact FP32','quantized_matrices':len(policy),'bit_counts':dict(Counter(v['bits'] for v in policy.values())),'activation_calibration':False,'omlx_oq_source_sha256':hashlib.sha256(Path(omlx.oq.__file__).read_bytes()).hexdigest(),'roundtrip_verified':True,'weight_bytes':(args.output/'model.safetensors').stat().st_size}
(args.output/'quantization-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
