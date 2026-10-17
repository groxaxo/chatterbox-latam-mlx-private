#!/usr/bin/env python3
"""Verified local LatAm MLX bundle: verify, Metal smoke, or Spanish speech."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import importlib.metadata as md
import json
import platform
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
VARIANTS=('oQ8','FP16T3','FP32','oQ6','oQ5')

def digest(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def verify(model:str='oQ8',all_files:bool=False)->dict:
    manifest=json.loads((ROOT/'SHA256SUMS.json').read_text())
    verified=0
    for name,expected in manifest.items():
        if not all_files and name.startswith('models/') and not name.startswith(f'models/{model}/'):
            continue
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Manifest path escapes package')
        if not path.is_file():
            raise FileNotFoundError(f'Missing {name}. Run download_models.py --model {model}, or --all.')
        if digest(path)!=expected:
            raise ValueError('Checksum mismatch: '+name)
        verified+=1
    return {'checksums_verified':verified,'scope':'all' if all_files else model}

def load(model_name:str):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_audio.tts.models.chatterbox.chatterbox import Model
    from runtime_fix import apply
    lock=json.loads((ROOT/'runtime-lock.json').read_text())
    for package,version in lock['versions'].items():
        if md.version(package)!=version:
            raise RuntimeError(f'{package} must be {version}; install requirements-mac.txt')
    for module,expected in lock['module_sha256'].items():
        actual=Path(importlib.import_module(module).__file__)
        if digest(actual)!=expected:
            raise RuntimeError('Pinned runtime source changed: '+module)
    if not mx.metal.is_available() or platform.system()!='Darwin' or platform.machine()!='arm64':
        raise RuntimeError('This package is qualified for native Apple Silicon Metal only')
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(512*1024*1024)
    apply()
    folder=ROOT/'models'/model_name
    token_dir=ROOT/'speech-tokenizer'
    token_lock=json.loads((ROOT/'speech_tokenizer.lock.json').read_text())
    for name,expected in token_lock['sha256'].items():
        if digest(token_dir/name)!=expected:
            raise ValueError('Pinned speech tokenizer checksum mismatch: '+name)
    config=json.loads((folder/'config.json').read_text())
    layers={k:v for k,v in config.get('quantization_config',{}).items() if isinstance(v,dict)}
    original_quantize=nn.quantize
    def mixed_quantize(model,group_size=None,bits=None,class_predicate=None,**kwargs):
        def predicate(path,module):
            if class_predicate is not None and not class_predicate(path,module):
                return False
            if path not in layers:
                raise ValueError('Missing per-layer quantization policy: '+path)
            return dict(layers[path])
        return original_quantize(model,group_size=group_size,bits=bits,class_predicate=predicate,**kwargs)
    def local_snapshot(repo_id,*args,**kwargs):
        if repo_id!=token_lock['repository']:
            raise RuntimeError('Unexpected runtime download request: '+str(repo_id))
        return str(token_dir)
    mx.random.seed(20261017)
    start=time.perf_counter()
    with patch('huggingface_hub.snapshot_download',side_effect=local_snapshot),patch.object(nn,'quantize',mixed_quantize):
        model=Model.from_pretrained(folder)
    if layers:
        modules=dict(tree_flatten(model.leaf_modules(),is_leaf=nn.Module.is_module))
        for path,expected in layers.items():
            module=modules[path]
            if not isinstance(module,nn.QuantizedLinear) or module.bits!=expected['bits'] or module.group_size!=expected['group_size']:
                raise ValueError('Loaded quantization differs from manifest: '+path)
    if model_name=='FP16T3':
        values=dict(tree_flatten(model.parameters()))
        stored=mx.load(str(folder/'model.safetensors'))
        for key,value in stored.items():
            if values[key].dtype!=value.dtype:
                raise ValueError('Loaded precision mismatch: '+key)
        del stored
    return model,{'model':model_name,'backend':str(mx.default_device()),'load_seconds':time.perf_counter()-start,'runtime_source_verified':True,'local_tokenizer_verified':True}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ['verify','smoke','generate']:
        p=sub.add_parser(name)
        p.add_argument('--model',choices=VARIANTS,default='oQ8')
        p.add_argument('--report',type=Path)
        if name=='verify':p.add_argument('--all',action='store_true')
        if name=='generate':
            p.add_argument('--text',required=True)
            p.add_argument('--ref-audio',required=True,type=Path)
            p.add_argument('--output',required=True,type=Path)
            p.add_argument('--seed',default=42,type=int)
    args=parser.parse_args()
    if args.report and args.report.exists():raise FileExistsError(args.report)
    report=verify(args.model,getattr(args,'all',False))
    if args.command!='verify':
        import mlx.core as mx
        import numpy as np
        model,details=load(args.model)
        report.update(details)
        if args.command=='smoke':
            from mlx_audio.tts.models.chatterbox.tokenizer import MTLTokenizer
            tokenizer=MTLTokenizer(ROOT/'models'/args.model/'tokenizer.json',text_preprocessing='NFKD,fullcase')
            ids=tokenizer.encode('¡Hola! El niño vio un pingüino.',language_id='es')[:,:8]
            logits=model.t3.speech_head(model.t3.tfmr.model(inputs=None,input_embeddings=model.t3.text_emb(ids)))
            mx.eval(logits)
            if list(logits.shape)!=[1,ids.shape[1],8194] or not bool(mx.all(mx.isfinite(logits))):
                raise ValueError('Transformer smoke failed')
            report.update(logits_shape=list(logits.shape),logits_finite=True,audio_generated=False,test='Unconditioned transformer forward; this is not speech-quality assessment')
        else:
            import soundfile as sf
            if args.output.exists():raise FileExistsError(args.output)
            reference,sr=sf.read(args.ref_audio,dtype='float32',always_2d=True)
            if reference.shape[1]!=1 or not reference.size or not np.isfinite(reference).all() or float(np.sqrt(np.mean(reference.astype(np.float64)**2)))<1e-5:
                raise ValueError('Reference must be nonempty, finite, non-silent mono audio')
            mx.random.seed(args.seed)
            mx.reset_peak_memory()
            parts=[]
            start=time.perf_counter()
            for result in model.generate(text=args.text,lang_code='es',ref_audio=str(args.ref_audio.resolve()),temperature=.8,exaggeration=.5,cfg_weight=.5,repetition_penalty=1.2,min_p=.05,top_p=1.,max_new_tokens=1200,stream=False):
                mx.eval(result.audio)
                parts.append(np.asarray(result.audio,dtype=np.float32).reshape(-1))
            if not parts:raise ValueError('No audio produced')
            audio=np.concatenate(parts)
            elapsed=time.perf_counter()-start
            saturation=float(np.mean(np.abs(audio)>=.989999))
            if not np.isfinite(audio).all() or saturation>=.001:
                raise ValueError(f'Invalid or saturated decoder audio: {saturation:.4%} at hard limit')
            args.output.parent.mkdir(parents=True,exist_ok=True)
            sf.write(args.output,audio,model.sample_rate,subtype='FLOAT')
            report.update(audio_generated=True,output=str(args.output),sample_rate=model.sample_rate,audio_seconds=len(audio)/model.sample_rate,generation_seconds=elapsed,rtf=elapsed/(len(audio)/model.sample_rate),decoder_limit_fraction=saturation,peak_mlx_memory_bytes=mx.get_peak_memory(),audio_sha256=digest(args.output),human_listening_assessed=False)
    report['status']='PASS'
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    main()
