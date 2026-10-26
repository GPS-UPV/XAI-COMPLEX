"""Checkpointed bottleneck relaxations, aligned with the frozen JSPLIB cohort."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import time
from threadpoolctl import threadpool_limits
from xai_complex.bottleneck import relaxation_features

ROOT=Path(__file__).resolve().parents[1]


def extract(path,base,budget):
    with threadpool_limits(limits=1):
        return relaxation_features(path,base_features=base,budget=budget)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--budget',type=float,default=.02)
    args=ap.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'models/jsplib-8a6e36ef3126-v1/training-manifest.json').read_text())
    examples=manifest['examples'];base_path=ROOT/'models/jsplib-easy-v7-trajectory/features.json'
    base=json.loads(base_path.read_text())
    meta=json.loads(base_path.with_name('features-provenance.json').read_text())
    if meta['instance_hashes']!=[e['sha256'] for e in examples]:raise ValueError('Base matrix alignment differs')
    for e in examples:
        if sha256(Path(e['path']).read_bytes()).hexdigest()!=e['sha256']:raise ValueError('Dataset content changed')
    key={'instance_hashes':[e['sha256'] for e in examples],'source_sha256':sha256((ROOT/'pypi/src/xai_complex/bottleneck.py').read_bytes()).hexdigest(),'base_sha256':sha256(base_path.read_bytes()).hexdigest(),'budget':args.budget,'ortools':importlib.metadata.version('ortools')}
    provenance=out/'features-provenance.json'
    if provenance.exists() and json.loads(provenance.read_text())!=key:raise ValueError('Incompatible cache provenance')
    provenance.write_text(json.dumps(key,indent=2))
    partial=out/'features-partial.json'
    rows=json.loads(partial.read_text()) if partial.exists() else [None]*len(examples)
    started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(extract,e['path'],base[i],args.budget):i for i,e in enumerate(examples) if rows[i] is None}
        for future in as_completed(futures):
            rows[futures[future]]=future.result()
            temporary=out/'features-partial.pending';temporary.write_text(json.dumps(rows));temporary.replace(partial)
            count=sum(r is not None for r in rows)
            if count%20==0:print(json.dumps({'features':count,'total':len(rows),'seconds':time.perf_counter()-started}),flush=True)
    (out/'features.json').write_text(json.dumps(rows))
    print(json.dumps({'complete':True,'seconds':time.perf_counter()-started}),flush=True)


if __name__=='__main__':main()
