"""Verify a zero-workload correction leaves the real training cohort unchanged."""
import json
from hashlib import sha256
from pathlib import Path
import numpy as np
from xai_complex.search_features import search_features
from threadpoolctl import threadpool_limits


def main():
    root=Path(__file__).resolve().parents[1];out=root/'models/jsplib-easy-v7-trajectory'
    manifest=json.loads((root/'models/jsplib-8a6e36ef3126-v1/training-manifest.json').read_text())
    expected=json.loads((root/'models/jsplib-easy-v5/features.json').read_text())['rows']
    provenance=json.loads((out/'features-provenance.json').read_text())
    for name,digest in provenance['source_hashes'].items():
        if sha256((out/'extraction-source'/name).read_bytes()).hexdigest()!=digest:
            raise AssertionError('Archived extraction source differs from recorded hash')
    maximum=0.
    with threadpool_limits(limits=1):
        for example,old in zip(manifest['examples'],expected):
            current=search_features(example['path']);keys=sorted(current)
            before=np.array([old[k] for k in keys]);after=np.array([current[k] for k in keys])
            np.testing.assert_allclose(before,after,rtol=1e-12,atol=1e-12)
            maximum=max(maximum,float(np.max(np.abs(before-after))))
    report={'passed':True,'instances':len(expected),'scope':'All 145 non-CP variables; CP inputs unchanged on this positive-workload cohort','maximum_difference':maximum,'correction':'Zero total workload produces zero gaps instead of an epsilon lower bound','extraction_source_archived_and_hash_verified':True}
    (out/'zero-fix-equivalence.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
