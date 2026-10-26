"""Audit and test the exact CEJSP 0.5.0 distribution artifacts."""
from pathlib import Path
from hashlib import sha256
import subprocess,sys,json,zipfile,tarfile,re
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'pypi/validation/release-0.5.0-final';OUT.mkdir(exist_ok=True)
DIST=ROOT/'pypi/dist/0.5.0'
artifacts=[DIST/'cejsp-0.5.0-py3-none-any.whl',DIST/'cejsp-0.5.0.tar.gz']
checked=0;entries={}
pattern=re.compile(rb'pypi-[A-Za-z0-9_-]{16,}')
for artifact in artifacts:
    if artifact.suffix=='.whl':
        with zipfile.ZipFile(artifact) as z:members={n:z.read(n) for n in z.namelist() if not n.endswith('/')}
        assert 'cejsp/data/reference.joblib' in members
        assert sha256(members['cejsp/data/reference.joblib']).hexdigest()==sha256((ROOT/'models/jsplib-easy-v7-trajectory/reference-accuracy.joblib').read_bytes()).hexdigest()
    else:
        with tarfile.open(artifact,'r:gz') as t:members={m.name:t.extractfile(m).read() for m in t.getmembers() if m.isfile()}
    for name,data in members.items():
        assert not any(part in name.split('/') for part in ('__pycache__','validation','checkpoints','tests','build','dist')),name
        assert not name.endswith(('.log','.pyc','.pyo')),name
        assert not pattern.search(data),'Credential-like content detected; publication blocked'
        if '/docs/' in name:assert name.rsplit('/',1)[1] in ('USER_GUIDE.md','MODEL_CARD.md','RELEASING.md'),name
        if name.endswith(('.py','.md','.toml')):
            text=data.decode('utf-8-sig')
            assert not re.search('[áéíóúñÁÉÍÓÚÑ]',text),'Non-English text detected in '+name
        checked+=1
    entries[artifact.name]=sorted(members)
(OUT/'archive-contents.json').write_text(json.dumps(entries,indent=2))

def run(args):
    result=subprocess.run([sys.executable,*args],capture_output=True,text=True)
    with (OUT/'checks.log').open('a',encoding='utf-8') as stream:stream.write(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('Artifact verification failed; inspect checks.log')

run(['-m','pip','wheel','--no-cache-dir','--no-deps','--no-build-isolation',str(artifacts[1]),'-w',str(OUT/'sdist-wheel')])
bootstrap="import pathlib,sys,runpy; target=pathlib.Path(sys.argv[1]).resolve(); script=sys.argv[2]; sys.path.insert(0,str(target)); sys.argv=[script]+sys.argv[3:]; import cejsp,xai_complex; assert pathlib.Path(cejsp.__file__).resolve().is_relative_to(target); assert pathlib.Path(xai_complex.__file__).resolve().is_relative_to(target); assert cejsp.__version__=='0.5.0'; runpy.run_path(script,run_name='__main__')"
reports={}
for name,wheel in [('wheel',artifacts[0]),('sdist',OUT/'sdist-wheel/cejsp-0.5.0-py3-none-any.whl')]:
    target=OUT/(name+'-site')
    run(['-m','pip','install','--no-cache-dir','--no-deps','--no-compile','--target',str(target),str(wheel)])
    run(['-I','-c',bootstrap,str(target),str(ROOT/'pypi/examples/test_igjsp_classifier.py'),'--output',str(OUT/(name+'-demo'))])
    reports[name]=json.loads((OUT/(name+'-demo')/'result.json').read_text())
    code="import sys,pathlib; target=pathlib.Path(sys.argv[1]).resolve();sys.path.insert(0,str(target));import cejsp,pytest;assert pathlib.Path(cejsp.__file__).resolve().is_relative_to(target);raise SystemExit(pytest.main([sys.argv[2],'-q']))"
    run(['-I','-c',code,str(target),str(ROOT/'pypi/tests/test_public_api.py')])
assert reports['wheel']['classification']==reports['sdist']['classification']
report={'passed':True,'version':'0.5.0','fixed_reference_bundled':True,'original_reference_bytes_preserved':True,'archive_members_scanned':checked,'no_credentials_or_experimental_artifacts':True,'public_api_tests_per_installation':6,'integration':{k:v['label'] for k,v in reports.items()},'artifacts':{p.name:sha256(p.read_bytes()).hexdigest() for p in artifacts},'scope':'Two separate installation targets and isolated Python processes; dependency environment shared from the verified working venv, not freshly resolved from PyPI','published':False}
(OUT/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
