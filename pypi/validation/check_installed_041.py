"""Offline wheel/sdist integration checks, sharing the verified dependency environment."""
from pathlib import Path
import subprocess,sys,json,hashlib
root=Path(__file__).resolve().parents[2]
validation=root/'pypi/validation/release-0.4.1';validation.mkdir(exist_ok=True)
python=sys.executable

def run(args):
    result=subprocess.run([python,*args],capture_output=True,text=True)
    with (validation/'installation.log').open('a',encoding='utf-8') as stream:stream.write(result.stdout+result.stderr)
    result.check_returncode()

run(['-m','pip','wheel','--no-cache-dir','--no-deps','--no-build-isolation',str(root/'pypi/dist/cejsp-0.4.1.tar.gz'),'-w',str(validation/'sdist-wheel')])
artifacts={'wheel':root/'pypi/dist/cejsp-0.4.1-py3-none-any.whl','sdist':validation/'sdist-wheel/cejsp-0.4.1-py3-none-any.whl'}
bootstrap="import sys,runpy,pathlib; target=pathlib.Path(sys.argv[1]).resolve(); script=sys.argv[2]; sys.path.insert(0,str(target)); sys.argv=[script]+sys.argv[3:]; import cejsp,xai_complex; assert pathlib.Path(cejsp.__file__).resolve().is_relative_to(target); assert pathlib.Path(xai_complex.__file__).resolve().is_relative_to(target); assert cejsp.__version__=='0.4.1'; runpy.run_path(script,run_name='__main__')"
results={}
for name,wheel in artifacts.items():
    target=validation/(name+'-site')
    run(['-m','pip','install','--no-cache-dir','--no-deps','--no-compile','--target',str(target),str(wheel)])
    output=validation/(name+'-demo')
    run(['-I','-c',bootstrap,str(target),str(root/'pypi/examples/test_igjsp_classifier.py'),'--model',str(root/'models/jsplib-easy-v7-trajectory/reference-accuracy.joblib'),'--output',str(output)])
    results[name]=json.loads((output/'result.json').read_text())
assert results['wheel']['classification']==results['sdist']['classification']
report={'passed':True,'version':'0.4.1','artifacts':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'pypi/dist').glob('cejsp-0.4.1*')},'integration':{name:r['classification'] for name,r in results.items()},'scope':'Separate installation targets with -I; dependencies shared from the working venv; not fresh dependency resolution','model_bundled':False}
(validation/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
