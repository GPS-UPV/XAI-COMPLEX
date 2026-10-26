"""Generate a real IGJSP instance and classify it with the fixed CEJSP reference."""
import argparse,json,time,sys
from pathlib import Path
from importlib.util import find_spec
from importlib.metadata import version

# Prefer checkout sources for direct execution, while preserving a distribution
# explicitly preloaded by an isolated artifact check.
source=Path(__file__).resolve().parents[1]/'src'
if 'cejsp' not in sys.modules and (source/'cejsp/__init__.py').is_file():
    sys.path.insert(0,str(source))
elif find_spec('cejsp') is None:
    raise SystemExit('Install cejsp[integration]>=0.5.0 in this Python environment first.')

from IGJSP.generador import Generator
from cejsp import CEJSP,classify,CATEGORIES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='Empty output directory')
    parser.add_argument('--jobs',type=int,default=6);parser.add_argument('--machines',type=int,default=6);parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if min(args.jobs,args.machines)<1:parser.error('Dimensions must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    if any(args.output.iterdir()):parser.error('Use an empty output directory')
    started=time.perf_counter()
    instance=Generator(json=False,dzn=False,taillard=False).generate_new_instance(jobs=args.jobs,machines=args.machines,speed=1,ReleaseDateDueDate=0,distribution='uniform',seed=args.seed,instance_type='JSP')
    path=args.output/'instance.json';instance.saveJsonFile(str(path))
    label=CEJSP(path);report=classify(path)
    if label not in CATEGORIES or label!=report['category']:raise AssertionError('Inference APIs disagree')
    result={'passed':True,'generator':'IGJSP','generator_version':version('IGJSP'),'seed':args.seed,'jobs':args.jobs,'machines':args.machines,'label':label,'classification':report,'ground_truth_available':False,'seconds':time.perf_counter()-started}
    (args.output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
