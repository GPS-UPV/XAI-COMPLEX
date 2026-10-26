"""Check a persisted reference on equivalent representations of existing ft06."""
import argparse
import json
from pathlib import Path
import tempfile
from cejsp import JobShopComplexityClassifier
from cejsp.io import load_instance


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('model',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    model=JobShopComplexityClassifier.load(args.model,trusted=True)
    source=root/'TaillardInstancesJSON/ft06.json'
    inst=load_instance(source); n=len(inst['jobs']); m=inst['n_machines']
    times=[]; ranks=[]; lines=[]
    for job in inst['jobs']:
        by_machine={o['machine']:(i,o['speeds'][0]['processing_time']) for i,o in enumerate(job['operations'])}
        times.extend(by_machine[k][1] for k in range(m)); ranks.extend(by_machine[k][0] for k in range(m))
        lines.append(' '.join(f"{o['machine']} {int(o['speeds'][0]['processing_time'])}" for o in job['operations']))
    with tempfile.TemporaryDirectory() as temp:
        directory=Path(temp); dzn=directory/'ft06.dzn'; txt=directory/'ft06.txt'
        dzn.write_text(f'JOBS=1..{n}; MACHINES=1..{m}; SPEED=1; time=array3d(JOBS,MACHINES,1..SPEED,{times}); energy=array3d(JOBS,MACHINES,1..SPEED,{[0]*(n*m)}); precedence=array2d(JOBS,MACHINES,{ranks});')
        txt.write_text(f'{n} {m}\n'+'\n'.join(lines)+'\n')
        outputs=[model.classify(path) for path in (source,dzn,txt)]
    assert outputs[0]['class_probabilities']==outputs[1]['class_probabilities']==outputs[2]['class_probabilities']
    features=model._features(source)
    assert model.predict([features])[0]==model.predict([features,features])[0]
    args.out.write_text(json.dumps({'passed':True,'model_version':model.model_version_,'formats':['IGJSP JSON','DZN','Taillard'],'same_votes':True,'batch_invariance':True,'category':outputs[0]['category']},indent=2))
    print(args.out.read_text())


if __name__=='__main__':
    main()
