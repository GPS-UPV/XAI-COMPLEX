"""Minimal fixed-reference CEJSP inference."""
import sys
from pathlib import Path
source=Path(__file__).resolve().parents[1]/'src'
if 'cejsp' not in sys.modules and (source/'cejsp/__init__.py').is_file():
    sys.path.insert(0,str(source))
from cejsp import CEJSP,classify

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('instance',help='JSON, DZN or Taillard file')
    args=parser.parse_args()
    print(CEJSP(args.instance))
    print(classify(args.instance))
