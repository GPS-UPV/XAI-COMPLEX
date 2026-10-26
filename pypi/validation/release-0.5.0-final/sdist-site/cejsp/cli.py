"""Command-line inference with the bundled CEJSP reference."""
import argparse
import json


def main(argv=None):
    parser=argparse.ArgumentParser(prog='cejsp',description='Classify job shop instances using the fixed bundled CEJSP reference.')
    sub=parser.add_subparsers(dest='command',required=True)
    command=sub.add_parser('classify',help='Classify a JSON, DZN or Taillard instance')
    command.add_argument('input',help='Path to the instance file')
    command.add_argument('--details',action='store_true',help='Print votes, applicability and reference provenance as JSON')
    batch=sub.add_parser('batch',help='Classify multiple instance files in input order')
    batch.add_argument('inputs',nargs='+');batch.add_argument('--workers',type=int,default=1)
    args=parser.parse_args(argv)
    try:
        from .api import CEJSP,classify,classify_many
        if args.command=='classify':
            print(json.dumps(classify(args.input),indent=2,allow_nan=False) if args.details else CEJSP(args.input))
        else:
            print(json.dumps(classify_many(args.inputs,workers=args.workers),indent=2,allow_nan=False))
    except (ValueError,OSError,ImportError,RuntimeError,KeyError) as exc:
        parser.exit(2,f'cejsp: {exc}\n')

__all__=['main']
