"""Explicit English-only component assessment; never invokes research/security tasks."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from capability_assessments import assess_english_foundation
from io_utils import atomic_write_json


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv);path=args.output
    bundle={'schema':'ina.foundation_assessment_bundle/V1','assessments':{},'promotion_authorized':False}
    if path.exists():
        if path.stat().st_size>2*1024*1024:raise ValueError('assessment bundle byte budget')
        bundle=json.loads(path.read_text())
        if not isinstance(bundle,dict) or not isinstance(bundle.get('assessments'),dict):raise ValueError('existing assessment bundle is malformed')
    bundle['assessments']['language_english']=assess_english_foundation()
    bundle['promotion_authorized']=False
    atomic_write_json(path,bundle,indent=2,ensure_ascii=False)
    print(path)
    return 0


if __name__=='__main__':raise SystemExit(main())
