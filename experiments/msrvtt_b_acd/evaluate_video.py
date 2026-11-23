"""Evaluate a complete captioning split with COCO PTB tokenization and metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from video_captioning.data import load_split


def main(args):
    if not shutil.which('java'):
        raise RuntimeError('Java runtime is required for standard COCO PTB tokenization and METEOR')
    rows=load_split(args.references,args.split)
    predictions=[json.loads(line) for line in Path(args.predictions).read_text().splitlines() if line.strip()]
    identities=[row['video_id'] for row in predictions]
    if len(identities)!=len(set(identities)) or any(x['split']!=args.split or not x['caption'].strip() for x in predictions):
        raise ValueError('Duplicate/mislabelled/empty predictions')
    expected={row['video_id'] for row in rows}
    if set(identities)!=expected:
        raise ValueError(f'Predictions must cover the complete selected split: {len(identities)} predictions vs {len(expected)} references')
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.meteor.meteor import Meteor
    tokenizer=PTBTokenizer()
    references=tokenizer.tokenize({r['video_id']:[{'caption':x} for x in r['references']] for r in rows})
    results=tokenizer.tokenize({r['video_id']:[{'caption':r['caption']}] for r in predictions})
    raw={}
    bleu,_=Bleu(4).compute_score(references,results)
    raw.update({f'BLEU_{i+1}':float(x) for i,x in enumerate(bleu)})
    for name,scorer in [('CIDEr',Cider()),('ROUGE_L',Rouge()),('METEOR',Meteor())]:
        try:
            score,_=scorer.compute_score(references,results);raw[name]=float(score)
        finally:
            if name=='METEOR':scorer.meteor_p.kill()
    report={'split':args.split,'videos':len(rows),'references_per_video':sorted({len(x['references']) for x in rows}),
            'raw_scores':raw,'scores_times_100':{k:v*100 for k,v in raw.items()},
            'tokenization':'COCO PTBTokenizer','prediction_sha256':hashlib.sha256(Path(args.predictions).read_bytes()).hexdigest(),
            'references_sha256':hashlib.sha256(Path(args.references).read_bytes()).hexdigest()}
    Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--references',required=True)
    parser.add_argument('--predictions',required=True)
    parser.add_argument('--split',choices=['val','test'],required=True)
    parser.add_argument('--output',required=True)
    main(parser.parse_args())
