"""Add A/C/D predictions to the saved B candidate pool in isolation."""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from cider_reward import CiderReward, counts
from evaluate_video import main as evaluate_predictions
from refine_msrvtt_b import DECODE, load_checkpoint
from relation_events.data import EventDataset, collate_events
from relation_events.model import EventCaptioner
from relation_events.train import atomic_json, move


EXPECTED_B_PLUS = 'f130b286dc2812d90d10134b7176df4936358bcad236149c3025806e04f512dd'
EXPECTED_B_PLUS_ACD = '97d2df40a1f5c1b908aa219e37e623668bef861a42d4e148195b2ed5755b5111'
GROUPS = {'B_plus': (), 'B_plus_A': ('A',), 'B_plus_C': ('C',),
          'B_plus_D': ('D',), 'B_plus_ACD': ('A', 'C', 'D')}


def load_captions(path, ids, reference_digest):
    metadata = json.loads(path.with_name(path.name.replace('_predictions.jsonl', '_metrics.json')).read_text())
    if metadata['prediction_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError(f'Prediction checksum mismatch for {path}')
    if metadata['references_sha256'] != reference_digest:
        raise ValueError(f'Inconsistent validation references for {path}')
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    captions = {row['video_id']: row['caption'] for row in rows}
    if len(rows) != len(ids) or set(captions) != set(ids):
        raise ValueError(f'Expected one candidate for each validation video in {path}')
    if any(row['split'] != 'val' or not row['caption'].strip() for row in rows):
        raise ValueError(f'Invalid validation candidate in {path}')
    return captions


def pick(candidates, likelihood, retrieval, admitted, weight=0.32):
    def gain(index):
        return likelihood[index] - likelihood[0] + weight * (retrieval[index] - retrieval[0])

    eligible = [index for index in admitted if index > 0
                and retrieval[index] - retrieval[0] > .05
                and likelihood[index] - likelihood[0] >= -.20 and gain(index) > .025]
    return candidates[max(eligible, key=gain, default=0)]


def visual_key(path):
    sample = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    frames = sample['visual'][sample['frame_mask'].bool()].float().mean(1)
    return F.normalize(F.normalize(frames, dim=-1).mean(0), dim=0)


def caption_likelihood(model, batch, candidates):
    prefix, prefix_mask, prefix_length = model._prefix(batch)
    flat = [text for group in candidates for text in group]
    source = torch.tensor([i for i, group in enumerate(candidates) for _ in group], device=prefix.device)
    ids, valid = model._caption_tensors(flat, prefix.device)
    mask = torch.cat((prefix_mask[source], valid), dim=1)
    positions = mask.long().cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 1)
    embeddings = torch.cat((prefix[source], model.decoder._get_embed_fn()(ids)), dim=1)
    logits = model.decoder.llm(inputs_embeds=embeddings, attention_mask=mask,
        position_ids=positions, use_cache=False, return_dict=True).logits
    values = logits[:, prefix_length-1:-1].float().log_softmax(-1)
    values = values.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    mean = (values * valid).sum(-1) / valid.sum(-1)
    return [x.tolist() for x in mean.split([len(group) for group in candidates])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--original-abcd', type=Path, required=True)
    parser.add_argument('--split-root', type=Path)
    parser.add_argument('--feature-cache', type=Path)
    parser.add_argument('--spatial-cache', type=Path)
    parser.add_argument('--event-cache', type=Path)
    parser.add_argument('--image-checkpoint', type=Path)
    parser.add_argument('--decoder', type=Path)
    parser.add_argument('--verify-archive', action='store_true',
                        help='Check both the B-only and B+A/C/D archived prediction hashes.')
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.monotonic()
    root, run, output = args.root, args.run, args.output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob('*_metrics.json')):
        raise RuntimeError('Choose a fresh output directory for the candidate comparison')
    splits = args.split_root or root / 'data/msrvtt/caption_splits'
    cache = args.feature_cache or root / 'runs/msrvtt_eva_recovered_20261115/features'
    spatial = args.spatial_cache or root / 'runs/msrvtt_hierarq_spatial_20261115/spatial_features'
    events = args.event_cache or run / 'event_features'
    val = EventDataset(splits, cache, spatial, events, 'val')
    train_rows = [json.loads(line) for line in (splits/'train.jsonl').read_text().splitlines() if line.strip()]
    index_ids = [row['video_id'] for row in train_rows]
    query_ids = [row['video_id'] for row in val.rows]
    if set(query_ids) & set(index_ids) or len(index_ids) != len(set(index_ids)):
        raise ValueError('Retrieval requires distinct training and validation video IDs')
    train_keys = torch.stack([visual_key(cache/'train'/f'{id_}.pt') for id_ in index_ids])
    val_keys = torch.stack([visual_key(cache/'val'/f'{id_}.pt') for id_ in query_ids])
    similarities, indices = (val_keys @ train_keys.T).topk(8, dim=1)
    neighbors = dict(zip(query_ids, zip(indices.tolist(), (similarities/.05).softmax(1).tolist())))
    reward = CiderReward([row['references'] for row in train_rows])

    @lru_cache(maxsize=8000)
    def encoded_ref(index):
        return [counts(text) for text in train_rows[index]['references']]

    def retrieval_score(video_id, caption):
        encoded = counts(caption)
        matched, weights = neighbors[video_id]
        score = 0.0
        for index, weight in zip(matched, weights, strict=True):
            references = encoded_ref(index)
            score += weight * sum(reward.similarity(encoded, ref) for ref in references) / len(references)
        return score * 10.0

    reference_digest = hashlib.sha256((splits/'val.jsonl').read_bytes()).hexdigest()
    anchors = load_captions(run/'B/validation/epoch_007_predictions.jsonl', query_ids, reference_digest)
    candidate_files = {
        'A': args.original_abcd/'A_validation/epoch_008_predictions.jsonl',
        'C': args.original_abcd/'C/validation/epoch_007_predictions.jsonl',
        'D': args.original_abcd/'D/validation/epoch_002_predictions.jsonl',
    }
    predictions_by_variant = {name: load_captions(path, query_ids, reference_digest)
                              for name, path in candidate_files.items()}
    source = run / 'B/training/epoch_007.pt'
    decoder = args.decoder or root / 'models/vicuna-7b-v1.5'
    image_checkpoint = args.image_checkpoint or root / 'stage2_final.pt'
    model = EventCaptioner(str(decoder), image_checkpoint, 'B').cuda().eval()
    model.load_adapted_state(load_checkpoint(source, val))
    loader = DataLoader(val, batch_size=4, num_workers=0, collate_fn=collate_events)
    decode = dict(DECODE, num_beams=5, num_return_sequences=5)
    outputs = {name: [] for name in GROUPS}
    changed = dict.fromkeys(GROUPS, 0)
    video_count = 0
    for batch in loader:
        ids = batch['video_ids']
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            gpu_batch = move(batch, 'cuda:0')
            beams = model.generate(gpu_batch, **decode)
            if len(beams) != 5 * len(ids):
                raise RuntimeError('Expected five candidates per validation video')
            captions = [list(dict.fromkeys([anchors[i], *(s.strip() for s in beams[5*j:5*j+5] if s.strip())]))
                        for j, i in enumerate(ids)]
            likelihoods = caption_likelihood(model, gpu_batch, captions)
            # Keep the original candidates in their original tensor batch so
            # BF16 batch shapes cannot affect the archived control scores.
            extras = [list(dict.fromkeys(text for text in (
                predictions_by_variant[v][i] for v in ('A', 'C', 'D')) if text not in group))
                for i, group in zip(ids, captions, strict=True)]
            extra_scores = (caption_likelihood(model, gpu_batch, extras) if any(extras)
                            else [[] for _ in ids])
        for video_id, original, original_scores, extra, added_scores in zip(
                ids, captions, likelihoods, extras, extra_scores, strict=True):
            candidates = original + extra
            likelihood = original_scores + added_scores
            retrieval = [retrieval_score(video_id, text) for text in candidates]
            control = pick(candidates, likelihood, retrieval, range(len(original)))
            for name, variants in GROUPS.items():
                sources = {predictions_by_variant[v][video_id] for v in variants}
                admitted = [i for i, text in enumerate(candidates) if i < len(original) or text in sources]
                choice = pick(candidates, likelihood, retrieval, admitted)
                outputs[name].append({'video_id':video_id, 'split':'val', 'caption':choice})
                changed[name] += int(choice != control)
        video_count += len(ids)
        if video_count % 100 == 0 or video_count == len(val):
            atomic_json(output/'progress.json', {'stage':'generating_and_reranking','videos':video_count,'total':len(val)})
    reports = {}
    expected_digests = {'B_plus': EXPECTED_B_PLUS, 'B_plus_ACD': EXPECTED_B_PLUS_ACD}
    for name, rows in outputs.items():
        predictions = output/f'{name}_predictions.jsonl'
        predictions.write_text(''.join(json.dumps(row)+'\n' for row in rows),
                               encoding='utf-8', newline='\n')
        computed_digest = hashlib.sha256(predictions.read_bytes()).hexdigest()
        if args.verify_archive and name in expected_digests and computed_digest != expected_digests[name]:
            raise RuntimeError(f'{name} did not reproduce the archived predictions')
        metrics = output/f'{name}_metrics.json'
        evaluate_predictions(argparse.Namespace(references=str(splits/'val.jsonl'),
            predictions=str(predictions), split='val', output=str(metrics)))
        report = json.loads(metrics.read_text())
        report.update(trial=name, checkpoint=str(source), changed_vs_B_plus=changed[name],
                      extra_sources={v:str(candidate_files[v]) for v in GROUPS[name]},
                      retrieval_weight=.32, retrieval_source='train_only', neighbors=8)
        atomic_json(metrics,report)
        reports[name] = report
        print('B_ACD_RESULT', name, report['scores_times_100']['CIDEr'], changed[name], flush=True)
    atomic_json(output/'comparison.json', {'results':reports,'seconds':round(time.monotonic()-started,2)})
    atomic_json(output/'progress.json', {'stage':'complete'})


if __name__ == '__main__':
    main()
