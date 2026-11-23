"""Rerank B's captions against visually similar training videos only."""
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
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.monotonic()
    root, run, output = args.root, args.run, args.output
    output.mkdir(parents=True, exist_ok=True)
    splits = root / 'data/msrvtt/caption_splits'
    cache = root / 'runs/msrvtt_eva_recovered_20261115/features'
    spatial = root / 'runs/msrvtt_hierarq_spatial_20261115/spatial_features'
    val = EventDataset(splits, cache, spatial, run / 'event_features', 'val')
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

    anchor_path = run / 'B/validation/epoch_007_predictions.jsonl'
    anchor_metrics = json.loads((run/'B/validation/epoch_007_metrics.json').read_text())
    if anchor_metrics['prediction_sha256'] != hashlib.sha256(anchor_path.read_bytes()).hexdigest():
        raise ValueError('Original B predictions do not match their saved metrics')
    anchors = {row['video_id']:row['caption'] for row in map(json.loads, anchor_path.read_text().splitlines())}
    if set(anchors) != set(query_ids):
        raise ValueError('B anchor predictions must cover the validation split')
    source = run / 'B/training/epoch_007.pt'
    model = EventCaptioner(str(root/'models/vicuna-7b-v1.5'),root/'stage2_final.pt','B').cuda().eval()
    model.load_adapted_state(load_checkpoint(source, val))
    loader = DataLoader(val, batch_size=4, num_workers=0, collate_fn=collate_events)
    decode = dict(DECODE, num_beams=5, num_return_sequences=5)
    options = {'retrieval_008': .08, 'retrieval_016': .16, 'retrieval_032': .32}
    outputs = {name: [] for name in options}
    changed = dict.fromkeys(options, 0)
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
        for video_id, candidates, likelihood in zip(ids, captions, likelihoods, strict=True):
            retrieval = [retrieval_score(video_id, text) for text in candidates]
            for name, weight in options.items():
                def gain(j):
                    return likelihood[j]-likelihood[0] + weight*(retrieval[j]-retrieval[0])
                eligible = [j for j in range(1, len(candidates)) if retrieval[j]-retrieval[0] > .05
                            and likelihood[j]-likelihood[0] >= -.20 and gain(j) > .025]
                choice = max(eligible, key=gain, default=0)
                outputs[name].append({'video_id':video_id, 'split':'val', 'caption':candidates[choice]})
                changed[name] += int(choice != 0)
        video_count += len(ids)
        if video_count % 100 == 0 or video_count == len(val):
            atomic_json(output/'progress.json', {'stage':'generating_and_reranking','videos':video_count,'total':len(val)})
    reports = {}
    for name, rows in outputs.items():
        predictions = output/f'{name}_predictions.jsonl'
        predictions.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        metrics = output/f'{name}_metrics.json'
        evaluate_predictions(argparse.Namespace(references=str(splits/'val.jsonl'),
            predictions=str(predictions), split='val', output=str(metrics)))
        report = json.loads(metrics.read_text())
        report.update(trial=name, checkpoint=str(source), changed_from_original=changed[name],
                      retrieval_weight=options[name], retrieval_source='train_only', neighbors=8)
        atomic_json(metrics,report)
        reports[name] = report
        print('B_RERANK_RESULT', name, report['scores_times_100']['CIDEr'], changed[name], flush=True)
    atomic_json(output/'comparison.json', {'results':reports,'seconds':round(time.monotonic()-started,2)})
    atomic_json(output/'progress.json', {'stage':'complete'})


if __name__ == '__main__':
    main()
