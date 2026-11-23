"""Preflight, shared extraction, parity, then sequential B/C/D experiments."""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

BASELINE_SHA256 = '4f8ef1adf1ec549dbddf0efc43eca15b3e6865de1331c97b63bee931a5d763ee'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def main(args):
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'pipeline.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('This experiment pipeline is already running')
    environment = dict(os.environ, PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    for variable, directory in [('PYTORCH_KERNEL_CACHE_PATH', 'kernel_cache'), ('TMPDIR', 'tmp')]:
        target = output / directory; target.mkdir(exist_ok=True)
        environment[variable] = str(target)
    environment['PYTHONPATH'] = os.pathsep.join([
        str(Path(__file__).resolve().parent.parent), str(Path(args.base_code).resolve()),
        str(Path(args.root) / 'working_image_captioning'), environment.get('PYTHONPATH', '')])
    def stage(name, **values):
        record = {'stage': name, 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), **values}
        temporary = output / 'pipeline_status.json.tmp'
        temporary.write_text(json.dumps(record, indent=2)); temporary.replace(output / 'pipeline_status.json')
        print('RELATION_EVENT_STAGE', json.dumps(record), flush=True)
    def command(arguments, name, env=None):
        with (output / (name + '.log')).open('a') as log:
            subprocess.run(arguments, env={**environment, **(env or {})},
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    common = ['--root', args.root, '--split-root', args.split_root,
              '--cache', args.cache, '--spatial-cache', args.spatial_cache]
    events = str(output / 'event_features')
    try:
        required = [args.baseline, Path(args.cache) / 'cache_metadata.json',
                    Path(args.spatial_cache) / 'cache_metadata.json',
                    Path(args.base_code) / 'video_captioning/hierarchical.py',
                    Path(args.base_code) / 'evaluate_video.py',
                    Path(args.root) / 'stage2_final.pt',
                    Path(args.split_root) / 'train.jsonl', Path(args.split_root) / 'val.jsonl']
        missing = [str(p) for p in required if not Path(p).is_file()]
        if missing:
            stage('waiting_for_baseline_transfer', missing=missing)
            return 3
        stage('checking_7396_checkpoint_and_environment')
        if digest(args.baseline) != BASELINE_SHA256:
            raise ValueError('Not the verified CIDEr-73.96 checkpoint; refusing substitution')
        # Import heavy modules only after obtaining a complete baseline payload.
        import torch
        world = torch.cuda.device_count()
        if world not in (1, 2):
            raise RuntimeError('Expected one or two visible GPUs')
        stage('gpu_configuration', visible_gpus=world, effective_batch=48)
        checkpoint = torch.load(args.baseline, map_location='cpu', weights_only=True)
        if digest(Path(args.root) / 'stage2_final.pt') != checkpoint['image_checkpoint_sha256']:
            raise ValueError('Image-language initialization differs from 73.96 base')
        command([sys.executable, '-m', 'relation_events.test_components'], 'components')
        stage('extracting_shared_roi_appearance_and_relation_events')
        jobs = []
        extraction_workers = world * 4
        for shard in range(extraction_workers):
            log = (output / f'extract_{shard}.log').open('a')
            process = subprocess.Popen([sys.executable, '-m', 'relation_events.extract', *common,
                                       '--output', events, '--num-shards', str(extraction_workers), '--shard-index', str(shard),
                                       '--device', 'cuda:0'],
                                      env={**environment, 'CUDA_VISIBLE_DEVICES': str(shard % world),
                                           'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'},
                                      stdout=log, stderr=subprocess.STDOUT)
            jobs.append((process, log))
        for process, log in jobs:
            code = process.wait(); log.close()
            if code:
                for pending, pending_log in jobs:
                    if pending.poll() is None:
                        pending.terminate()
                        pending.wait()
                    pending_log.close()
                raise subprocess.CalledProcessError(code, process.args)
        shared = [*common, '--event-cache', events, '--baseline', args.baseline]
        distributed = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                       f'--nproc_per_node={world}', '-m', 'relation_events.train']
        stage('training_and_initialization_checks', visible_gpus=world)
        command([sys.executable, '-m', 'relation_events.verify', *common,
                 '--event-cache', events, '--baseline', args.baseline,
                 '--output', str(output / 'initialization_checks.json')], 'parity')
        # Fixed effective batch = 48 while choosing a measured safe microbatch.
        batch = None
        for candidate in (24, 12, 8, 4, 2):
            accumulation = 48 // (world * candidate)
            try:
                smoke_name = f'order_stress_b{candidate}'
                command([sys.executable, '-m', 'relation_events.order_smoke', *shared,
                         '--batch-size', str(candidate), '--output', str(output/(smoke_name+'.json'))],
                        smoke_name)
                for smoke_variant in ('B', 'C', 'D'):
                    smoke_name = f'smoke_{smoke_variant}_b{candidate}'
                    command([*distributed, *shared, '--variant', smoke_variant,
                             '--output', str(output / smoke_name),
                             '--batch-size', str(candidate), '--accumulation', str(accumulation),
                             '--epochs', '1', '--max-steps', '2', '--train-limit', '32', '--workers', '0'],
                            smoke_name)
                batch = candidate; break
            except subprocess.CalledProcessError:
                log_text = (output / (smoke_name + '.log')).read_text()
                if 'out of memory' not in log_text.lower():
                    raise
        if batch is None:
            raise RuntimeError('No safe shared microbatch found')
        accumulation = 48 // (world * batch)
        stage('shared_batch_verified', visible_gpus=world, per_gpu_batch=batch,
              accumulation=accumulation, effective_batch=48)
        eval_base = [sys.executable, '-m', 'relation_events.evaluate', *common, '--event-cache', events]
        stage('reproducing_7396_full_validation')
        command([*eval_base, '--variant', 'B', '--checkpoints', args.baseline,
                 '--output', str(output / 'A_validation')], 'baseline_validation')
        reports = list((output / 'A_validation').glob('*_metrics.json'))
        initial = json.loads(reports[0].read_text())
        actual = initial['scores_times_100']['CIDEr']
        if abs(actual - 73.95684838876862) > 0.15:
            raise RuntimeError(f'Baseline reproduction changed to {actual}; investigate before training')
        # No test data are used; all new branches independently start from A.
        for variant in ['B', 'C', 'D']:
            training = output / variant / 'training'
            validation = output / variant / 'validation'
            for epoch in range(1, args.epochs + 1):
                metric = validation / f'epoch_{epoch:03d}_metrics.json'
                if metric.exists():
                    continue
                stage('training', variant=variant, epoch=epoch, epochs=args.epochs,
                      per_gpu_batch=batch, accumulation=accumulation, all_references=20)
                if not (training / f'epoch_{epoch:03d}.pt').exists():
                    command([*distributed, *shared, '--variant', variant, '--output', str(training),
                             '--epochs', str(args.epochs), '--stop-after-epoch', str(epoch),
                             '--batch-size', str(batch), '--accumulation', str(accumulation), '--resume'],
                            f'{variant}_training')
                stage('full_validation', variant=variant, epoch=epoch)
                command([*eval_base, '--variant', variant, '--checkpoints',
                         str(training / f'epoch_{epoch:03d}.pt'), '--output', str(validation)],
                        f'{variant}_validation')
                # Save a reviewable comparison after each checkpoint.
                available = reports + list(output.glob('*/validation/*_metrics.json'))
                records = [json.loads(p.read_text()) for p in available]
                best = max(records, key=lambda item: item['scores_times_100']['CIDEr'])
                result = {'baseline_validation_cider': actual, 'best': best,
                          'results': records, 'test_set_evaluated': False}
                (output / 'comparison.json').write_text(json.dumps(result, indent=2))
        stage('complete', comparison=str(output / 'comparison.json'))
        return 0
    except Exception as error:
        stage('failed', error=repr(error)); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/root/autodl-tmp/DRAMCap_video')
    parser.add_argument('--split-root', required=True)
    parser.add_argument('--base-code', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--spatial-cache', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    sys.exit(main(parser.parse_args()))
