#!/usr/bin/env python3
"""Run the actual EvalScope review pipeline over immutable cached predictions.

Keep imports inside main so multiprocessing-spawn children do not initialize
the complete evaluation framework before the four-second checker can start.
Inference is explicitly forbidden, and original outputs are never overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    assert 1 <= args.workers <= 4
    assert not args.output.exists(), 'Use a new directory; preserve all old results'
    predictions = list((args.source / 'predictions').rglob('*.jsonl'))
    assert len(predictions) == 1
    samples = [json.loads(line) for line in predictions[0].read_text().splitlines() if line]
    assert len(samples) == 164
    args.output.mkdir(parents=True)
    shutil.copytree(args.source / 'predictions', args.output / 'predictions')

    import yaml
    from evalscope.evaluator.evaluator import Evaluator
    from evalscope.run import run_task

    config = yaml.safe_load((args.source / 'configs/task_config.yaml').read_text())
    assert config['eval_batch_size'] == 64
    assert config['dataset_args']['humaneval']['review_timeout'] == 4
    assert config['generation_config']['batch_size'] == 64
    config.update(use_cache=str(args.output), work_dir=str(args.output),
                  rerun_review=True, no_timestamp=True,
                  eval_batch_size=args.workers, judge_worker_num=None)
    inference_attempts = []

    def refuse_prediction(self, sample, model_prediction_dir):
        inference_attempts.append(str(getattr(sample, 'id', 'unknown')))
        raise RuntimeError('Offline review requires a complete prediction cache; inference forbidden')

    Evaluator._predict_sample = refuse_prediction
    source_hash = hashlib.sha256(predictions[0].read_bytes()).hexdigest()
    manifest = {
        'source': str(args.source), 'output': str(args.output),
        'source_prediction_file': str(predictions[0]),
        'source_prediction_sha256': source_hash,
        'generated_request_concurrency': 64, 'review_workers': args.workers,
        'review_timeout_seconds': 4, 'GPU_used': False,
        'original_results_overwritten': False,
        'changes': ['Reuse all 164 cached predictions', 'Rerun review only',
                    'Lower review concurrency', 'Remove deprecated judge_worker_num',
                    'Defer framework imports in spawn children', 'Forbid prediction calls'],
    }
    (args.output / 'rejudge_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    start = time.monotonic()
    run_task(config)
    assert not inference_attempts, inference_attempts
    cached_copy = list((args.output / 'predictions').rglob('*.jsonl'))
    assert len(cached_copy) == 1
    assert hashlib.sha256(cached_copy[0].read_bytes()).hexdigest() == source_hash
    reviews = list((args.output / 'reviews').rglob('*.jsonl'))
    assert len(reviews) == 1
    records = [json.loads(line) for line in reviews[0].read_text().splitlines() if line]
    assert len(records) == 164 and len({item['index'] for item in records}) == 164
    first32 = [item for item in records if item['index'] < 32]
    executions = [item['sample_score']['score']['metadata']['execution_result'] for item in records]
    passed = sum(item['passed'] for item in executions)
    first32_passed = sum(item['sample_score']['score']['metadata']['execution_result']['passed']
                         for item in first32)
    timeouts = sum(item['result'] == 'timed out' for item in executions)
    summary = {
        **manifest, 'inference_attempts': inference_attempts,
        'prediction_cache_unchanged': True, 'total': len(records), 'passed': passed,
        'pass_at_1': passed / len(records), 'first32_passed': first32_passed,
        'timed_out': timeouts, 'elapsed_seconds': time.monotonic() - start,
        'failed_tasks': [item['task_id'] for item in executions if not item['passed']],
        'accuracy_gate_passed': passed >= 156 and first32_passed == 32 and timeouts == 0,
    }
    (args.output / 'rejudge_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ('total', 'passed', 'first32_passed',
                     'timed_out', 'inference_attempts', 'accuracy_gate_passed')}), flush=True)


if __name__ == '__main__':
    main()
