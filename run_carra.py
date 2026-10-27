import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from datasets import SkeletonDataset, load_records, prepare_data
from model_defs import get_model
from utils.classifier import fit_classifier
from utils.coverage import generate_coverage
from utils.memory import initialize_memory, update_memory
from utils.metrics import classification_metrics, summarize_run, summarize_runs
from utils.model_inversion import generate_replay
from utils.planner import build_replay_plan
from utils.training import extract_features, set_seed, train_backbone

ROOT = Path(__file__).resolve().parent
DATASETS = ('hgr_shrec_2017', 'ego_gesture', 'ntu60', 'ntu120', 'nw_ucla')

def load_config(dataset, override=None):
    with (ROOT / 'configs' / 'default.yaml').open() as stream:
        cfg = yaml.safe_load(stream)
    with (ROOT / 'configs' / f'{dataset}.yaml').open() as stream:
        cfg.update(yaml.safe_load(stream))
    if override:
        with Path(override).open() as stream:
            changes = yaml.safe_load(stream)
        if not isinstance(changes, dict):
            raise ValueError('Configuration override must be a YAML mapping.')
        for key, value in changes.items():
            if key not in cfg:
                raise ValueError(f'Unknown configuration field: {key}')
            if isinstance(value, dict) and isinstance(cfg[key], dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    if cfg['dataset'] != dataset:
        raise ValueError('A configuration override cannot change the selected dataset.')
    return cfg

def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value

def write_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(_json_value(value), indent=2, allow_nan=False) + '\n')

def base_development_split(records, base_classes, dataset):
    
    base_classes = set(base_classes)
    train = [record for record in records if record['split'] == 'train'
             and record['label'] in base_classes]
    if dataset == 'ego_gesture':
        validation = [record for record in records if record['split'] == 'val'
                      and record['label'] in base_classes]
        return train, validation
    train_part, validation = [], []
    for class_id in sorted(base_classes):
        samples = sorted((record for record in train if record['label'] == class_id),
                         key=lambda record: record['id'])
        count = len(samples)
        if count < 2:
            raise ValueError(f'Base class {class_id} needs at least two training samples.')
        order = np.random.RandomState(0).permutation(count)
        n_val = max(1, min(count - 1, int(round(count * 0.1))))
        validation.extend(samples[index] for index in order[:n_val])
        train_part.extend(samples[index] for index in order[n_val:])
    return train_part, validation

def _dataset(records, cfg, train=False):
    return SkeletonDataset(records, cfg['seq_len'],
                           views=int(cfg['train']['views']) if train else 1,
                           augment=train, dataset=cfg['dataset'])

def _features(model, records, cfg, device):
    return extract_features(model, _dataset(records, cfg), device,
                            batch_size=cfg['train']['batch_size'], workers=cfg['train']['workers'])

def evaluate_task(model, classifier, evaluation_records, new_classes, history, cfg, device):
    
    features, labels = _features(model, evaluation_records, cfg, device)
    predictions = classifier.predict(features)
    return classification_metrics(labels, predictions, new_classes, history)

def save_latest_state(path, model, classifier, memory, cfg, task_id, order_seed, init_seed):
    
    state = {'version': 2, 'task_id': task_id, 'class_order_seed': order_seed,
             'init_seed': init_seed, 'config': cfg, 'memory': memory,
             'classifier': classifier,
             'backbone': {key: value.detach().cpu() for key, value in model.state_dict().items()}}
    torch.save(state, path)

def run_experiment(cfg, records, protocol, order_seed, init_seed, device, output_dir):
    
    if order_seed < 0 or init_seed < 0:
        raise ValueError('Seeds must be nonnegative.')
    if protocol not in cfg['protocols']:
        raise ValueError(f'Unknown protocol {protocol!r}; choose {tuple(cfg["protocols"])}')
    base_count, n_tasks, increment = map(int, cfg['protocols'][protocol])
    if base_count < 2 or increment < 1 or n_tasks < 1 or base_count + n_tasks * increment != cfg['n_classes']:
        raise ValueError('Task protocol must cover all classes, with at least two base classes.')
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing run: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(init_seed)
    class_order = np.random.RandomState(order_seed).permutation(cfg['n_classes']).tolist()
    expected = set(class_order)
    for split in ('train', 'test'):
        available = {record['label'] for record in records if record['split'] == split}
        if available != expected:
            raise ValueError(f'{split} classes differ from the dataset configuration: {sorted(available ^ expected)}')
    write_json(output_dir / 'config.json', {
        **cfg, 'protocol': protocol, 'class_order_seed': order_seed,
        'init_seed': init_seed, 'class_order': class_order,
        'memory_regime': 'No external SV bank',
        'replay_execution': 'bounded_quota_completion',
        'kappa_reference_unit': 'base class over final development features',
    })
    model = get_model(cfg).to(device)
    history, plans = [], []
    memory = classifier = None
    seen_classes = []
    for task_id in range(n_tasks + 1):
        task_started = time.perf_counter()
        if task_id == 0:
            new_classes = class_order[:base_count]
            base_train, base_val = base_development_split(records, new_classes, cfg['dataset'])
            
            new_records = base_train + base_val
            write_json(output_dir / 'base_split.json', {
                'base_train': [record['id'] for record in base_train],
                'base_internal_val': [record['id'] for record in base_val],
                'split_seed': 0, 'hyperparameters': 'fixed manuscript S1; no search',
            })
            del base_train, base_val
        else:
            start = base_count + (task_id - 1) * increment
            new_classes = class_order[start:start + increment]
            new_records = [record for record in records if record['split'] == 'train'
                           and record['label'] in new_classes]
        if {record['label'] for record in new_records} != set(new_classes):
            raise ValueError(f'Task {task_id} has a new class without training samples.')
        if task_id and set(new_classes) & set(memory['class_ids']):
            raise ValueError('Incremental learning inputs contain old training classes.')
        print(f'Task {task_id}/{n_tasks}, new classes={new_classes}, new samples={len(new_records)}', flush=True)
        stage_times = {}
        replay_records = []
        plan = None
        if task_id:
            frozen_encoder = copy.deepcopy(model).eval()
            for parameter in frozen_encoder.parameters():
                parameter.requires_grad_(False)
            started = time.perf_counter()
            before_features, before_labels = _features(frozen_encoder, new_records, cfg, device)
            plan = build_replay_plan(memory, classifier, before_features, before_labels, cfg['planner'])
            stage_times['planning'] = time.perf_counter() - started
            del before_features, before_labels
            replay_seed = int(np.random.SeedSequence([init_seed, order_seed, task_id]).generate_state(1)[0])
            replay_records = generate_replay(frozen_encoder, classifier, memory, plan,
                                             cfg['inversion'], replay_seed)
            stage_times['inversion'] = plan['inversion_seconds']
            
            del frozen_encoder
        started = time.perf_counter()
        model = train_backbone(model, _dataset(new_records + replay_records, cfg, train=True),
                               cfg['train'], device, stage='base' if task_id == 0 else 'incremental')
        stage_times['backbone'] = time.perf_counter() - started
        started = time.perf_counter()
        new_features, new_labels = _features(model, new_records, cfg, device)
        replay_features, replay_labels = _features(model, replay_records, cfg, device)
        stage_times['reencoding'] = time.perf_counter() - started
        started = time.perf_counter()
        if task_id:
            coverage_features, coverage_labels = generate_coverage(memory, plan, cfg['n_pseudo'])
            fit_features = np.concatenate((new_features, replay_features, coverage_features))
            fit_labels = np.concatenate((new_labels, replay_labels, coverage_labels))
            plan['pseudo_coverage_count'] = int(coverage_labels.size)
            del coverage_features, coverage_labels
        else:
            fit_features, fit_labels = new_features, new_labels
        classifier = fit_classifier(fit_features, fit_labels, **cfg['svm'])
        
        del fit_features, fit_labels
        stage_times['classifier_refit'] = time.perf_counter() - started
        started = time.perf_counter()
        if task_id:
            memory = update_memory(memory, new_features, new_labels, replay_features,
                                   replay_labels, plan, classifier, cfg['boundary_threshold'])
        else:
            memory = initialize_memory(new_features, new_labels, classifier,
                                       seed=init_seed, boundary_threshold=cfg['boundary_threshold'])
        stage_times['memory'] = time.perf_counter() - started
        del new_features, new_labels, replay_features, replay_labels, new_records, replay_records
        seen_classes.extend(new_classes)
        
        evaluation_records = [record for record in records if record['split'] == 'test'
                              and record['label'] in seen_classes]
        started = time.perf_counter()
        metrics = evaluate_task(model, classifier, evaluation_records, new_classes, history, cfg, device)
        stage_times['evaluation'] = time.perf_counter() - started
        del evaluation_records
        metrics.update({'task_id': task_id, 'stage_seconds': stage_times,
                        'task_seconds': time.perf_counter() - task_started})
        history.append(metrics)
        if task_id:
            
            report = {key: value for key, value in plan.items()
                      if key not in ('axes', 'matched_centers')}
            plans.append(report)
            write_json(output_dir / f'plan_task_{task_id}.json', report)
            del plan
        write_json(output_dir / f'metrics_task_{task_id}.json', metrics)
        save_latest_state(output_dir / 'state.pt', model, classifier, memory, cfg,
                          task_id, order_seed, init_seed)
        print(f'  G={metrics["G"]:.3f}, IFM={metrics["IFM"]}, AF={metrics["AF"]}', flush=True)
    summary = summarize_run(history, plans, cfg['planner']['total_budget'])
    summary.update({'class_order_seed': order_seed, 'init_seed': init_seed, 'protocol': protocol,
                    'stage_seconds': {key: sum(task['stage_seconds'].get(key, 0) for task in history)
                                      for key in history[-1]['stage_seconds']}})
    write_json(output_dir / 'summary.json', summary)
    return summary

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--data-root', type=Path, help='Dataset directory; default: datasets/<dataset>')
    parser.add_argument('--prepare', action='store_true', help='Prepare raw skeletons once, then exit')
    parser.add_argument('--protocol', default='main')
    parser.add_argument('--class-order-seed', type=int, default=0)
    parser.add_argument('--init-seed', type=int, default=0)
    parser.add_argument('--all-seeds', action='store_true', help='Explicitly run 10 class orders x 3 initializations')
    parser.add_argument('--gpu', default='0', help='One CUDA device index, or cpu')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--config', type=Path, help='Optional explicit YAML overrides')
    args = parser.parse_args()
    root = (args.data_root or ROOT / 'datasets' / args.dataset).resolve()
    if args.prepare:
        records = prepare_data(args.dataset, root)
        print(f'Prepared {len(records)} samples under {root / "prepared"}')
        return
    cfg = load_config(args.dataset, args.config)
    device = torch.device('cpu' if args.gpu == 'cpu' else f'cuda:{int(args.gpu)}')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Restore the driver for training, or explicitly select --gpu cpu.')
        torch.cuda.set_device(device)
    records = load_records(args.dataset, root)
    seed_pairs = ([(order, init) for order in range(10) for init in range(3)] if args.all_seeds
                  else [(args.class_order_seed, args.init_seed)])
    summaries = []
    experiment_root = args.output_root / args.dataset / args.protocol
    for order_seed, init_seed in seed_pairs:
        output_dir = experiment_root / f'order_{order_seed}_init_{init_seed}'
        summaries.append(run_experiment(cfg, records, args.protocol, order_seed, init_seed, device, output_dir))
    if args.all_seeds:
        write_json(experiment_root / 'aggregate_30_runs.json', summarize_runs(summaries))

if __name__ == '__main__':
    main()
