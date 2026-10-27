import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DATASETS = {
    'hgr_shrec_2017': (14, 22, 1, 8),
    'ego_gesture': (83, 42, 1, 8),
    'ntu60': (60, 25, 2, 64),
    'ntu120': (120, 25, 2, 64),
    'nw_ucla': (10, 20, 1, 52),
}
PREPARATION_VERSION = 2
NTU60_TRAIN_SUBJECTS = {
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28,
    31, 34, 35, 38,
}
NTU120_TRAIN_SUBJECTS = NTU60_TRAIN_SUBJECTS | {
    45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74,
    78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97,
    98, 100, 103,
}
EGO_VAL_SUBJECTS = {1, 7, 12, 13, 24, 29, 33, 34, 35, 37}
EGO_TEST_SUBJECTS = {2, 9, 11, 14, 18, 19, 28, 31, 41, 47}
UCLA_ACTIONS = {action: label for label, action in enumerate(
    (1, 2, 3, 4, 5, 6, 8, 9, 11, 12))}

def _check_dataset(dataset):
    if dataset not in DATASETS:
        raise ValueError(f'Unknown dataset {dataset!r}; expected {tuple(DATASETS)}')

def _unique_file(root, name):
    matches = sorted(p for p in root.rglob(name) if 'prepared' not in p.parts)
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one {name} under {root}, found {len(matches)}')
    return matches[0]

def _shrec_records(root):
    records = []
    for split, filename in (('train', 'train_gestures.txt'), ('test', 'test_gestures.txt')):
        listing = _unique_file(root, filename)
        for line_number, line in enumerate(listing.read_text().splitlines(), 1):
            if not line.strip():
                continue
            values = [int(x) for x in line.split()]
            if len(values) < 6:
                raise ValueError(f'{listing}:{line_number}: expected official SHREC row')
            gesture, finger, subject, trial, label = values[:5]
            relative = (Path(f'gesture_{gesture}') / f'finger_{finger}' /
                        f'subject_{subject}' / f'essai_{trial}' / 'skeletons_world.txt')
            records.append({'source': str(listing.parent / relative),
                            'id': relative.parent.as_posix(), 'label': label - 1,
                            'split': split, 'subject': subject, 'view': None})
    return records, []

def _ego_records(root):
    candidates = sorted(p for p in root.rglob('train.txt')
                        if p.parent.name == 'split_files' and 'prepared' not in p.parts)
    if len(candidates) != 1:
        raise FileNotFoundError(f'Expected one split_files/train.txt under {root}')
    split_dir = candidates[0].parent
    data_root = split_dir.parent
    records = []
    for split in ('train', 'val', 'test'):
        listing = split_dir / f'{split}.txt'
        with listing.open(newline='') as stream:
            for line_number, row in enumerate(csv.reader(stream), 1):
                if not row:
                    continue
                if len(row) != 4:
                    raise ValueError(f'{listing}:{line_number}: expected path,label,length,subject')
                relative, label, length, subject = row
                subject = int(subject)
                expected = ('test' if subject in EGO_TEST_SUBJECTS else
                            'val' if subject in EGO_VAL_SUBJECTS else 'train')
                if not 1 <= subject <= 50 or expected != split:
                    raise ValueError(f'{listing}:{line_number}: subject {subject} violates {split} split')
                source = Path(relative)
                if source.is_absolute():
                    raise ValueError(f'{listing}:{line_number}: use a relative path inside the dataset')
                source = data_root / source
                records.append({'source': str(source), 'id': relative,
                                'label': int(label), 'split': split,
                                'subject': subject, 'view': None,
                                'declared_frames': int(length)})
    return records, []

def _ntu_records(dataset, root):
    filename = ('NTU_RGBD_samples_with_missing_skeletons.txt' if dataset == 'ntu60'
                else 'NTU_RGBD120_samples_with_missing_skeletons.txt')
    text = (Path(__file__).parent / 'metadata' / filename).read_text()
    missing = set(re.findall(r'S\d{3}C\d{3}P\d{3}R\d{3}A\d{3}', text))
    subjects = NTU60_TRAIN_SUBJECTS if dataset == 'ntu60' else NTU120_TRAIN_SUBJECTS
    records, excluded = [], []
    for path in sorted(root.rglob('*.skeleton')):
        match = re.fullmatch(r'S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})', path.stem)
        if match is None:
            raise ValueError(f'Invalid NTU sample name: {path.name}')
        setup, view, subject, repeat, action = map(int, match.groups())
        if dataset == 'ntu60' and (setup > 17 or action > 60):
            excluded.append({'id': path.stem, 'reason': 'outside_ntu60'})
            continue
        if path.stem in missing:
            excluded.append({'id': path.stem, 'reason': 'official_missing_skeleton'})
            continue
        max_subject = 40 if dataset == 'ntu60' else 106
        if not (1 <= subject <= max_subject and 1 <= setup <= 32 and 1 <= view <= 3):
            raise ValueError(f'Invalid NTU metadata: {path.name}')
        records.append({'source': str(path), 'id': path.stem, 'label': action - 1,
                        'split': 'train' if subject in subjects else 'test',
                        'subject': subject, 'view': view, 'setup': setup})
    return records, excluded

def _ucla_records(root):
    records = []
    for path in sorted(root.rglob('*.json')):
        if 'prepared' in path.parts:
            continue
        match = re.fullmatch(r'a(\d+)_s(\d+)_e(\d+)_v(\d+)', path.stem)
        if match is None:
            continue
        action, subject, trial, view = map(int, match.groups())
        if action not in UCLA_ACTIONS or view not in (1, 2, 3):
            raise ValueError(f'Unknown UCLA action/view: {path.name}')
        records.append({'source': str(path), 'id': path.stem,
                        'label': UCLA_ACTIONS[action],
                        'split': 'test' if view == 3 else 'train',
                        'subject': subject, 'view': view})
    return records, []

def _read_ntu(path):
    
    tracks = {}
    with Path(path).open() as stream:
        frames = int(stream.readline())
        for frame in range(frames):
            bodies = int(stream.readline())
            for _ in range(bodies):
                body_id = stream.readline().split()[0]
                joints = int(stream.readline())
                if joints != 25:
                    raise ValueError(f'{path}: expected 25 joints, got {joints}')
                points = np.array([[float(v) for v in stream.readline().split()[:3]]
                                   for _ in range(joints)], dtype=np.float32)
                if points.shape != (25, 3) or not np.isfinite(points).all():
                    raise ValueError(f'{path}: invalid body coordinates at frame {frame}')
                if np.any(points != 0):
                    tracks.setdefault(body_id, []).append((frame, points))
    if frames <= 0 or not tracks:
        raise ValueError(f'{path}: no usable skeleton tracks')
    if len(tracks) > 1:
        tracks = {key: track for key, track in tracks.items() if len(track) > 11}
        if not tracks:
            raise ValueError(f'{path}: all tracks were removed by the <=11-frame filter')
    def motion(item):
        points = np.stack([p for _, p in item[1]])
        return float(points.reshape(-1, 3).var(axis=0).sum())
    ranked = sorted(tracks.items(), key=lambda item: (-motion(item), item[0]))[:2]
    x = np.zeros((3, frames, 25, 2), dtype=np.float32)
    for person, (_, track) in enumerate(ranked):
        for frame, points in track:
            x[:, frame, :, person] = points.T
    return x[:, np.any(x != 0, axis=(0, 2, 3))]

def _read_source(dataset, record):
    path = record['source']
    if dataset.startswith('ntu'):
        return _read_ntu(path)
    if dataset == 'hgr_shrec_2017':
        points = np.loadtxt(path, dtype=np.float32).reshape(-1, 22, 3)
    elif dataset == 'ego_gesture':
        points = np.loadtxt(path, delimiter=',', dtype=np.float32).reshape(-1, 42, 3)
    else:
        points = np.asarray(json.loads(Path(path).read_text())['skeletons'], dtype=np.float32)
        if points.ndim != 3 or points.shape[1:] != (20, 3):
            raise ValueError(f'{path}: expected UCLA [T,20,3], got {points.shape}')
    return np.transpose(points, (2, 0, 1))[..., None]

def _normalize(dataset, x):
    
    x = np.asarray(x, dtype=np.float32).copy()
    if x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError('Skeleton must have frames and finite coordinates')
    if dataset == 'ego_gesture':
        masks = [np.any(x[:, :, offset:offset + 21, 0] != 0, axis=(0, 2))
                 for offset in (0, 21)]
        for offset, valid in zip((0, 21), masks):
            if valid.any():
                lengths = np.linalg.norm(
                    x[:, :, offset + 5, 0] - x[:, :, offset, 0], axis=0)
                scale = lengths.mean()
                if scale <= 1e-8:
                    raise ValueError('Ego hand has zero wrist-to-index scale')
                x[:, :, offset:offset + 21, 0] /= scale
        if not any(mask.any() for mask in masks):
            raise ValueError('Skeleton contains no present hand')
        for offset, valid in zip((0, 21), masks):
            x[:, ~valid, offset:offset + 21, 0] = 0
        return x
    valid = np.any(x != 0, axis=(0, 2))  
    if not valid.any():
        raise ValueError('Skeleton contains no present body')
    if dataset == 'hgr_shrec_2017':
        lengths = np.linalg.norm(
            x[:, :, 1, 0] - x[:, :, 0, 0], axis=0)
        scale = lengths.mean()
        if scale <= 1e-8:
            raise ValueError('SHREC hand has zero wrist-to-palm scale')
        x /= scale
        return x
    first = np.flatnonzero(valid[:, 0])[0]
    origin = x[:, first, 1, 0].copy()
    x -= origin[:, None, None, None]
    for person in range(x.shape[-1]):
        x[:, ~valid[:, person], :, person] = 0
    if dataset == 'nw_ucla':
        selected = x[:, :, :, 0][:, valid[:, 0], :]
        lo = selected.min(axis=(1, 2))[:, None, None, None]
        span = (selected.max(axis=(1, 2)) - selected.min(axis=(1, 2)))[:, None, None, None]
        x = 2 * (x - lo) / np.maximum(span, 1e-8) - 1
        x[:, ~valid[:, 0], :, 0] = 0
    return x.astype(np.float32, copy=False)

def prepare_data(dataset, root):
    
    _check_dataset(dataset)
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Dataset directory does not exist: {root}')
    if dataset == 'hgr_shrec_2017':
        records, excluded = _shrec_records(root)
    elif dataset == 'ego_gesture':
        records, excluded = _ego_records(root)
    elif dataset.startswith('ntu'):
        records, excluded = _ntu_records(dataset, root)
    else:
        records, excluded = _ucla_records(root)
    if not records:
        raise FileNotFoundError(f'No supported {dataset} source samples found under {root}')
    labels, joints, persons, _ = DATASETS[dataset]
    ids, sources = set(), set()
    for record in records:
        source = Path(record['source']).resolve()
        try:
            source.relative_to(root)
        except ValueError:
            raise ValueError(f'Source path escapes dataset directory: {source}') from None
        if record['id'] in ids or source in sources:
            raise ValueError(f'Duplicate or overlapping sample: {record["id"]}')
        if not 0 <= record['label'] < labels:
            raise ValueError(f'Invalid class label in {record["id"]}: {record["label"]}')
        if not source.is_file():
            raise FileNotFoundError(f'Missing sample: {source}')
        ids.add(record['id']); sources.add(source)
    prepared = root / 'prepared'
    prepared.mkdir(exist_ok=True)
    old = {}
    manifest_path = prepared / 'manifest.json'
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if existing.get('version') == PREPARATION_VERSION and existing.get('dataset') == dataset:
            old = {r['id']: r for r in existing['records']}
    output = []
    for index, record in enumerate(records):
        source = Path(record['source']).resolve()
        stat = source.stat()
        key = hashlib.sha256(record['id'].encode()).hexdigest()[:24]
        relative_path = f'prepared/{key}.npy'
        dest = root / relative_path
        previous = old.get(record['id'], {})
        if not (dest.is_file() and previous.get('source_size') == stat.st_size and
                previous.get('source_mtime_ns') == stat.st_mtime_ns):
            x = _normalize(dataset, _read_source(dataset, record))
            if x.shape[0] != 3 or x.shape[2:] != (joints, persons):
                raise ValueError(f'{source}: wrong shape {x.shape}')
            temporary = dest.with_suffix('.npy.tmp')
            with temporary.open('wb') as stream:
                np.save(stream, x, allow_pickle=False)
            temporary.replace(dest)
        item = {**record, 'source': source.relative_to(root).as_posix(),
                'path': relative_path, 'source_size': stat.st_size,
                'source_mtime_ns': stat.st_mtime_ns}
        output.append(item)
        if (index + 1) % 1000 == 0:
            print(f'Prepared {dataset}: {index + 1}/{len(records)}', flush=True)
    payload = {'version': PREPARATION_VERSION, 'dataset': dataset,
               'format': 'float32 C,T,V,M; prepared coordinates; variable length', 'records': output}
    temporary = manifest_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    temporary.replace(manifest_path)
    report = {'dataset': dataset, 'samples': len(output),
              'splits': dict(Counter(r['split'] for r in output)), 'excluded': excluded,
              'normalization': 'See datasets/metadata/SOURCES.md'}
    (prepared / 'preparation_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return load_records(dataset, root)

def load_records(dataset, root):
    
    _check_dataset(dataset)
    root = Path(root).expanduser().resolve()
    path = root / 'prepared' / 'manifest.json'
    if not path.is_file():
        raise FileNotFoundError(f'{path} is missing. Run the prepare command for {dataset} first.')
    manifest = json.loads(path.read_text())
    if manifest.get('dataset') != dataset or manifest.get('version') != PREPARATION_VERSION:
        raise ValueError(f'{path}: incompatible manifest; rerun data preparation')
    records = []
    for item in manifest['records']:
        record = dict(item)
        resolved = (root / record['path']).resolve()
        try:
            resolved.relative_to(root / 'prepared')
        except ValueError:
            raise ValueError(f'{path}: prepared sample path escapes its directory') from None
        record['path'] = str(resolved)
        records.append(record)
    return records

def _sample_frames(x, length, augment):
    frames = x.shape[1]
    if frames == 0:
        raise ValueError('Cannot sample an empty sequence')
    if frames == length and not augment:
        return x.copy()
    if augment and frames > 1 and np.random.random() < 0.5:
        amount = np.random.uniform()
        x = x[:, :-1] * (1 - amount) + x[:, 1:] * amount
        frames = x.shape[1]
    if frames < length:
        indices = np.minimum(np.arange(length), frames - 1)
    else:
        indices = np.linspace(0, frames - 1, length).astype(np.int64)
    return x[:, indices].copy()

def _shrec_view(x, augment):
    
    absent = ~np.any(x != 0, axis=(0, 2))
    if augment:
        x *= np.random.uniform(0.1, 5.0)
        x += np.random.uniform(-0.1, 0.1, size=3).astype(np.float32)[:, None, None, None]
        if np.random.random() >= 0.5:
            angle = np.deg2rad(np.random.uniform(-5, 5, size=3))
            sx, sy, sz = np.sin(angle); cx, cy, cz = np.cos(angle)
            rotation = np.array([[cz*cy, sz*cy, -sy],
                                 [cz*sy*sx-sz*cx, sz*sy*sx+cz*cx, cy*sx],
                                 [cz*sy*cx+sz*sx, sz*sy*cx-cz*sx, cy*cx]], dtype=np.float32)
            
            x = np.einsum('cd,ctvm->dtvm', rotation, x)
    x -= x[:, 0, 1, 0].copy()[:, None, None, None]
    x[:, absent[:, 0], :, 0] = 0
    return x

class SkeletonDataset(Dataset):
    
    def __init__(self, records, seq_len=None, views=1, augment=False,
                 dataset='hgr_shrec_2017'):
        _check_dataset(dataset)
        self.records = list(records)
        self.seq_len = DATASETS[dataset][3] if seq_len is None else int(seq_len)
        self.views = int(views)
        self.augment = bool(augment)
        self.dataset = dataset
        if self.seq_len <= 0 or self.views <= 0:
            raise ValueError('seq_len and views must be positive')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        x = (np.asarray(record['x'], dtype=np.float32) if 'x' in record else
             np.load(record['path'], mmap_mode='r', allow_pickle=False))
        _, joints, persons, _ = DATASETS[self.dataset]
        if x.ndim != 4 or x.shape[0] != 3 or x.shape[2:] != (joints, persons):
            raise ValueError(f'Expected C,T,V,M for {self.dataset}, got {x.shape}')
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite skeleton coordinates')
        views = []
        replay = bool(record.get('replay', False))
        for _ in range(self.views):
            temporal_augment = self.augment and not replay and self.dataset != 'ego_gesture'
            view = _sample_frames(x, self.seq_len, temporal_augment)
            if not replay and self.dataset == 'hgr_shrec_2017':
                view = _shrec_view(view, self.augment)
            views.append(torch.from_numpy(np.ascontiguousarray(view, dtype=np.float32)))
        return {'x': views[0] if self.views == 1 else torch.stack(views),
                'label': int(record['label'])}
