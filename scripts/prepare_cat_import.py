#!/usr/bin/env python3
"""Deterministic bounded derivative of the reviewed Cat cache; no network."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.fetch_detection import verify_cache, EXPECTED_MANIFEST_SHA256
MAX_SIDE = 256
CONFIG = {'epochs': 20, 'learning_rate': 0.001, 'seed': 20261015}


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def derivative(image, boxes):
    """Normalized continuous coordinates retain exact independent axis scales."""
    from PIL import Image
    image = image.convert('RGB')
    width, height = image.size
    scale = min(1., MAX_SIDE / max(width, height))
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    if size != image.size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, format='PNG', optimize=False)
    labels = []
    for box in boxes:
        x1, y1, x2, y2 = box['xyxy']
        if box['class_id'] != 0 or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError('Invalid original Cat box')
        labels.append('0 ' + ' '.join(format(x, '.17g') for x in
                      ((x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height)))
    return buffer.getvalue(), ('\n'.join(labels) + ('\n' if labels else '')).encode(), size


def prepare(cache, output):
    from PIL import Image
    from model_platform.dataset_validation import Validation
    verify_cache(cache)
    manifest = json.loads((cache/'manifest.json').read_text())
    if output.exists():
        raise ValueError('Output must be new; never overwrite evidence')
    entries = {'data.yaml': b'train: images/train\nval: images/val\ntest: images/test\nnames: [Cat]\n'}
    records = []
    for source_split, split in [('train', 'train'), ('validation', 'val'), ('test', 'test')]:
        for record in manifest['splits'][source_split]:
            raw = (cache/record['image_path']).read_bytes()
            if sha256(raw) != record['sha256']:
                raise ValueError('Source changed after validation')
            with Image.open(io.BytesIO(raw)) as source:
                png, labels, size = derivative(source, record['boxes'])
            stem = record['image_id']
            image_path, label_path = f'images/{split}/{stem}.png', f'labels/{split}/{stem}.txt'
            entries[image_path], entries[label_path] = png, labels
            records.append({'split': split, 'official_split': source_split, 'image_id': stem,
                            'image': image_path, 'label': label_path, 'width': size[0], 'height': size[1],
                            'source_width': record['width'], 'source_height': record['height'],
                            'scale_x': size[0]/record['width'], 'scale_y': size[1]/record['height'],
                            'source_sha256': record['sha256'], 'image_sha256': sha256(png),
                            'label_sha256': sha256(labels), 'boxes': len(record['boxes']),
                            'attribution': record['attribution']})
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as z:
        for name, content in sorted(entries.items()):
            entry = zipfile.ZipInfo(name, date_time=(2026, 10, 15, 0, 0, 0))
            entry.external_attr = 0o100644 << 16
            z.writestr(entry, content)
    raw = archive.getvalue()
    if len(raw) > 32*1024*1024 or sum(r['width']*r['height'] for r in records) > 20_000_000:
        raise ValueError('Derivative exceeds existing import/training budget')
    report, _, _ = Validation('allow_empty').validate(raw)
    if not report['valid']:
        raise ValueError('Derived ZIP rejected: ' + json.dumps(report))
    evidence = {'scope': 'Fixed Open Images Cat derivative; not official Open Images benchmark',
                'original_manifest_sha256': EXPECTED_MANIFEST_SHA256, 'archive_sha256': sha256(raw),
                'archive_bytes': len(raw), 'predeclared_training_config': CONFIG,
                'predeclared_thresholds': {'evaluation': .001, 'serving': .01},
                'transformation': {'max_side': MAX_SIDE, 'upsample': False, 'RGB': True,
                                   'orientation': 'raw-decoded-no-exif-transpose', 'resize': 'Pillow bilinear',
                                   'format': 'PNG', 'labels': 'original normalized continuous coordinates'},
                'licenses': {'annotations': 'CC BY 4.0', 'image_metadata': 'CC BY 2.0',
                             'caveat': 'Open Images does not guarantee individual image licensing; retain attribution'},
                'samples': records, 'validation': report}
    output.mkdir(parents=True)
    (output/'cat-import.zip').write_bytes(raw)
    (output/'source-and-derivative-manifest.json').write_text(json.dumps(evidence, indent=2)+'\n')
    return evidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = prepare(args.cache, args.output)
    print(json.dumps({k: result[k] for k in ('archive_sha256', 'archive_bytes', 'predeclared_training_config')}))
