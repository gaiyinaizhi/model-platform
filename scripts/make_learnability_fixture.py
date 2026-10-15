#!/usr/bin/env python3
"""Own synthetic learnability calibration; never evidence of photo accuracy."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import zipfile
from PIL import Image

CONFIG = {'epochs': 20, 'learning_rate': 0.001, 'seed': 20261015}
SPLITS = (('train', 128, 41000), ('val', 16, 51000), ('test', 16, 61000))


def generate():
    records, files, pixels_seen = [], {}, set()
    files['data.yaml'] = b'train: images/train\nval: images/val\ntest: images/test\nnames: [red_rectangle, blue_rectangle]\n'
    for split, count, seed_base in SPLITS:
        for index in range(count):
            rng = random.Random(seed_base + index)
            width, height = ((96, 64), (64, 96), (80, 80))[index % 3]
            image = Image.new('RGB', (width, height))
            image.putdata([(rng.randrange(20, 45), rng.randrange(20, 45), rng.randrange(20, 45)) for _ in range(width*height)])
            boxes = []
            for cid, cx, cy, color in [(0, .27, .35, (225, 30, 35)), (1, .73, .65, (30, 40, 225))]:
                cx += rng.uniform(-.035, .035)
                cy += rng.uniform(-.035, .035)
                bw, bh = rng.uniform(.25, .31), rng.uniform(.25, .31)
                x1, y1 = round((cx-bw/2)*width), round((cy-bh/2)*height)
                x2, y2 = round((cx+bw/2)*width), round((cy+bh/2)*height)
                image.paste(color, (x1, y1, x2, y2))
                boxes.append({'class_id': cid, 'xyxy': [x1, y1, x2, y2]})
            ph = hashlib.sha256(f'{width},{height}:'.encode()+image.tobytes()).hexdigest()
            if ph in pixels_seen:
                raise ValueError('Duplicate synthetic image')
            pixels_seen.add(ph)
            stream = io.BytesIO(); image.save(stream, format='PNG', optimize=False)
            raw = stream.getvalue()
            label_lines = []
            for box in boxes:
                x1,y1,x2,y2 = box['xyxy']
                label_lines.append(str(box['class_id'])+' '+' '.join(format(v, '.17g') for v in
                    ((x1+x2)/2/width, (y1+y2)/2/height, (x2-x1)/width, (y2-y1)/height)))
            label = ('\n'.join(label_lines)+'\n').encode()
            stem = f'{index:03d}'
            path, lp = f'images/{split}/{stem}.png', f'labels/{split}/{stem}.txt'
            files[path], files[lp] = raw, label
            records.append({'split': split, 'image': path, 'label': lp, 'width': width, 'height': height,
                            'generation_seed': seed_base+index, 'boxes': boxes, 'pixel_sha256': ph,
                            'image_sha256': hashlib.sha256(raw).hexdigest(), 'label_sha256': hashlib.sha256(label).hexdigest()})
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_STORED) as z:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2026, 10, 15, 0, 0, 0)); info.external_attr = 0o100644 << 16
            z.writestr(info, content)
    raw = out.getvalue()
    evidence = {'scope': 'Own generated two-color rectangles; learnability and UI mechanics only, not real-photo or production quality',
                'rights': 'Original procedurally generated pixels and labels; no external images or weights',
                'split_policy': 'Fixed disjoint RNG seed ranges 41000/51000/61000; 128/16/16; all images pixel-unique',
                'predeclared_training_config': CONFIG, 'predeclared_thresholds': {'evaluation': .001, 'serving': .01},
                'archive_sha256': hashlib.sha256(raw).hexdigest(), 'archive_bytes': len(raw), 'samples': records}
    return raw, evidence


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('output', type=Path)
    args=parser.parse_args(); raw, evidence=generate()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'learnability.zip').write_bytes(raw)
    (args.output/'manifest.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps({'archive_sha256':evidence['archive_sha256'], 'images':len(evidence['samples']), 'bytes':len(raw)}))
