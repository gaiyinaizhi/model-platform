#!/usr/bin/env python3
"""Make a wholly synthetic two-class YOLO ZIP for mechanics, not quality proof."""
import argparse
import io
from pathlib import Path
import random
import struct
import zipfile
import zlib


def fixture_bytes(*, test=True, variant=0):
    output = io.BytesIO()
    def chunk(kind, content):
        return struct.pack('>I', len(content)) + kind + content + struct.pack('>I', zlib.crc32(kind+content))
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_STORED) as archive:
        meta = 'train: images/train\nval: images/val\n' + ('test: images/test\n' if test else '') + 'names: [red_square, blue_square]\n'
        archive.writestr('data.yaml', meta)
        for si, (split, count) in enumerate([('train', 8), ('val', 2)] + ([('test', 2)] if test else [])):
            for index in range(count):
                rng = random.Random(20261015 + si*100 + index + variant*10000)
                raw = bytearray()
                for y in range(64):
                    raw.append(0)
                    for x in range(64):
                        color = (220, 35, 40) if 8<=x<24 and 16<=y<32 else (30, 45, 220) if 40<=x<56 and 32<=y<48 else (rng.randrange(20,50), rng.randrange(20,50), rng.randrange(20,50))
                        raw.extend(color)
                png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB',64,64,8,2,0,0,0)) + chunk(b'IDAT',zlib.compress(raw)) + chunk(b'IEND',b'')
                archive.writestr(f'images/{split}/{index}.png', png)
                archive.writestr(f'labels/{split}/{index}.txt', '0 0.25 0.375 0.25 0.25\n1 0.75 0.625 0.25 0.25\n')
    return output.getvalue()


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('output', type=Path);p.add_argument('--no-test',action='store_true');p.add_argument('--variant',type=int,default=0)
    a=p.parse_args(); a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('xb') as f: f.write(fixture_bytes(test=not a.no_test,variant=a.variant))
    print('Created own synthetic fixture:', a.output, '; mechanics only, not quality proof')
