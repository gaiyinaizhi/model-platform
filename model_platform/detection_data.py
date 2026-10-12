"""Read-only, hash-pinned public Cat cache. No runtime network or arbitrary paths."""
import hashlib
import json
import os
from pathlib import Path
import stat
from .algorithm import TrainingFailure

# Filled from the reviewed fixed public selection before final acceptance.
MANIFEST_SHA256 = '06a9b65a1e1c9826b2e4c4ae0284d28b6c2ab89b2dfc63d83ebca796dc2e3387'
REFERENCE_DIRECTORY = 'openimages-cat-reviewed-v1'
MAX_IMAGE_BYTES = 20 * 1024 * 1024


def _read(path, maximum):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        st = os.fstat(stream.fileno())
        if not stat.S_ISREG(st.st_mode) or not 0 < st.st_size <= maximum:
            raise ValueError('Invalid cache file')
        return stream.read(maximum + 1)


def load_dataset(runtime):
    from PIL import Image
    import io
    root = runtime.home / 'datasets' / REFERENCE_DIRECTORY
    try:
        if root.is_symlink() or root.parent.is_symlink():
            raise ValueError('Cache symlink forbidden')
        content = _read(root / 'manifest.json', 8 * 1024 * 1024)
        if hashlib.sha256(content).hexdigest() != MANIFEST_SHA256:
            raise ValueError('Unreviewed cache manifest')
        manifest = json.loads(content)
        splits = {}
        seen_ids, seen_bytes, seen_pixels = set(), set(), set()
        for split, count in (('train', 128), ('validation', 32), ('test', 32)):
            records = manifest['splits'][split]
            if len(records) != count:
                raise ValueError('Invalid split count')
            prepared = []
            for record in records:
                relative = Path(record['image_path'])
                if relative.is_absolute() or '..' in relative.parts or not relative.parts:
                    raise ValueError('Invalid cache path')
                path = root / relative
                if any((root.joinpath(*relative.parts[:i])).is_symlink() for i in range(1, len(relative.parts) + 1)):
                    raise ValueError('Cache symlink forbidden')
                raw = _read(path, MAX_IMAGE_BYTES)
                sha = hashlib.sha256(raw).hexdigest()
                if sha != record['sha256'] or record['image_id'] in seen_ids or sha in seen_bytes:
                    raise ValueError('Image integrity or split overlap')
                image = Image.open(io.BytesIO(raw))
                if image.width * image.height > 20_000_000:
                    raise ValueError('Image exceeds decode budget')
                image = image.convert('RGB')
                if image.size != (record['width'], record['height']):
                    raise ValueError('Image dimensions mismatch')
                # Pixel digest format is fixed by the reviewed download manifest.
                pixel_sha = hashlib.sha256(f'{image.width},{image.height}:'.encode() + image.tobytes()).hexdigest()
                if pixel_sha != record['pixel_sha256'] or pixel_sha in seen_pixels:
                    raise ValueError('Decoded pixel split overlap')
                seen_ids.add(record['image_id']); seen_bytes.add(sha); seen_pixels.add(pixel_sha)
                if len(record['boxes']) > 100:
                    raise ValueError('Annotation budget exceeded')
                for box in record['boxes']:
                    x1, y1, x2, y2 = box['xyxy']
                    if (box['class_id'] != 0 or box['category_id'] != 1 or box['iscrowd'] != 0
                            or not 0 <= x1 < x2 <= image.width or not 0 <= y1 < y2 <= image.height):
                        raise ValueError('Invalid annotation')
                prepared.append({'record': record, 'image': image})
            splits[split] = prepared
        return manifest, splits
    except (OSError, ValueError, KeyError, TypeError):
        raise TrainingFailure('detection_dataset_unavailable_or_invalid') from None
