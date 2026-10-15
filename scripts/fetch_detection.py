#!/usr/bin/env python3
"""Fetch the fixed local-only Open Images Cat engineering subset; no training.

Official source records are regenerated at runtime, never embedded in this file.
Requires the optional Pillow dependency. See docs/detection-data.md.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import csv
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request
import warnings

EXPECTED_MANIFEST_SHA256 = "06a9b65a1e1c9826b2e4c4ae0284d28b6c2ab89b2dfc63d83ebca796dc2e3387"
MID = "/m/01yrx"
ALLOWED_HOSTS = {"storage.googleapis.com", "open-images-dataset.s3.amazonaws.com"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_DECODED_PIXELS = 16 * 1024 * 1024
MAX_SOURCE_STREAM_BYTES = 800 * 1024 * 1024
SOURCE_SECONDS = 180

MANIFEST_BASE = {'annotations_license': 'CC-BY-4.0',
 'audit': {'counts': {'test': {'boxes': 26, 'images': 32, 'negative': 8, 'positive': 24},
                      'train': {'boxes': 110, 'images': 128, 'negative': 32, 'positive': 96},
                      'validation': {'boxes': 26, 'images': 32, 'negative': 8, 'positive': 24}},
           'exact_cross_split_overlaps': 0,
           'near_duplicate_candidates': [],
           'near_duplicate_screen': '64-bit dHash Hamming distance <=4, cross split; heuristic only, '
                                    'not a scene-independence guarantee'},
 'class_mapping': [{'category_id': 1,
                    'label_name': '/m/01yrx',
                    'leaf_class': True,
                    'model_class_id': 0,
                    'name': 'Cat'}],
 'class_source_integrity': {'bbox_labels_600_hierarchy.json': {'sha256': 'a28f06e83c8378c4285d92d768ceda7568a48cac3c02994b6c346df804e7d612',
                                                               'url': 'https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy.json'},
                            'class-descriptions-boxable.csv': {'sha256': '2fa47fb1b87e71c90b9fbee2dc1184eb8c59601bb811ea4424b200653cedff8e',
                                                               'url': 'https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv'}},
 'dataset_source': 'https://storage.googleapis.com/openimages/web/download_v7.html',
 'image_license': 'CC-BY-2.0',
 'license_verification': 'Per-image official Open Images metadata checked; original Flickr licenses not '
                         'independently live-verified; no commercial-use warranty; local engineering '
                         'verification only, do not redistribute images.',
 'schema': 'openimages-cat-engineering-subset/v1',
 'selection': {'evaluation': 'COCO-style AP on complete selected Cat task, not official Open Images '
                             'challenge metric; deliberately balanced engineering subset, not '
                             'representative benchmark',
               'excluded': 'unknown image-level Cat; any Cat group-of or depiction; absent attribution; '
                           'nonzero/unknown rotation; cross-split identical pixels/bytes/source image',
               'ordering': 'lexicographic ImageID within each human-verified positive/negative '
                           'candidate pool',
               'test': [24, 8],
               'train': [96, 32],
               'validation': [24, 8]},
 'source_headers': {'https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv': {'Content-Length': '45227339',
                                                                                                             'ETag': '"d832feb775b3cb78077bf8ae350adce5"',
                                                                                                             'Last-Modified': 'Thu, '
                                                                                                                              '17 '
                                                                                                                              'May '
                                                                                                                              '2018 '
                                                                                                                              '14:33:36 '
                                                                                                                              'GMT',
                                                                                                             'x-goog-generation': '1526567616185301',
                                                                                                             'x-goog-hash': 'crc32c=788fxw=='},
                    'https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv': {'Content-Length': '638407721',
                                                                                                                       'ETag': '"a4ac0bcedb5c2df4d1b1230dc6b41b8f"',
                                                                                                                       'Last-Modified': 'Thu, '
                                                                                                                                        '17 '
                                                                                                                                        'May '
                                                                                                                                        '2018 '
                                                                                                                                        '14:33:46 '
                                                                                                                                        'GMT',
                                                                                                                       'x-goog-generation': '1526567626302755',
                                                                                                                       'x-goog-hash': 'crc32c=FrUT4g=='},
                    'https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv': {'Content-Length': '15245485',
                                                                                                                         'ETag': '"643a54e43b0bab8acce8817b4a569780"',
                                                                                                                         'Last-Modified': 'Thu, '
                                                                                                                                          '17 '
                                                                                                                                          'May '
                                                                                                                                          '2018 '
                                                                                                                                          '14:33:38 '
                                                                                                                                          'GMT',
                                                                                                                         'x-goog-generation': '1526567618256414',
                                                                                                                         'x-goog-hash': 'crc32c=vYP9Ww=='},
                    'https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv': {'Content-Length': '77484237',
                                                                                               'ETag': '"1cc058a7003b4e73d47642276e9b123b"',
                                                                                               'Last-Modified': 'Wed, '
                                                                                                                '08 '
                                                                                                                'May '
                                                                                                                '2019 '
                                                                                                                '08:24:09 '
                                                                                                                'GMT',
                                                                                               'x-goog-generation': '1557303849552598',
                                                                                               'x-goog-hash': 'crc32c=RX/26w=='},
                    'https://storage.googleapis.com/openimages/v5/test-annotations-human-imagelabels-boxable.csv': {'Content-Length': '32055857',
                                                                                                                    'ETag': '"207ed0ce2755ad5a689c8eb6eeb039bf"',
                                                                                                                    'Last-Modified': 'Wed, '
                                                                                                                                     '08 '
                                                                                                                                     'May '
                                                                                                                                     '2019 '
                                                                                                                                     '08:24:06 '
                                                                                                                                     'GMT',
                                                                                                                    'x-goog-generation': '1557303846555264',
                                                                                                                    'x-goog-hash': 'crc32c=BDZXJQ=='},
                    'https://storage.googleapis.com/openimages/v5/train-annotations-human-imagelabels-boxable.csv': {'Content-Length': '376764810',
                                                                                                                     'ETag': '"f2709ae3f91391549c56d45a4a5ffec6"',
                                                                                                                     'Last-Modified': 'Wed, '
                                                                                                                                      '08 '
                                                                                                                                      'May '
                                                                                                                                      '2019 '
                                                                                                                                      '08:24:32 '
                                                                                                                                      'GMT',
                                                                                                                     'x-goog-generation': '1557303872084087',
                                                                                                                     'x-goog-hash': 'crc32c=x1M2+w=='},
                    'https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv': {'Content-Length': '25105048',
                                                                                                     'ETag': '"c5e8200df129ea6867e913e8b21fcab9"',
                                                                                                     'Last-Modified': 'Wed, '
                                                                                                                      '08 '
                                                                                                                      'May '
                                                                                                                      '2019 '
                                                                                                                      '08:24:04 '
                                                                                                                      'GMT',
                                                                                                     'x-goog-generation': '1557303844893197',
                                                                                                     'x-goog-hash': 'crc32c=9xFCFA=='},
                    'https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels-boxable.csv': {'Content-Length': '10649275',
                                                                                                                          'ETag': '"2e129f40a209e65dfe4342ed0e7bd5f2"',
                                                                                                                          'Last-Modified': 'Wed, '
                                                                                                                                           '08 '
                                                                                                                                           'May '
                                                                                                                                           '2019 '
                                                                                                                                           '08:24:04 '
                                                                                                                                           'GMT',
                                                                                                                          'x-goog-generation': '1557303844533180',
                                                                                                                          'x-goog-hash': 'crc32c=h6fFzA=='},
                    'https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv': {'Content-Length': '2258447590',
                                                                                                      'ETag': '"3c3e70cfaba5757ea5c2604b19cac3b2"',
                                                                                                      'Last-Modified': 'Mon, '
                                                                                                                       '03 '
                                                                                                                       'Feb '
                                                                                                                       '2020 '
                                                                                                                       '15:15:49 '
                                                                                                                       'GMT',
                                                                                                      'x-goog-generation': '1580742949706099',
                                                                                                      'x-goog-hash': 'crc32c=fw7RlA=='}},
 'sources': {'test': {'boxes': 'https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv',
                      'labels': 'https://storage.googleapis.com/openimages/v5/test-annotations-human-imagelabels-boxable.csv',
                      'metadata': 'https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv'},
             'train': {'boxes': 'https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv',
                       'labels': 'https://storage.googleapis.com/openimages/v5/train-annotations-human-imagelabels-boxable.csv',
                       'metadata': 'https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv'},
             'validation': {'boxes': 'https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv',
                            'labels': 'https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels-boxable.csv',
                            'metadata': 'https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv'}}}


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def check_url(url):
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS
            or parsed.username or parsed.password or parsed.port not in (None, 443)):
        raise ValueError("Download URL is outside the official HTTPS allowlist")
    return url


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_official(url):
    return urllib.request.build_opener(SafeRedirect()).open(check_url(url), timeout=60)


def read_bounded(url, maximum):
    start = time.monotonic()
    parts = []
    total = 0
    with open_official(url) as response:
        if int(response.headers.get("Content-Length", "0")) > maximum:
            raise ValueError("Image/source exceeds download size budget")
        while chunk := response.read(min(65536, maximum + 1 - total)):
            total += len(chunk)
            if total > maximum or time.monotonic() - start > SOURCE_SECONDS:
                raise ValueError("Download exceeded byte/time budget")
            parts.append(chunk)
    return b"".join(parts)


def source_rows(url):
    expected = MANIFEST_BASE["source_headers"][url]
    pinned_url = url + "?generation=" + expected["x-goog-generation"]
    with open_official(pinned_url) as response:
        for field in ("ETag", "x-goog-generation", "Content-Length"):
            if response.headers.get(field) != expected[field]:
                raise ValueError("Official source identity changed: " + field)
        def lines():
            total = 0
            start = time.monotonic()
            while line := response.readline(2 * 1024 * 1024):
                total += len(line)
                if len(line) >= 2 * 1024 * 1024:
                    raise ValueError("CSV line exceeds budget")
                if total > MAX_SOURCE_STREAM_BYTES or time.monotonic() - start > SOURCE_SECONDS:
                    raise ValueError("Source stream exceeded byte/time budget")
                yield line.decode("utf-8")
        yield from csv.DictReader(lines())


def candidates(split, count):
    urls = MANIFEST_BASE["sources"][split]
    positive, negative = {}, {}
    rows = source_rows(urls["labels"])
    try:
        for row in rows:
            if row["LabelName"] != MID or row["Source"] not in ("verification", "crowdsource-verification"):
                continue
            confidence = row["Confidence"]
            if confidence == "1" and len(positive) < 3 * count:
                positive[row["ImageID"]] = row
            elif confidence == "0" and len(negative) < count:
                negative[row["ImageID"]] = row
            if len(positive) == 3 * count and len(negative) == count:
                break
    finally:
        rows.close()
    if len(positive) != 3 * count or len(negative) != count:
        raise ValueError("Insufficient verified candidate pool")
    ids = set(positive) | set(negative)
    boxes = {image_id: [] for image_id in ids}
    rows = source_rows(urls["boxes"])
    previous = ""
    try:
        for row in rows:
            image_id = row["ImageID"]
            if image_id < previous:
                raise ValueError("Official box source order changed")
            previous = image_id
            if image_id > max(ids):
                break
            if image_id in ids and row["LabelName"] == MID:
                boxes[image_id].append(row)
    finally:
        rows.close()
    eligible = {
        image_id for image_id in ids
        if (image_id in positive and boxes[image_id]
            and all(b["IsGroupOf"] == "0" and b["IsDepiction"] == "0" for b in boxes[image_id]))
        or (image_id in negative and not boxes[image_id])
    }
    metadata = {}
    # Metadata is not ordered by ImageID; scan this complete source.
    for row in source_rows(urls["metadata"]):
        if (row["ImageID"] in eligible
                and row["License"] == "https://creativecommons.org/licenses/by/2.0/"
                and row["Author"] and row["OriginalLandingURL"]
                and row["Rotation"] in ("0", "0.0")):
            metadata[row["ImageID"]] = row
    return [{"image_id": image_id, "official_split": split,
             "label": positive.get(image_id, negative.get(image_id)),
             "boxes_raw": boxes[image_id], "attribution": metadata[image_id]}
            for image_id in sorted(metadata)]


def decode_image(data):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = MAX_DECODED_PIXELS
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "JPEG" or image.width * image.height > MAX_DECODED_PIXELS:
                raise ValueError("Unsupported image format or decoded size")
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            rgb = image.convert("RGB")
            rgb.load()
            return rgb


def local_image(root, relative):
    path = root / relative
    if path.is_symlink() or root.resolve() not in path.resolve().parents:
        raise ValueError("Unsafe image cache path")
    return path


def materialize(root, candidate):
    row = dict(candidate)
    split, image_id = row["official_split"], row["image_id"]
    relative = f"images/{split}/{image_id}.jpg"
    path = local_image(root, relative)
    url = f"https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
    if path.exists():
        if path.stat().st_size > MAX_IMAGE_BYTES:
            raise ValueError("Cached image exceeds budget")
        data = path.read_bytes()
    else:
        data = read_bounded(url, MAX_IMAGE_BYTES)
    rgb = decode_image(data)
    width, height = rgb.size
    row.update(image_path=relative, image_url=url, sha256=sha256(data),
               pixel_sha256=sha256(f"{width},{height}:".encode() + rgb.tobytes()),
               width=width, height=height)
    row["boxes"] = [{"category_id": 1, "class_id": 0,
                     "xyxy": [float(b["XMin"]) * width, float(b["YMin"]) * height,
                              float(b["XMax"]) * width, float(b["YMax"]) * height],
                     "iscrowd": 0} for b in row["boxes_raw"]]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
    return row


def validate(root, manifest):
    from PIL import Image
    counts, seen_ids, seen_pixels, seen_sources, hashes = {}, set(), set(), set(), []
    for split, records in manifest["splits"].items():
        positive = negative = box_count = 0
        for row in records:
            if row["official_split"] != split or row["image_id"] in seen_ids:
                raise ValueError("Image split/identity mismatch")
            seen_ids.add(row["image_id"])
            path = local_image(root, row["image_path"])
            if path.stat().st_size > MAX_IMAGE_BYTES:
                raise ValueError("Cached image exceeds budget")
            data = path.read_bytes()
            if sha256(data) != row["sha256"]:
                raise ValueError("Image cache SHA-256 mismatch")
            rgb = decode_image(data)
            width, height = rgb.size
            pixel_digest = sha256(f"{width},{height}:".encode() + rgb.tobytes())
            if (width, height) != (row["width"], row["height"]) or pixel_digest != row["pixel_sha256"]:
                raise ValueError("Pixel digest/dimension mismatch")
            if pixel_digest in seen_pixels or row["attribution"]["OriginalLandingURL"] in seen_sources:
                raise ValueError("Duplicate pixels/source image")
            seen_pixels.add(pixel_digest)
            seen_sources.add(row["attribution"]["OriginalLandingURL"])
            gray = rgb.convert("L").resize((9, 8), Image.Resampling.BICUBIC).tobytes()
            dhash = sum((gray[y*9+x] > gray[y*9+x+1]) << (y*8+x) for y in range(8) for x in range(8))
            hashes.append((split, row["image_id"], dhash))
            label = row["label"]
            if label["LabelName"] != MID or label["Source"] not in ("verification", "crowdsource-verification"):
                raise ValueError("Missing human-verified leaf-class label")
            if label["Confidence"] == "1" and row["boxes"]:
                positive += 1
            elif label["Confidence"] == "0" and not row["boxes"]:
                negative += 1
            else:
                raise ValueError("Inconsistent positive/negative annotation")
            attribution = row["attribution"]
            if (attribution["License"] != "https://creativecommons.org/licenses/by/2.0/"
                    or not attribution["Author"] or not attribution["OriginalLandingURL"]):
                raise ValueError("Missing official license metadata")
            for box in row["boxes"]:
                x1, y1, x2, y2 = box["xyxy"]
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
                        and box["class_id"] == 0 and box["category_id"] == 1):
                    raise ValueError("Invalid target box")
            if any(b["IsGroupOf"] != "0" or b["IsDepiction"] != "0" for b in row["boxes_raw"]):
                raise ValueError("Unresolved group-of/depiction annotation")
            box_count += len(row["boxes"])
        counts[split] = {"images": len(records), "positive": positive,
                         "negative": negative, "boxes": box_count}
    if counts != MANIFEST_BASE["audit"]["counts"]:
        raise ValueError("Unexpected split/box counts")
    for first, second in itertools.combinations(hashes, 2):
        if first[0] != second[0] and (first[2] ^ second[2]).bit_count() <= 4:
            raise ValueError("Cross-split near-duplicate candidate requires review")
    return counts


def verify_cache(root):
    path = root / "manifest.json"
    if path.is_symlink() or path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Unsafe manifest cache")
    data = path.read_bytes()
    if sha256(data) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Manifest cache SHA-256 mismatch")
    manifest = json.loads(data)
    return validate(root, manifest)


def build(root):
    # Fail closed: no mirrors, replacement images, synthetic fallback or live test tuning.
    for name, source in MANIFEST_BASE["class_source_integrity"].items():
        data = read_bounded(source["url"], 1024 * 1024)
        if sha256(data) != source["sha256"]:
            raise ValueError("Official class/hierarchy checksum mismatch")
        (root / name).write_bytes(data)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {split: executor.submit(candidates, split, count)
                   for split, count in (("train", 128), ("validation", 32), ("test", 32))}
        pools = {split: future.result() for split, future in futures.items()}
    selected = {}
    for split, pool in pools.items():
        positive_count, negative_count = MANIFEST_BASE["selection"][split]
        ids = {row["image_id"] for confidence, count in (("1", positive_count), ("0", negative_count))
               for row in [item for item in pool if item["label"]["Confidence"] == confidence][:count]}
        selected[split] = [row for row in pool if row["image_id"] in ids]
    manifest = dict(MANIFEST_BASE)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        manifest["splits"] = {split: list(executor.map(lambda row: materialize(root, row), records))
                              for split, records in selected.items()}
    validate(root, manifest)
    data = canonical(manifest)
    if sha256(data) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Rebuilt subset differs from reviewed manifest; no manifest published")
    temporary = root / "manifest.json.tmp"
    temporary.write_bytes(data)
    temporary.replace(root / "manifest.json")
    (root / "manifest.sha256").write_text(EXPECTED_MANIFEST_SHA256 + "  manifest.json\n")
    return validate(root, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(os.environ.get("MP_HOME", ".runtime")) / "datasets" / "openimages-cat-reviewed-v1")
    parser.add_argument("--verify-only", action="store_true", help="Verify every cached image without network access")
    parser.add_argument("--rebuild", action="store_true", help="Re-read pinned official metadata; verify/reuse existing image bytes")
    args = parser.parse_args()
    root = args.output
    if root.is_symlink():
        raise ValueError("Dataset output must not be a symlink")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if args.verify_only or ((root / "manifest.json").exists() and not args.rebuild):
        counts = verify_cache(root)
    else:
        # An existing manifest is verified before any rebuild, so tampering is not repaired silently.
        if (root / "manifest.json").exists():
            verify_cache(root)
        counts = build(root)
    print(json.dumps({"manifest_sha256": EXPECTED_MANIFEST_SHA256, "counts": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
