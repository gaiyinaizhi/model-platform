"""Bounded, inert YOLO detection ZIP validation. No paths or code are executed."""
import hashlib
import io
import math
import re
import stat
import warnings
import zipfile
import zlib
from pathlib import PurePosixPath

from .common import Problem, digest, encode, object_fields, text_value

MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_UNPACKED_BYTES = 128 * 1024 * 1024
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_ENTRIES = 2048
MAX_IMAGES = 1000
MAX_PIXELS = 20_000_000
MAX_TOTAL_PIXELS = 200_000_000
MAX_ISSUES = 200
MAX_BOXES = 100_000
SPLITS = ("train", "val", "test")
FORMAT = "yolo-detection/v1"
MAX_SOURCE_BYTES = 128 * 1024
MAX_SOURCE_IMAGES = 64
MAX_SOURCE_ANNOTATIONS = 10_000


def dependencies():
    try:
        from PIL import Image
        import yaml
    except ImportError:
        raise Problem(503, "dataset_dependencies_missing",
                      "Dataset import requires the optional Pillow and PyYAML dependencies") from None
    return Image, yaml


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def safe_path(name):
    try:
        encoded_length = len(name.encode("utf-8"))
    except UnicodeEncodeError:
        return False
    if (not name or encoded_length > 240 or "\\" in name or ":" in name
            or any(ord(c) < 32 or ord(c) == 127 for c in name)
            or name.startswith("/") or any(p in ("", ".", "..") for p in name.rstrip("/").split("/"))):
        return False
    return True


def validate_source(source):
    """Validate a declaration only; never fetch or certify the declared source.

    This v1 source schema is deliberately the exact reviewed COCO subset format.
    Its source IDs are not pycocotools' temporary contiguous category IDs.
    """
    object_fields(source, ("name", "version", "annotation_sha256", "selection_sha256",
                           "selection_policy", "manifest"))

    def string(value, name, maximum):
        text_value(value, name, maximum)
        if value != value.strip() or not value.strip() or any(ord(c) == 127 for c in value):
            raise Problem(400, "invalid_input", "Invalid dataset source text")

    def integer(value):
        if type(value) is not int or not 0 <= value <= 2**53 - 1:
            raise Problem(400, "invalid_input", "Source IDs and counts must be nonnegative JSON-safe integers")

    def hash_value(value):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise Problem(400, "invalid_input", "Source digests must be lowercase SHA-256 hex")

    for key, maximum in (("name", 80), ("version", 80), ("selection_policy", 160)):
        string(source[key], key, maximum)
    for key in ("annotation_sha256", "selection_sha256"):
        hash_value(source[key])
    manifest = source["manifest"]
    object_fields(manifest, ("schema", "source", "selection", "categories", "licenses", "images"))
    try:
        if len(encode(manifest)) > MAX_SOURCE_BYTES or digest(manifest) != source["selection_sha256"]:
            raise ValueError("Source manifest size or digest mismatch")
    except (ValueError, TypeError, RecursionError):
        raise Problem(400, "invalid_input", "Source manifest must match selection_sha256 and fit 128 KiB") from None
    if manifest["schema"] != "coco-subset-source/v1":
        raise Problem(400, "invalid_input", "Unsupported source manifest schema")
    object_fields(manifest["source"], ("name", "version", "annotation_sha256"))
    if (source["name"] != "COCO" or source["version"] != "val2017"
            or manifest["source"] != {key: source[key] for key in ("name", "version", "annotation_sha256")}):
        raise Problem(400, "invalid_input", "Source manifest must agree with the COCO val2017 declaration")
    selection = manifest["selection"]
    object_fields(selection, ("policy", "seed", "ordered_image_ids", "excluded_counts"))
    string(selection["policy"], "source selection policy", 160)
    string(selection["seed"], "source selection seed", 160)
    if selection["policy"] != source["selection_policy"]:
        raise Problem(400, "invalid_input", "Source selection policy mismatch")
    ordered_ids = selection["ordered_image_ids"]
    if not isinstance(ordered_ids, list) or not 1 <= len(ordered_ids) <= MAX_SOURCE_IMAGES:
        raise Problem(400, "invalid_input", "Source selection requires 1..64 images")
    for value in ordered_ids:
        integer(value)
    if len(set(ordered_ids)) != len(ordered_ids):
        raise Problem(400, "invalid_input", "Source image IDs must be unique")
    excluded = selection["excluded_counts"]
    if not isinstance(excluded, dict) or len(excluded) > 32:
        raise Problem(400, "invalid_input", "Source permits at most 32 exclusion reasons")
    for reason, count in excluded.items():
        string(reason, "source exclusion reason", 80)
        integer(count)

    from .pretrained_yolox import CLASS_NAMES, COCO_CATEGORY_IDS
    categories = manifest["categories"]
    if not isinstance(categories, list) or len(categories) != 80:
        raise Problem(400, "invalid_input", "Source requires the complete 80-class COCO mapping")
    for class_id, (category, name, source_id) in enumerate(zip(categories, CLASS_NAMES, COCO_CATEGORY_IDS)):
        object_fields(category, ("class_id", "source_category_id", "name"))
        integer(category["class_id"])
        integer(category["source_category_id"])
        if category != {"class_id": class_id, "source_category_id": source_id, "name": name}:
            raise Problem(400, "invalid_input", "Source category mapping differs from the reviewed full COCO mapping")
    licenses = manifest["licenses"]
    if not isinstance(licenses, list) or not 1 <= len(licenses) <= 16:
        raise Problem(400, "invalid_input", "Source requires 1..16 license declarations")
    license_ids = set()
    for license in licenses:
        object_fields(license, ("license_id", "name"))
        integer(license["license_id"])
        string(license["name"], "source license name", 160)
        if license["license_id"] in license_ids:
            raise Problem(400, "invalid_input", "Source license IDs must be unique")
        license_ids.add(license["license_id"])
    images = manifest["images"]
    if not isinstance(images, list) or len(images) != len(ordered_ids):
        raise Problem(400, "invalid_input", "Source image records must match the selection")
    names, annotation_ids = set(), set()
    for image, image_id in zip(images, ordered_ids):
        object_fields(image, ("image_id", "file_name", "width", "height", "image_sha256",
                              "pixel_sha256", "license_id", "annotation_ids"))
        for key in ("image_id", "width", "height", "license_id"):
            integer(image[key])
        string(image["file_name"], "source image filename", 160)
        if (not safe_path(image["file_name"]) or image["file_name"].endswith("/")
                or image["file_name"].casefold() in names or image["image_id"] != image_id
                or not 1 <= image["width"] * image["height"] <= MAX_PIXELS
                or image["license_id"] not in license_ids):
            raise Problem(400, "invalid_input", "Invalid source image identity, dimensions or license")
        names.add(image["file_name"].casefold())
        hash_value(image["image_sha256"])
        hash_value(image["pixel_sha256"])
        ids = image["annotation_ids"]
        if not isinstance(ids, list) or len(ids) > MAX_SOURCE_ANNOTATIONS:
            raise Problem(400, "invalid_input", "Source annotation IDs exceed the 10000 annotation budget")
        for annotation_id in ids:
            integer(annotation_id)
            if annotation_id in annotation_ids:
                raise Problem(400, "invalid_input", "Source annotation IDs must be globally unique")
            annotation_ids.add(annotation_id)
        if len(annotation_ids) > MAX_SOURCE_ANNOTATIONS:
            raise Problem(400, "invalid_input", "Source annotation IDs exceed the 10000 annotation budget")
    return source


def verify_source_samples(source, samples, classes):
    """Bind the full inert selection declaration to actual validated ZIP content."""
    if source is None:
        return
    validate_source(source)
    manifest = source["manifest"]
    if ([row["name"] for row in classes] != [row["name"] for row in manifest["categories"]]
            or len(samples) != len(manifest["images"])):
        raise ValueError("Source classes or image count mismatch")
    for sample, image in zip(samples, manifest["images"]):
        if (sample["split"] != "val" or sample["image"] != "images/val/" + image["file_name"]
                or sample["boxes"] != len(image["annotation_ids"])
                or any(sample[key] != image[key] for key in ("width", "height", "image_sha256", "pixel_sha256"))):
            raise ValueError("Source image, pixel or annotation count mismatch")


class Validation:
    def __init__(self, negative_policy, purpose="training", source=None):
        if purpose not in ("training", "evaluation") or (source is not None and purpose != "evaluation"):
            raise Problem(400, "invalid_input", "Invalid dataset purpose or source")
        if source is not None:
            validate_source(source)
        self.report = {"valid": False, "errors": [], "warnings": [], "errors_truncated": False,
                       "purpose": purpose, "source": source,
                       "counts": {"images": 0, "labels": 0, "boxes": 0, "negative_images": 0},
                       "classes": [], "splits": {s: {"images": 0, "boxes": 0, "negative_images": 0}
                                                   for s in SPLITS}, "content_sha256": None}
        self.policy = negative_policy
        self.purpose, self.source = purpose, source
        self.manifest = None
        self.files = {}

    def error(self, code, path, message, line=None):
        if len(self.report["errors"]) >= MAX_ISSUES:
            self.report["errors_truncated"] = True
            return
        # Paths originate in untrusted archives; keep messages bounded and controls invisible.
        path = "".join(c if ord(c) >= 32 and ord(c) != 127 else "?" for c in path)[:240]
        issue = {"code": code, "path": path, "message": message}
        if line is not None:
            issue["line"] = line
        self.report["errors"].append(issue)

    def unpack(self, raw):
        if not raw or len(raw) > MAX_ARCHIVE_BYTES:
            self.error("archive_size", "", "ZIP must be nonempty and at most 32 MiB")
            return False
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                entries = archive.infolist()
                if len(entries) > MAX_ENTRIES:
                    self.error("archive_entries", "", "ZIP exceeds 2048 entries")
                    return False
                names, files, total = set(), [], 0
                for entry in entries:
                    name = entry.orig_filename
                    canonical = name.rstrip("/")
                    mode = entry.external_attr >> 16
                    kind = stat.S_IFMT(mode)
                    if not safe_path(name) or name != entry.filename:
                        self.error("unsafe_path", name, "Unsafe or ambiguous ZIP path")
                    elif canonical.casefold() in names:
                        self.error("duplicate_path", name, "Duplicate or case-colliding ZIP path")
                    elif kind not in (0, stat.S_IFREG, stat.S_IFDIR) or (kind == stat.S_IFDIR) != entry.is_dir() and kind != 0:
                        self.error("unsafe_entry", name, "Only regular files and directories are allowed")
                    elif entry.flag_bits & 1 or entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                        self.error("unsupported_archive", name, "Encrypted or unsupported ZIP compression")
                    elif entry.file_size > MAX_FILE_BYTES or entry.file_size < 0:
                        self.error("file_size", name, "File exceeds 8 MiB")
                    elif entry.file_size > max(1, entry.compress_size) * 200:
                        self.error("compression_ratio", name, "ZIP entry exceeds compression ratio limit (200:1)")
                    else:
                        names.add(canonical.casefold())
                        if not entry.is_dir():
                            files.append(entry)
                            total += entry.file_size
                file_names = {x.filename.casefold() for x in files}
                for entry in files:
                    if any(p.as_posix().casefold() in file_names for p in PurePosixPath(entry.filename).parents if str(p) != "."):
                        self.error("path_conflict", entry.filename, "File conflicts with a parent directory")
                if total > MAX_UNPACKED_BYTES:
                    self.error("archive_size", "", "ZIP expands beyond 128 MiB")
                if self.report["errors"]:
                    return False
                actual = 0
                for entry in files:
                    with archive.open(entry) as stream:
                        data = stream.read(MAX_FILE_BYTES + 1)
                    actual += len(data)
                    if len(data) != entry.file_size or len(data) > MAX_FILE_BYTES or actual > MAX_UNPACKED_BYTES:
                        self.error("archive_size", entry.filename, "Actual decompressed bytes exceed the declared bounds")
                        return False
                    self.files[entry.filename] = data
        except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, EOFError, ValueError, RuntimeError, zlib.error):
            self.error("invalid_zip", "", "Cannot read a complete, valid ZIP archive")
            return False
        return True

    def metadata(self, yaml):
        configs = [p for p in self.files if p == "data.yaml" or p.endswith("/data.yaml")]
        if len(configs) != 1:
            self.error("data_yaml", "data.yaml", "ZIP must contain exactly one data.yaml")
            return None
        config = configs[0]
        prefix = config[:-len("data.yaml")]
        if prefix.count("/") > 1 or any(not p.startswith(prefix) for p in self.files):
            self.error("dataset_root", config, "Dataset must be at ZIP root or inside one enclosing directory")
            return None
        self.files = {p[len(prefix):]: b for p, b in self.files.items()}
        raw = self.files["data.yaml"]
        if len(raw) > 65536:
            self.error("data_yaml", "data.yaml", "data.yaml exceeds 64 KiB")
            return None
        class StrictLoader(yaml.SafeLoader):
            pass
        def mapping(loader, node):
            result = {}
            for key_node, value_node in node.value:
                key = loader.construct_object(key_node)
                if not isinstance(key, (str, int)) or isinstance(key, bool) or key in result:
                    raise ValueError("Duplicate or unsupported YAML key")
                result[key] = loader.construct_object(value_node)
            return result
        StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
        try:
            text = raw.decode("utf-8")
            tokens = list(yaml.scan(text))
            if len(tokens) > 10000 or any(isinstance(t, (yaml.tokens.AliasToken, yaml.tokens.AnchorToken, yaml.tokens.TagToken)) for t in tokens):
                raise ValueError("YAML aliases, anchors, tags or excessive tokens")
            depth = 0
            for token in tokens:
                if isinstance(token, (yaml.tokens.BlockMappingStartToken, yaml.tokens.BlockSequenceStartToken, yaml.tokens.FlowMappingStartToken, yaml.tokens.FlowSequenceStartToken)):
                    depth += 1
                    if depth > 10:
                        raise ValueError("YAML nesting")
                elif isinstance(token, (yaml.tokens.BlockEndToken, yaml.tokens.FlowMappingEndToken, yaml.tokens.FlowSequenceEndToken)):
                    depth -= 1
            meta = yaml.load(text, Loader=StrictLoader)
            if not isinstance(meta, dict) or set(meta) - {"names", "nc", "path", "train", "val", "test"}:
                raise ValueError("Unsupported YAML keys")
            if meta.get("path", ".") not in (".", ""):
                raise ValueError("path must be relative dataset root")
            if self.purpose == "evaluation" and any(split in meta for split in ("train", "test")):
                raise ValueError("Evaluation datasets permit only val")
            required_splits = ("val",) if self.purpose == "evaluation" else ("train", "val")
            for split in SPLITS:
                if split in required_splits or split in meta:
                    if meta.get(split) not in (f"images/{split}", f"./images/{split}"):
                        raise ValueError("Splits must use canonical local image directories")
            names = meta.get("names")
            if isinstance(names, dict):
                if set(names) != set(range(len(names))) or any(type(k) is not int for k in names):
                    raise ValueError("Class mapping must be contiguous integers")
                names = [names[i] for i in range(len(names))]
            if (not isinstance(names, list) or not 1 <= len(names) <= 100
                    or any(not isinstance(n, str) or not n.strip() or len(n) > 80
                           or any(ord(c) < 32 or ord(c) == 127 for c in n) for n in names)
                    or len(set(names)) != len(names)):
                raise ValueError("Invalid class names")
            if "nc" in meta and (type(meta["nc"]) is not int or meta["nc"] != len(names)):
                raise ValueError("nc does not match names")
        except (ValueError, UnicodeDecodeError, yaml.YAMLError, RecursionError, TypeError):
            split_help = "only local images/val" if self.purpose == "evaluation" else "local images/train, images/val, optional images/test"
            self.error("data_yaml", "data.yaml", f"Use unique names (1..100), matching nc and {split_help}; no tags, aliases, downloads or other keys")
            return None
        self.report["classes"] = [{"id": i, "name": name, "boxes": 0, "images": 0} for i, name in enumerate(names)]
        return meta

    def validate(self, raw):
        Image, yaml = dependencies()
        if not self.unpack(raw):
            return self.report, None, {}
        meta = self.metadata(yaml)
        if meta is None:
            return self.report, None, {}
        images, labels = {}, {}
        for path in sorted(self.files):
            if path == "data.yaml":
                continue
            pieces = path.split("/")
            if len(pieces) < 3 or pieces[1] not in SPLITS:
                self.error("unexpected_file", path, "Only data.yaml and canonical images/labels split directories are supported")
                continue
            split, tail = pieces[1], "/".join(pieces[2:])
            if self.purpose == "evaluation" and split != "val":
                self.error("evaluation_split", path, "Evaluation datasets permit only val images and labels")
                continue
            suffix = PurePosixPath(tail).suffix.lower()
            stem = str(PurePosixPath(tail).with_suffix(""))
            key = (split, stem.casefold())
            if pieces[0] == "images" and suffix in (".jpg", ".jpeg", ".png"):
                if key in images:
                    self.error("ambiguous_image", path, "Multiple images share the same label stem")
                images[key] = path
            elif pieces[0] == "labels" and suffix == ".txt":
                labels[key] = path
            else:
                self.error("unexpected_file", path, "Only JPEG/PNG images and .txt labels are supported")
        if len(images) > MAX_IMAGES:
            self.error("image_limit", "images", "At most 1000 images are supported")
            return self.report, None, {}
        self.report["counts"]["images"] = len(images)
        self.report["counts"]["labels"] = len(labels)
        for split in SPLITS:
            required_splits = ("val",) if self.purpose == "evaluation" else ("train", "val")
            if (split in required_splits or split in meta) and not any(k[0] == split for k in images):
                self.error("empty_split", f"images/{split}", "Declared train, val or test split must have images")
            if split not in meta and any(k[0] == split for k in images):
                self.error("undeclared_split", f"images/{split}", "Image split must be declared in data.yaml")
        for key, path in labels.items():
            if key not in images:
                self.error("orphan_label", path, "Label has no matching image")
        samples, seen_hashes, seen_pixels, total_pixels = [], {}, {}, 0
        for key, path in sorted(images.items()):
            split, _ = key
            stats = self.report["splits"][split]
            stats["images"] += 1
            raw_image = self.files[path]
            image_hash = sha256(raw_image)
            width = height = 0
            pixel_hash = None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(io.BytesIO(raw_image)) as im:
                        width, height = im.size
                        if (im.format not in ("JPEG", "PNG") or width * height > MAX_PIXELS
                                or width < 1 or height < 1 or getattr(im, "n_frames", 1) != 1):
                            raise ValueError("Unsupported image")
                        if (PurePosixPath(path).suffix.lower() == ".png") != (im.format == "PNG"):
                            raise ValueError("Extension mismatch")
                        total_pixels += width * height
                        if total_pixels > MAX_TOTAL_PIXELS:
                            self.error("pixel_limit", path, "Dataset exceeds 200 million decoded pixels")
                            break
                        im.verify()
                    with Image.open(io.BytesIO(raw_image)) as im:
                        im.load()
                        rgb = im.convert("RGB")
                        pixel_hash = sha256(f"{width}x{height}:".encode() + rgb.tobytes())
                for seen, value in ((seen_hashes, image_hash), (seen_pixels, pixel_hash)):
                    previous = seen.get(value)
                    if previous and previous[0] != split:
                        self.error("split_overlap", path, f"Image content also appears in {previous[1]}")
                        break
                seen_hashes.setdefault(image_hash, (split, path))
                seen_pixels.setdefault(pixel_hash, (split, path))
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning):
                self.error("invalid_image", path, "Cannot decode a single-frame JPEG/PNG within the 20 million pixel limit")
            label = labels.get(key)
            boxes, class_set = 0, set()
            if label is None:
                self.error("missing_label", path, "Every image needs a matching label file, even for an explicit negative")
            else:
                try:
                    text = self.files[label].decode("utf-8")
                    if len(self.files[label]) > 1024 * 1024:
                        raise ValueError("Label too large")
                    lines = text.splitlines()
                    if len(lines) > 10000:
                        raise ValueError("Too many lines")
                    for line_no, line in enumerate(lines, 1):
                        if not line.strip():
                            continue
                        parts = line.split()
                        if len(parts) != 5 or not re.fullmatch(r"0|[1-9][0-9]*", parts[0]):
                            self.error("invalid_label", label, "Expected class_id x_center y_center width height", line_no)
                            continue
                        if len(parts[0]) > 3 or not 0 <= int(parts[0]) < len(self.report["classes"]):
                            self.error("class_id", label, "Class ID is outside data.yaml names", line_no)
                            continue
                        try:
                            x, y, w, h = map(float, parts[1:])
                            if (not all(math.isfinite(v) for v in (x, y, w, h)) or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1)
                                    or x - w / 2 < -1e-9 or y - h / 2 < -1e-9 or x + w / 2 > 1 + 1e-9 or y + h / 2 > 1 + 1e-9):
                                raise ValueError("Invalid box")
                        except ValueError:
                            self.error("invalid_box", label, "Box must be finite, normalized, positive and entirely inside the image", line_no)
                            continue
                        cid = int(parts[0])
                        boxes += 1
                        class_set.add(cid)
                        self.report["classes"][cid]["boxes"] += 1
                    if not text.strip():
                        if self.policy != "allow_empty":
                            self.error("negative_not_allowed", label, "Empty labels require the allow_empty negative-image policy")
                        self.report["counts"]["negative_images"] += 1
                        stats["negative_images"] += 1
                except (UnicodeDecodeError, ValueError):
                    self.error("invalid_label", label, "Label must be UTF-8, at most 1 MiB and 10000 lines")
            for cid in class_set:
                self.report["classes"][cid]["images"] += 1
            stats["boxes"] += boxes
            self.report["counts"]["boxes"] += boxes
            if self.report["counts"]["boxes"] > MAX_BOXES:
                self.error("box_limit", "labels", "Dataset exceeds 100000 bounding boxes")
                break
            sample = {"image": path, "label": label, "split": split, "width": width, "height": height,
                      "boxes": boxes, "image_sha256": image_hash,
                      "label_sha256": sha256(self.files[label]) if label else None}
            if self.purpose == "evaluation":
                sample["pixel_sha256"] = pixel_hash
            samples.append(sample)
        if not self.report["errors"] and self.source is not None:
            try:
                verify_source_samples(self.source, samples, self.report["classes"])
            except (Problem, ValueError, KeyError, TypeError):
                self.error("source_mismatch", "source.manifest", "Source selection must match all classes, image order, dimensions, image/RGB hashes and annotation counts")
        if not self.report["errors"]:
            manifest = {"format": FORMAT, "negative_policy": self.policy,
                        "purpose": self.purpose, "source": self.source,
                        "counts": self.report["counts"], "classes": self.report["classes"], "splits": self.report["splits"],
                        "files": [{"path": p, "bytes": len(b), "sha256": sha256(b)} for p, b in sorted(self.files.items())],
                        "samples": samples}
            self.report["valid"] = True
            self.report["content_sha256"] = digest(manifest)
            self.manifest = manifest
        return self.report, self.manifest, self.files if self.manifest else {}
