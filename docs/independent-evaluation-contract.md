# Independent imported-model evaluation contract (review draft)

2026-10-18. **Independently reviewed and approved on 2026-10-18 before dependent implementation. Implementation and actual acceptance remain separately evidenced.**
Baseline backend: `8944e51b8d6e9a66b012105af8667127b7a34a7b`. This adds an evaluation-only path for the existing reviewed imported YOLOX-Nano COCO asset. It does not train, publish a new model, validate weight licensing, establish production quality, or run the official COCO benchmark.

## 1. Authority and compatibility

- An independent evaluation belongs to the Bearer-authenticated tenant and references one existing imported `model_version_id` and one published immutable `dataset_version_id`. Callers cannot choose a tenant, local file path, remote URL, model bytes, preprocessing code, or threshold.
- `ModelVersion`, its manifest, `origin`, `import_id`, `validation_id`, original `evaluation:null` and `local_evaluation_status:not_evaluated` remain byte-for-byte unchanged. They describe registration-time facts. No synthetic TrainingRun/attempt or registration-time evaluation is created. The UI separately shows later independent evaluation records.
- Existing trained-model evaluation, recipes, train+val validation, deployment identity, serving protocol, and delivery restrictions remain unchanged. No credentials are created and no external resource receives data.

## 2. Evaluation-only dataset import

The existing staged import/upload/explicit-publication flow remains authoritative. `POST /dataset-imports` adds optional `purpose`, exactly `training` or `evaluation`; omission preserves `training`. Existing name, rights confirmation and negative-image policy remain required.

- `training`: still requires nonempty `images/train` and `images/val`, and optional declared nonempty test. Existing metadata, labels, duplicate-content checks, class limits, byte/pixel/box budgets and unsafe-archive rejection remain unchanged.
- `evaluation`: requires one nonempty declared `val` split, forbids train/test declarations and files, permits val-only `data.yaml`, and uses the same per-file annotation/image integrity and explicit negative-image policy. It cannot be submitted to `yolox-nano-dataset/v1` training.
- The persisted import, report, published version and manifest expose `purpose`. Legacy rows and manifests lacking it read as `training`. The purpose is immutable from import creation through version publication. Hash the original submitted body for idempotency without inserting default fields; omitted-purpose legacy requests retain their original hash and replay behavior. New evaluation sample entries additionally bind `pixel_sha256=SHA256(ASCII(width)+"x"+ASCII(height)+":"+decoded_RGB_bytes)`; do not rewrite legacy manifests.
- Optional `source` on evaluation import is an inert bounded declaration with exactly `name`, `version`, `annotation_sha256`, `selection_sha256`, `selection_policy`, `manifest`. Name/version/policy are bounded strings (80/80/160 characters), digests are 64 lowercase hex. `manifest` is the full bounded source-selection object defined in section 9, max128 KiB; its canonical digest must equal `selection_sha256`, and its source/hash/policy fields must agree with the outer declaration. It is recorded in the immutable manifest and does not trigger fetching or independent source certification. Omitted source is null. Training requests do not accept source in this increment.
- Dataset label IDs remain contiguous 0..K-1. Evaluation submission requires exactly the imported model's 80 ordered class names, with no aliasing or partial-class projection. The evaluation binding stores the model's complete mapping `{class_id, name, source_category_id}` and its canonical hash. Here `source_category_id` is the real sparse COCO ID (cat=17, internal class=15); pycocotools' temporary `evaluator_internal_category_id=class_id+1` IDs are explicitly distinguished from original COCO IDs. Verify the exact all80 bijection for both GT and predictions; a bijective internal relabeling preserves metrics.
- Existing dataset limits remain 32 MiB ZIP, 128 MiB unpacked, 8 MiB/file, 2,048 entries, 1,000 images, 20 million pixels/image, 200 million pixels/import and 100,000 boxes. Independent evaluation has a tighter 1..64 images and 10,000 boxes limit; the actual acceptance run uses exactly 64.
- Safe additive schema migration may add immutable-purpose/source columns with legacy training defaults. No existing record or manifest is rewritten. Publication must still copy, verify and bind the complete snapshot.

Example evaluation metadata:

```json
{"name":"coco-val2017-fixed64","rights_confirmed":true,"negative_policy":"allow_empty","purpose":"evaluation","source":{"name":"COCO","version":"val2017","annotation_sha256":"<64 lowercase hex>","selection_sha256":"<64 lowercase hex>","selection_policy":"sha256-seed-first64-noncrowd-ccby-bysa/v1","manifest":{"...":"section 9 exact schema"}}}
```

The ZIP's `data.yaml` contains `val: images/val`, the full ordered 80 `names`, optional matching `nc`, and no train/test. Every image has a label file; empty labels require `allow_empty`.

## 3. Public evaluation API

All routes use existing Bearer auth, strict JSON parsing, 2 MiB request limit, connection/rate budgets and 404 tenant isolation.

| Route | Contract |
| --- | --- |
| `POST /model-evaluations` | Exactly one `Idempotency-Key`; body below. First submission 202, identical tenant/key/body replay 200 with same ID, differing body 409 `idempotency_conflict`. Validates tenant ownership, model identity and dataset compatibility before queueing. |
| `GET /model-evaluations` | Latest 100 tenant-owned summaries, newest first. Optional `model_version_id` filter is tenant-checked; unknown/foreign model 404. Returns `{evaluations:[...],limit:100}`. No pagination in this bounded first increment. Global record capacity is 256, retained terminal records are never silently evicted; the list is a documented latest100 window, not complete history. |
| `GET /model-evaluations/{id}` | Tenant-owned detail including immutable binding, state/timestamps and terminal result or bounded error code. Unknown/foreign evaluation 404. |

```json
{"model_version_id":"model_...","dataset_version_id":"dsv_...","split":"val","protocol":"imported-coco-serving-evaluation/v1"}
```

Unknown fields, user-supplied thresholds, model hash overrides, category overrides and missing idempotency keys are rejected. Supported models are registered `origin=imported`, `yolox-nano-coco-onnx/v1`, the exact reviewed artifact identity; trained models/other protocols return 409 `evaluation_model_unsupported`. Missing prerequisites return 503 `evaluation_dependencies_missing`; invalid dataset purpose, split, class ordering or evaluation budget returns 409 `evaluation_dataset_incompatible` (budget exhaustion in global runtime remains 429).

Summary/detail common fields:

```json
{"id":"mev_...","status":"queued","created_at":0,"started_at":null,"finished_at":null,"model_version_id":"model_...","dataset_version_id":"dsv_...","split":"val","protocol":"imported-coco-serving-evaluation/v1","request_sha256":"...","result":null,"error_code":null}
```

Detail additionally includes `binding` (section 4). Terminal successful `result` includes `metrics`, `operating_point`, `wrong_examples`, `digests`, `environment`, `warnings`, and `scope`. Raw image bytes, model paths, credentials and unbounded raw exceptions are never returned. No cancel/retry/delete/update API in this increment. A failed record remains terminal; an explicitly requested rerun uses a new key/new evaluation ID.

## 4. Immutable binding and transaction/state semantics

Persist a distinct `model_evaluations` identity/queue table and immutable terminal result table; do not reuse the existing train-attempt-bound `evaluations` table. Database triggers prohibit identity/binding update/delete, prohibit terminal mutation, and restrict state transitions to `queued -> running -> succeeded|failed` (a preexecution validation failure may terminate a claimed job as failed). All references are tenant-bound, checked at submission, execution, commit and read; composite constraints or equivalent explicit immutable-identity validation prevent same-tenant and cross-tenant mix-and-match.

The immutable binding contains:

- model version, origin, import/validation IDs, catalog/revision, artifact SHA-256/bytes and original manifest SHA-256
- dataset version/content SHA-256, purpose, split, sample and label counts, sorted sample identities and per-file hashes, source declaration, exact ordered category map and SHA-256
- fixed inference/evaluation protocol and SHA-256, submission code/source SHA-256 and environment snapshot (Python/platform plus pinned installed NumPy/ONNX/ORT/OpenCV/Pillow/pycocotools versions stored as canonical JSON in the existing dependencies string)

Submission atomically binds IDs, canonical request hash and snapshot identities before returning queued. A single worker claims one job transactionally. It recomputes and compares all bindings, current source hash and loaded worker source hash before inference and before successful commit. Code/source changes fail closed as `execution_code_changed`; a changed Python/platform/dependency fingerprint fails as `execution_environment_changed`. Missing/tampered model, dataset, binding, result or prediction evidence is an integrity failure; GET must not present a corrupted terminal result as successful evidence.

Success commits result, result hash, prediction digest and terminal status in one transaction. Failure stores only a bounded classified error code and terminal timestamp. No partial metrics are published as success. Duplicate delivery/claim cannot reexecute or create a second result. The same current Worker process consumes training and evaluation queues sequentially in global creation order, sharing the existing 32 queued/running logical slots; training submit/retry and evaluation submit enforce the combined budget in transactions. One runtime has one Worker lock.

Worker restart marks any previously running evaluation failed/`evaluation_interrupted`; queued work remains queued, no automatic replay. API restart alone does not fail work still owned by a live Worker. Whole-job evaluation wall deadline is 300 s; evaluation subprocess dies/reaps on timeout and the record becomes failed/`evaluation_timeout`. Per-image imported inference retains 3 s deadline, CPUExecutionProvider and single-thread settings; the child has 2 GiB address-space and 240 s CPU limit. Use a bounded result channel/file (8 MiB), at most 6,400 predictions, at most 20 wrong-example records, 256 total evaluation records and 64 MiB evaluation evidence storage. A worker crash must not leave a continuing orphan evaluator (parent-death monitoring). These are logical/process limits, not an OS multi-tenant sandbox or RSS/cgroup guarantee.

## 5. Fixed serving-equivalent measurement protocol

Every image is decoded to RGB; preserve aspect ratio while shrinking only when max(width,height)>384. Use OpenCV `INTER_LINEAR`, scale `384/max(width,height)`, and integer floored target dimensions clamped to at least 1. Record original and request dimensions and realized x/y factors, and SHA256(ASCII(width)+"x"+ASCII(height)+":"+RGB bytes) for both decoded original and actual resized request RGB. Each prediction-evidence image record binds both digests. Evaluate in original pixel coordinates: restore response xyxy using separate original/request x and y factors. Annotations remain original-image continuous coordinates. No image or annotation is selected/removed using predictions. This is unchanged server inference with a declared OpenCV caller-side resize; browser JPEG decoding and canvas high-quality resizing may produce different input pixels. No byte-exact browser-upload parity is claimed.

Then call the existing imported-model inference implementation unchanged:

1. max384 RGB input -> BGR, OpenCV INTER_LINEAR ideal-ratio letterbox at upper left into 416x416 with 114 padding, CHW float32 0..255
2. unchanged YOLOX decode and ideal letterbox ratio restoration
3. objectness × best-class score strictly >0.3; class-agnostic inclusive-pixel NMS at IoU 0.45
4. clip to request-image boundaries, drop degenerate boxes, cap globally at 100 detections/image

Record protocol constants and library versions, including upstream NumPy equal-score behavior. Do not tune scores/NMS, add evaluation-only inference behavior, filter class predictions, or claim equivalence to low-threshold official benchmarking. Canonical prediction evidence uses fixed sorted image order and preserves inference prediction order; store ordered original-coordinate detections and SHA-256.

## 6. Metrics and diagnostic examples

Reuse `detection_evaluation.evaluate_detections` and official pycocotools AP. Add AR@100 from accumulated recall tensor at area=all, maxDets=100, averaged over IoU .50:.05:.95 and classes with GT. AP50 and AP50:95 preserve their existing meaning. All 80 classes have rows; absent-GT class AP/AR is null and excluded from macro means. Empty detections with GT produce measured zero. Present full image, GT, prediction and represented-class counts.

Report an independent operating-point summary at score>0.3 and continuous-coordinate IoU>=0.5. For each image and class, sort detections by descending score with deterministic original prediction-index ties; greedily match each prediction to the unmatched GT of highest IoU>=0.5, using original annotation-index ties. Each GT and prediction can match once. Cross-class overlap never matches. Unmatched predictions count FP (including absent-GT classes), unmatched GT count FN. Per-class and total TP/FP/FN, precision and recall are recorded; undefined denominators are null. This operating-point recall is not AR@100, and the matching is not called COCO's AP matching.

Return at most 20 wrong-image examples, in stable dataset image order, each with public relative sample identity, TP/FP/FN counts, at most 20 GT boxes and 20 prediction boxes plus truncation flags and original row indices; boxes and scores are finite. Diagnostic type is only `false_positive` or `false_negative`, never an inferred causal diagnosis. No image bytes, arbitrary image URL or local path in this JSON response. The separately authorized optional sanitized preview is defined in [error-image-preview-contract.md](error-image-preview-contract.md); raw image bytes remain private. Record total wrong images and whether examples were truncated. Example selection never changes metric inputs. Full prediction digest covers every image including empty prediction lists.

## 7. Fixed 64-image COCO acceptance subset

Before any inference, prepare and persist a source/subset manifest from original val2017 annotations and image bytes. Filter to explicit CC BY/CC BY-SA image-license metadata and exclude entire images containing crowd/ignore/group annotations (the existing evaluator explicitly rejects these), malformed/out-of-bounds/duplicate boxes or unsupported dimensions/files; sort eligible images by SHA-256 of a fixed recorded seed plus integer COCO image ID, then select the first 64. The order key is SHA256(ASCII(seed+":"+zero_padded_12_digit_image_id)); order ascending digest then ascending numeric ID. Freeze the seed and tie-breaking rule in the selection manifest. Acquisition failures are reported and resolved before the final manifest; no inference-dependent substitutions. Freeze the exact ordered IDs, eligibility/exclusion reasons, archive/annotation hashes, per-image raw and decoded-RGB hashes, all original annotation IDs, all 80 class definitions and source-category mapping. The source manifest schema is `{schema:"coco-subset-source/v1",source:{name,version,annotation_sha256},selection:{policy,seed,ordered_image_ids,excluded_counts},categories:[{class_id,source_category_id,name}],images:[{image_id,file_name,width,height,image_sha256,pixel_sha256,license_id,annotation_ids}]}` plus the exact license metadata specified in section 9; no arbitrary source URL is accepted by the API. Its canonical `common.encode` compact sorted-key JSON (`ensure_ascii=True`, finite numbers) encoded UTF-8 SHA-256 is `selection_sha256`; keep complete original annotation bytes independently hashed. API treats the source claim as declared provenance while verifying the imported actual file/RGB/label content. Acquisition failures must be resolved before the selection manifest is frozen; no replacement after inference.

Retain every annotation of every class on each selected image; no cat-only projection and no annotation-level crowd removal. Use original JPEGs in the immutable dataset; explicit pre-request max384 conversion happens during evaluation and is traceable. Preserve lawful source and per-image license/attribution metadata in the evidence manifest; do not commit images or weight bytes. Rights remain the caller's declaration, not an independent legal certification.

Results must say: **64-image fixed non-crowd COCO val2017 subset, full 80-class mapping, serving operating point; not the official COCO benchmark, not independent external auditing, not device validation or production certification.** The imported model may have trained on COCO; this evaluates a declared validation subset and cannot certify absence of training overlap.

## 8. UI contract and acceptance

The UI places independent evaluation separately on imported model detail. It selects compatible evaluation-purpose dataset versions, submits the fixed protocol, polls state, then renders AP50/AP50:95/AR@100, distinct operating-point precision/recall/TP/FP/FN, all class rows, example count/truncation and limitations. Display the actual model/dataset hashes, source map, split and protocol; preserve registration-time unverified-license/not-evaluated provenance without treating it as the absence of later independent records. Failed/queued/running are never rendered as zero accuracy or success. BFF explicitly propagates new purpose/source/pixel/mapping fields through its allowlists, maps new classified error codes, forwards tenant session credentials through existing protected routes and exposes no new credential entry point. Detail response must remain <=512 KiB and list response <=256 KiB, below the existing 1 MiB BFF upstream limit; precommit serializer enforces bounds and truncates only already-defined diagnostics, never metric inputs. Large full prediction evidence stays private, with its digest returned. Frontend repository references to missing `ai/trellis/authority.md` and `ai/workflows/feature-lifecycle.md` are an unresolved documentation gap; no delivery-state authority is invented.

Acceptance includes legacy train+val regression; val-only purpose and training rejection; strict request/idempotency; all ownership/mapping/hash mismatch cases; duplicate claims; timeout; process interruption/restart; terminal mutation rejection; complete empty/missing/no-GT metrics; absent-class FP; duplicate/cross-class/tied-score one-to-one tests; bounded examples; protocol restoration fixtures; real reviewed-asset inference on the frozen64; actual UI submit/poll/result evidence. Report tests as passed/failed/skipped/blocked and distinguish synthetic regression from real subset measurements.


## 9. Exact wire schemas (v1)

Unless explicitly nullable, every field below is required. Times are finite Unix seconds. Counts/indices/dimensions are nonnegative integers (dimensions positive); hashes are lowercase SHA-256 hex. Metric values are finite [0,1] or null. Unknown public payload fields are not required for correctness and BFF explicitly projects these v1 fields. Private execution evidence is never forwarded wholesale.

`summary` has exactly the common fields shown in section 3. `result` is always null in list summaries (even on success); no `binding` appears in a list. Detail is `summary` plus `binding` and `result_sha256` (null until succeeded); `result` exists only for succeeded. Errors occur only for failed. All failed/queued/running results are null.

`binding`:

```text
{
 model: {model_version_id, origin:"imported", import_id, validation_id,
         catalog_id, catalog_revision, artifact_sha256, artifact_bytes, manifest_sha256},
 dataset: {dataset_version_id, content_sha256, purpose:"evaluation", split:"val",
           images:int, boxes:int, samples:[{image:str,label:str,width:int,height:int,
           image_sha256,label_sha256,pixel_sha256}], source:Source|null},
 categories: [{class_id:int,source_category_id:int,evaluator_internal_category_id:int,name:str}],
 category_mapping_sha256:sha256,
 protocol: {id:"imported-coco-serving-evaluation/v1", inference_protocol:"yolox-nano-coco-onnx/v1",
            max_input_side:384,input_size:416,score_threshold:0.3,score_operator:">",
            nms_iou_threshold:0.45,nms_class_agnostic:true,max_detections:100,
            resize:"opencv-inter-linear-floor-max384-no-upscale/v1",
            metric_protocol:"coco-style-bbox-subset/v1",matching_protocol:"class-aware-greedy-iou50/v1"},
 protocol_sha256:sha256, submission_environment:{python:str,platform:str,code_sha256:sha256,dependencies:str}
}
```

`result`:

```text
{
 metrics: {protocol:"coco-style-bbox-subset/v1",status:"evaluated"|"no_ground_truth",
           map_50:number|null,map_50_95:number|null,ar_100:number|null,
           evaluator:{implementation:"pycocotools.cocoeval.COCOeval",version:str},
           parameters:existing_evaluate_detections_parameters,
           counts:existing_evaluate_detections_counts,
           per_class:[{category_id:int,name:str,ground_truth_boxes:int,predictions:int,
                       status:"evaluated"|"no_ground_truth",ap_50:number|null,
                       ap_50_95:number|null,ar_100:number|null}]},
 operating_point: {protocol:"class-aware-greedy-iou50/v1",iou_threshold:0.5,
                   score_threshold:0.3,score_operator:">",
                   totals:{tp:int,fp:int,fn:int,precision:number|null,recall:number|null},
                   per_class:[{category_id:int,name:str,tp:int,fp:int,fn:int,
                               precision:number|null,recall:number|null}]},
 wrong_examples: {total:int,limit:20,truncated:bool,items:[
   {image_id:str,width:int,height:int,tp:int,fp:int,fn:int,
    false_positives:[{prediction_index:int,category_id:int,score:number,bbox:[number,number,number,number]}],
    false_negatives:[{annotation_index:int,category_id:int,bbox:[number,number,number,number]}],
    false_positives_truncated:bool,false_negatives_truncated:bool}]},
 digests: {binding_sha256,model_artifact_sha256,dataset_content_sha256,
           category_mapping_sha256,protocol_sha256,source_selection_sha256:sha256|null,
           prediction_sha256,execution_code_sha256},
 environment: {python:str,platform:str,code_sha256:sha256,dependencies:{numpy:str,
               onnxruntime:str,opencv:str,Pillow:str,pycocotools:str}},
 scope: {images:int,split:"val",official_benchmark:false,external_audit:false,
         production_allowed:false,device_certified:false},
 warnings:[str]
}
```

No per-image match arrays are in the public response. Wrong-example FP/FN arrays are each capped at20; examples are capped at20. `image_id` is the bounded relative sample identity (e.g. `images/val/000000000139.jpg`). Box coordinates are original-image continuous xyxy. The result hash is SHA256 of its entire canonical object, including matching outcomes; prediction digest is SHA256 of private ordered records `[{image_id,width,height,request_width,request_height,original_rgb_sha256,request_rgb_sha256,detections:[{category_id,score,bbox}]}]`. This evidence is immutable private server storage, not a path returned by API. A missing or mismatched evidence hash makes detail retrieval return409 `evaluation_integrity`.

`Source` is exactly the six fields described in section2. Its `manifest` is:

```text
{schema:"coco-subset-source/v1",
 source:{name:"COCO",version:"val2017",annotation_sha256:sha256},
 selection:{policy:str,seed:str,ordered_image_ids:[int],excluded_counts:{reason:int}},
 categories:[{class_id:int,source_category_id:int,name:str}],
 licenses:[{license_id:int,name:str}],
 images:[{image_id:int,file_name:str,width:int,height:int,image_sha256:sha256,
          pixel_sha256:sha256,license_id:int,annotation_ids:[int]}]}
```

At most64 images,80 categories,16 license entries,32 exclusion reasons (keys<=80 chars),10,000 annotation IDs total, filename<=160 chars and seed<=160 chars. IDs are unique and selected list equals image-record order; exact model category bijection and source hash/policy agreement are checked. Each image's license ID must be declared. Source annotation IDs must be globally unique; per-image annotation_ids count must equal validated label rows for that image. Full acquisition URLs, attribution and original license evidence remain separately retained acceptance evidence; the API does not claim a license audit. At ZIP validation, the declared image list must exactly match `images/val/<file_name>` and actual decoded dimensions/raw/RGB hashes. The actual label content remains bound separately by the dataset's label SHA-256. This cannot independently prove the declared annotations originated from COCO; source bytes/annotation hash and the full preparer manifest remain independently reviewable acceptance evidence. The full source object is persisted in immutable dataset manifest and copied into evaluation binding, available through authenticated detail. Selection changes require a new import/version/evaluation.

### Reviewed list-summary bounds

`GET /dataset-versions` returns `source:null` or the five-field summary `{name,version,annotation_sha256,selection_sha256,selection_policy}` without `manifest`. Authenticated dataset detail/import detail and independent evaluation binding retain the full six-field Source with manifest. This avoids multiplying up to128KiB manifests across100 list entries; full source evidence remains accessible through detail. All source integer identifiers/counts are bounded to JavaScript's exact safe-integer range (<=2^53-1), in addition to narrower field-specific limits.

## 10. Actual measurement and verification

The independently reviewed implementation was exercised twice on the same frozen64-image dataset, actual reviewed ONNX bytes and fixed serving protocol. The two final records have identical bindings, result objects and prediction bytes. Measured AP50=0.4146083434, AP50:95=0.2747798977, AR@100=0.2885501246. The subset contains457 GT boxes,273 predictions,62 images with GT and60 GT-present classes; all80 class definitions/rows remain present. At continuous IoU>=0.5 the class-aware one-to-one counts are TP190/FP83/FN267, precision0.695970696 and recall0.415754923. These are not a generic “accuracy” or a full COCO benchmark. There are55 images with an FP or FN; the response shows the first20 bounded examples.

An independent reviewer verified all64 actual raw-image, decoded-RGB, label and request-RGB digests, both prediction files, then recomputed complete AP/AR/per-class/matching/diagnostic outputs identically. The original imported model manifest remains byte-for-byte unchanged and the runtime has zero TrainingRun/attempt/legacy evaluation records. Machine evidence: [actual repeat verification](independent-evaluation-verification.json) and [independent recomputation](independent-evaluation-review-evidence.json). Full optional-suite and visible-browser acceptance are separate checks; their current status is recorded by the overall delivery evidence, not inferred from these measurements.

Dedicated backend regression command:

```sh
python -m unittest discover -s tests -p test_model_evaluations.py -v
```

Twenty-two focused tests passed in the optional dependency environment, including real loopback HTTP auth/replay/filter, actual child timeout/reaping and parent-death exit, queue fairness/combined budgets, source/dependency changes, terminal immutability, polling race consistency, tamper/symlink rejection and bounded diagnostics. Synthetic transport/queue fixtures do not claim ONNX validation or recognition quality. A separate host full-suite snapshot ran342 tests:288 passed and54 explicitly skipped for optional prerequisites; skips are not successful actual-model measurements.


### Legacy-protocol regression correction

The first optional-suite pass exposed a strict legacy dataset-serving regression caused by placing new AR-reporting metadata inside the shared evaluator's parameter object. The final implementation keeps that shared parameter object exactly legacy-shaped, retaining unchanged trained-model serving validation and comparison signatures; only the independent evaluation wrapper adds its approved AR metadata. Shared AP/AR numerical outputs are unchanged. The focused post-fix suite passed69 tests with no skips, including real ONNX legacy serving and strict protocol/comparison identity regressions. The refreshed actual64 pair and audit supersede the earlier source-hash evidence; historical audit evidence is explicitly labeled as superseded.

## 2026-10-19 authorized optional preview amendment

The former image-byte exclusion is narrowly expanded for the separate authenticated, read-only [error-image preview contract](error-image-preview-contract.md). Existing detail/list/comparison JSON remains image-free; raw files/paths, arbitrary URLs, complete dataset/TP browsing and client-directed transforms remain excluded. Only a verified bounded wrong-example index can produce a metadata-free, limited PNG. No metrics, evidence, model registration, training, inference or rights claims are changed.
