# Nonempty detection validation: bounded calibration

This increment adds preparation and acceptance scripts, not a new model architecture,
pretrained-weight importer, production quality gate, or business workflow.

## Frozen protocol

- Published baseline: `c3376ff109385360cf4c7d2e829d54a1b1d96594`; existing model-platform implementation unchanged
- Original plan: reviewed official Open Images Cat 128/32/32 slice. The pinned official train bounding-box metadata download returned proxy HTTP 403. No alternate host, replaced manifest, relaxed checksum or replacement photos were used
- `scripts/prepare_cat_import.py` prepares a max-side-256 RGB PNG derivative only after the original manifest, image bytes, attribution and duplicate checks pass. Normalized boxes preserve the actual independent resized x/y scales. Real-data conversion remains **not executed** because no complete cache is available
- Fallback: `scripts/make_learnability_fixture.py` creates original red/blue rectangle scenes, 128 train / 16 validation / 16 test, disjoint fixed random-seed ranges. Landscape/portrait/square dimensions and varied rectangle positions/sizes exercise coordinate handling; all decoded pixels are unique. This tests learnability and display mechanics, **not real-photo accuracy or production suitability**
- Training configuration fixed before any run: 20 epochs, Adam learning rate .001, seed 20261015, existing batch4/input160/from-scratch/final-epoch recipe
- Existing service confidence .01 and evaluation confidence .001, NMS .65, max100 unchanged. No threshold lowering, extra run, checkpoint selection or hyperparameter adjustment based on held-out results

## Acceptance procedure

Run the existing verified cloud-desktop development launcher with the published backend,
then generate the fixture in a new output directory using its hash-locked Python environment:

```sh
python scripts/make_learnability_fixture.py /path/to/new-fixture
MP_HOME=/path/to/launcher/session python scripts/validate_learned_detections.py \
  --api http://127.0.0.1:8787 --origin http://127.0.0.1:5666 \
  --fixture /path/to/new-fixture --output /path/to/new-evidence
```

The acceptance script uses the existing loopback BFF and does not read or export launcher
credentials. It records import/version/run IDs before waiting, submits one idempotent
training intent, compares all 16 test images' online predictions against a separately
reloaded hash-verified ONNX artifact, then independently recomputes overall and per-class
COCO-style AP. Same thresholds must be compared: service .01 outputs are not substituted
for .001 evaluation outputs. A fully empty detector records `not-achieved` and exits
unsuccessfully; it is not silently called nonempty-result validation.

The deployment is deliberately left ready for the same-session browser overlay check.
Afterward stop it using the existing UI or stop the managed launcher. No services are
published publicly. Evidence/data/artifacts stay outside the source repository.

## Licenses and limits

YOLOX source retains its existing Apache-2.0 notice. No external pretrained weights are
used. Synthetic pixels and labels are generated here without third-party imagery.
For the blocked official-data option, Open Images annotations are CC BY 4.0; the selected
image metadata reports CC BY 2.0, but Open Images does not guarantee individual-image
license status. Preserve per-image author/source/license information and the resize notice.
See [official license statement](https://storage.googleapis.com/openimages/web/factsfigures.html#licenses)
and [official download/metadata documentation](https://storage.googleapis.com/openimages/web/download_v7.html).

## Execution status

- One actual 20-epoch run / one attempt completed, exported and reloaded a 3,664,533-byte ONNX artifact
- All 16 synthetic test images produced nonempty service results. Online and independently reloaded ONNX service predictions were exactly equal; independently recomputed overall and per-class AP matched the registered evaluation
- Synthetic mAP50: 1.0; mAP50:95: 0.6953688604. Per-class AP50:95: red 0.7280644535, blue 0.6626732673
- At unchanged .01 service threshold there are 3–9 predictions per image for two true rectangles, including low-confidence duplicates. AP50=1.0 does not mean every prediction is correct
- Full backend suite: 233 tests, 232 passed, one official Fashion-MNIST cache test skipped. Five new focused preparation tests passed
- Browser checks passed on preselected landscape/portrait/square images (3/9/8 boxes): exact coordinates, class/score labels and aligned image/SVG bounds. A real small-image SVG scaling bug was fixed separately in vision-lab; labels remain 12 screen pixels and strokes 2 pixels, including actual preview resize from 640 to 572 pixels
- Frontend fix: 222 tests, frontend/BFF typechecks, production build and ESLint passed. Existing baseline stylelint/formatting debt remains; not a clean all-formatters claim
- One Chromium tab crash during checks recovered after the build finished; cause was not established. No retraining, changed thresholds or hidden low-confidence detections
- Both local deployments and all owned launcher services were cleanly stopped after QA. Runtime, one run/attempt/model and artifact were preserved
- No claim of real-photo accuracy, production readiness or remote publication

Structured safe summary: [nonempty-detection-verification.json](nonempty-detection-verification.json). Raw runtime/images/weights are excluded from the repository.
