# Model evaluation

## Status

**No quantitative accuracy evaluation has been performed, and no real-world soccer detection benchmark exists yet.** SoccerVision 3D has not measured precision, recall or mAP on any labelled soccer dataset. RF-DETR's published COCO results describe general-domain performance only and are not reproduced here.

An **evaluation module** now exists (`soccervision.evaluation`, Phase 2A.1). It loads COCO-format ground truth and predictions and computes detection metrics, but it has only been run on hand-built synthetic test fixtures. No numbers from those fixtures are model accuracy.

The rest of this file covers the evaluation module, the annotation format, the metric definitions, and how soccer footage can be annotated later. It ends with what *has* been measured so far: runtime and qualitative inspection on a single short clip.

## Evaluation module

### Supported annotation format

Ground truth is a single [COCO detection](https://cocodataset.org/#format-data) JSON file. Only these fields are read; any others are ignored.

| Section | Required fields | Optional fields |
| --- | --- | --- |
| `images` | `id` (int), `width`, `height` (positive ints), `file_name` | anything else |
| `categories` | `id` (int), `name` | `supercategory` |
| `annotations` | `id`, `image_id`, `category_id` (ints), `bbox` `[x, y, width, height]` in pixels | `iscrowd` (0 or 1, default 0), `area` (> 0, default `width × height`) |
| `info` | — | `description`, `version`, `url` (used as default provenance) |
| `licenses` | — | `name` (used as default provenance license) |

The loader rejects the file if any of the following are found, and lists every problem in one `InvalidDatasetError` (`code: invalid_dataset`):

- Missing sections, wrong types, or no images.
- Duplicate image, category or annotation ids, duplicate file names, or duplicate category names.
- Annotations that refer to images or categories that don't exist.
- Boxes with non-positive width or height, negative coordinates, non-finite or non-numeric values, the wrong number of values, or extents beyond the image (a box may touch the edge).

**Category mapping.** Dataset categories are mapped onto evaluation classes. Only `person` is supported. The default map is `{"person": "person"}`. A soccer dataset might use `{"player": "person", "referee": "person", "goalkeeper": "person"}`. Annotations in unmapped categories (for example `ball`) are ignored and counted in `num_ignored_annotations`. A map key that isn't a dataset category is an error.

**Images without objects** are valid and fully evaluated. Any prediction on them is a false positive.

**Provenance.** `DatasetMetadata` records the annotation file name, its SHA-256, the category map, the counts, and a `DatasetProvenance` block: name, version, license, source URL, description, and a `synthetic` flag. Pass a provenance block explicitly, or it's taken from the file's `info` and `licenses`. The `synthetic` flag is copied into every metrics result as `dataset_synthetic`.

**Predictions** use the standard COCO results list, `[{"image_id", "category_id", "bbox": [x, y, w, h], "score"}]`, where `category_id` refers to the dataset's categories. They can also be built in code as `Prediction` objects using this project's `(x_min, y_min, x_max, y_max)` convention.

- **Rejected when loading a results file** (`code: invalid_predictions`, every issue listed): unknown image or category IDs, scores outside [0, 1], non-finite values, and boxes with negative coordinates or non-positive width or height.
- **Dropped:** predictions in valid dataset categories that the category map doesn't include, mirroring how unmapped ground truth is ignored.
- Boxes that extend past the image edge are accepted, as COCO allows.
- `Prediction` objects built in code validate their own boxes on construction, and `evaluate_detections` rejects any whose image ID is not in the dataset.

**Nothing is downloaded.** You provide the annotation file; datasets with gated terms must be fetched by you after accepting them.

### Metric definitions

All metrics are for one class (`person`) and come from a single run of the reference COCO evaluator, `pycocotools` `COCOeval` (bbox mode, BSD-2-Clause; install with `pip install -e ".[eval]"`).

**Matching**, per image: predictions are sorted by score, highest first, and capped at `max_detections_per_image` (default 100). Each is matched to the unmatched ground-truth box with the highest IoU, provided that IoU is at least the threshold. Consequences:

- A second prediction on an already-matched box (a **duplicate**) is a false positive.
- A prediction that matches only a **crowd** region (`iscrowd: 1`) is ignored: neither a true nor a false positive. Crowd regions are never counted as missed.
- IoU is standard continuous box IoU. For crowd regions, COCO uses intersection divided by the prediction's area.

| Metric | Definition |
| --- | --- |
| `map_50` | Average precision at IoU 0.50: the mean of interpolated precision at 101 recall points (0.00 to 1.00). Uses **all** predictions regardless of the confidence threshold, as COCO does. `null` if the class has no objects. |
| `map_50_95` | Mean of AP over IoU 0.50, 0.55, …, 0.95 (COCO's headline "AP"). |
| `precision` | TP / (TP + FP) among predictions with score ≥ `confidence_threshold` (default 0.5), at `iou_threshold` (default 0.50; any of the ten COCO thresholds). `null` if no prediction passes the threshold. |
| `recall` | TP / number of non-crowd objects at the same threshold and IoU. `null` if there are no objects. |
| `true_positives`, `false_positives`, `false_negatives` | The counts behind precision and recall. |
| `recall_by_size` | Recall at the same threshold and IoU for COCO area buckets: **small** ≤ 32² px, **medium** 32²–96² px, **large** > 96² px, bucketed by the ground truth's `area`. As in COCO, the bounds are inclusive, so an area of exactly 32² or 96² falls in both adjacent buckets. |

Because matching takes the highest scores first, applying the confidence threshold after matching gives the same result as matching only the predictions above it. The counts and mAP therefore share one matching. A test checks that `map_50` and `map_50_95` equal `COCOeval.summarize()` on the same data.

Because mAP uses every prediction, predictions should be exported at a low score threshold. Exporting only detections above the deployed 0.5 threshold would truncate the precision-recall curve and understate mAP.

`evaluate_detections` raises `ValueError` for a class that isn't one of the dataset's evaluation classes, a confidence threshold outside [0, 1], an IoU threshold that isn't one of the ten COCO values, or `max_detections_per_image` below 1. It raises `EvaluationUnavailableError` (`evaluation_unavailable`) if `pycocotools` isn't installed. The result also records the `pycocotools` version used.

### How the module works

```python
from pathlib import Path
from soccervision.evaluation import (
    DatasetProvenance,
    evaluate_detections,
    load_coco_dataset,
    load_coco_predictions,
)

dataset = load_coco_dataset(
    Path("annotations/test.json"),
    category_map={"player": "person", "referee": "person", "goalkeeper": "person"},
    provenance=DatasetProvenance(name="my-soccer-set", version="2026-11", license="CC-BY-4.0"),
)
predictions = load_coco_predictions(Path("predictions/rfdetr-nano.json"), dataset)
metrics = evaluate_detections(dataset, predictions, confidence_threshold=0.5, iou_threshold=0.5)
print(metrics.model_dump_json(indent=2))
```

| File | Role |
| --- | --- |
| `packages/soccervision/evaluation/dataset.py` | Ground-truth loading, validation, category mapping and provenance |
| `packages/soccervision/evaluation/predictions.py` | COCO results loading and validation |
| `packages/soccervision/evaluation/metrics.py` | `evaluate_detections`: builds in-memory COCO indexes, runs `COCOeval`, and derives the thresholded counts from its per-image match records |
| `tests/unit/test_evaluation_*.py`, `tests/fixtures/evaluation/` | Synthetic fixtures and tests. The fixture file and builders are labelled `SYNTHETIC`, and their provenance sets `synthetic: true`. |

**Not implemented yet** (see [ROADMAP.md](ROADMAP.md#phase-2a-detection-evaluation-and-optimization-in-progress)):

- An RF-DETR prediction exporter that runs the detector over a dataset's images and writes COCO results (milestone 2A.2, next).
- An evaluation CLI (2A.3).
- A real soccer evaluation set (2A.4).
- Model comparison and error analysis (2A.5, 2A.6).
- Any frontend for evaluation.

Until 2A.2 exists, predictions must come from some other source.

### Annotating real soccer footage later

1. **Use footage you're allowed to use.** That means your own recordings, or clips under a license that permits annotation and publishing derived metrics, such as CC BY. Don't use broadcast footage without permission. Record the license in `DatasetProvenance`.
2. **Extract frames, not every frame.** Sample frames at a fixed interval (for example 1 per second) so neighbouring frames aren't near-duplicates. Keep the source video ID and frame index in `file_name`, for example `match_a/frame_000030.jpg`.
3. **Split by match, not by frame.** All frames from one match go into the same split (test or train), otherwise results are inflated.
4. **Label with a tool that exports COCO.** [CVAT](https://github.com/cvat-ai/cvat) (MIT) and [Label Studio](https://github.com/HumanSignal/label-studio) (Apache-2.0) both can be self-hosted. Draw tight boxes around every visible person, including partially occluded ones. Use separate categories (`player`, `goalkeeper`, `referee`, `other_person`) and map them to `person` at evaluation time; the roles will be useful later.
5. **Use `iscrowd: 1` for dense crowds**, such as stands, rather than leaving spectators unlabelled. Otherwise correct detections of spectators count as false positives.
6. **Write labelling guidelines down** (minimum box size, how to handle occlusion and truncation at the frame edge), and double-annotate a small subset to measure agreement.
7. **Keep the data out of Git.** Store frames and annotation files outside the repository, and record the annotation file's SHA-256 (done automatically) and version with every reported result.

## Model

| | |
| --- | --- |
| Detector | RF-DETR Nano / Small / Medium (`rfdetr` 1.11.2) |
| Weights | COCO-pretrained checkpoints published by Roboflow (Apache-2.0), downloaded at runtime and MD5-verified |
| Classes used | COCO `person` only (other classes are discarded in post-processing) |
| Default input | Native resolution per variant: 384 (N), 512 (S), 576 (M). Overridable in multiples of 32. |
| Default threshold | 0.5 |

## Observed behaviour (single clip, qualitative)

The clip is ["FC Utrecht takes a Corner"](https://commons.wikimedia.org/wiki/File:FC_Utrecht_takes_a_Corner.ogv) by Pel Laurens (CC BY 3.0): 1280×720, 30 fps, 419 frames, a handheld stadium-stand view of a corner kick. It was converted to H.264 MP4 with `scripts/convert_to_mp4.py`.

These are runs of RF-DETR Nano at its native 384 px input with confidence 0.5, on an Apple Silicon Mac:

| Device | Stride | Analyzed frames | Total detections | Mean / analyzed frame | Max in a frame | Mean inference |
| --- | --- | --- | --- | --- | --- | --- |
| MPS | 1 | 419 | 3,502 | 8.36 | 12 | 38.6 ms |
| CPU | 1 | 419 | 3,502 | 8.36 | 12 | 58 ms |
| MPS | 2 | 210 | 1,754 | 8.35 | 12 | 35.6 ms |

At confidence 0.65 with stride 2, the same clip gave a mean of 3.98 and a maximum of 8 per analyzed frame. Many detected players score between 0.5 and 0.65 at this distance.

A CLI run on the original Ogg/Theora file, before H.264 conversion, reported 3,520 detections, compared with 3,502 on the converted MP4. Lossy re-encoding changes pixels slightly, and borderline detections near the threshold flip as a result. Expect small differences like this across encodings, devices and library versions.

**Qualitative findings:**

- Players in the penalty area and the corner taker are detected reliably, typically at confidence 0.5–0.9.
- **Small, distant players, for example near the far goal, are frequently missed** at 384 px. Raising `inference_resolution` or using Small or Medium is expected to help, but this hasn't been measured yet.
- The model cannot distinguish players, referees, linesmen or spectators. All are `person`.
- Because there is no tracking, per-frame counts fluctuate as boxes appear and disappear (visible in the UI timeline).

None of these numbers are accuracy metrics. "Total detections" counts boxes, not correct boxes, and not unique people.

## Planned evaluation protocol

This protocol corresponds to roadmap milestones 2A.4–2A.6. None of it has been run yet.

1. **Data:** a soccer dataset with person and ball boxes under a license that permits this use. Gated datasets are not downloaded automatically; users fetch them themselves after accepting the terms. Train and test splits are separated by match, not by frame.
2. **Metrics:** COCO mAP@[.5:.95] and AP50 for `person`, recall split by box area (small objects matter most), and precision and recall at the deployed threshold. These are implemented in the evaluation module above. Ball evaluation and false positives per frame on crowd regions are not implemented yet.
3. **Sweep:** model variant × input resolution × confidence threshold, reported alongside latency on CPU, MPS and CUDA.
4. **Reporting:** results tables in this file, with the exact model ID, weights MD5, package versions and dataset version, so every number is reproducible.
