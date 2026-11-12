# Model evaluation

## Status

**No quantitative accuracy evaluation has been performed.** SoccerVision 3D has not measured precision, recall or mAP on any labelled soccer dataset. RF-DETR's published COCO results describe general-domain performance only and are not reproduced here.

What *has* been done is described below: runtime measurements and qualitative inspection on a single short clip.

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

## Planned evaluation protocol (Phase 6)

1. **Data:** a soccer dataset with person and ball boxes under a license that permits this use. Gated datasets are not downloaded automatically; users fetch them themselves after accepting the terms. Train and test splits are separated by match, not by frame.
2. **Metrics:** COCO mAP@[.5:.95] and AP50 for `person`; recall split by box area (small objects matter most); and false positives per frame on crowd regions.
3. **Sweep:** model variant × input resolution × confidence threshold, reported alongside latency on CPU, MPS and CUDA.
4. **Reporting:** results tables in this file, with the exact model ID, weights MD5, package versions and dataset version, so every number is reproducible.
