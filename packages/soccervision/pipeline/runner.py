"""Phase 1 pipeline: decode -> detect -> annotate -> export.

This is the single implementation used by both the CLI and the API worker.
"""

import importlib.metadata
import logging
import math
import platform
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from soccervision import __version__
from soccervision.config import PipelineConfig
from soccervision.errors import InvalidVideoError, SoccerVisionError, VideoLimitExceededError
from soccervision.pipeline.artifacts import (
    ANNOTATED_VIDEO_FILE,
    DETECTIONS_FILE,
    METADATA_FILE,
    sha256_file,
    write_json_atomic,
)
from soccervision.schemas.artifacts import (
    CoordinateSystem,
    DetectionsArtifact,
    EnvironmentInfo,
    ErrorInfo,
    OutputVideoInfo,
    PipelineInfo,
    RunMetadata,
    RunStats,
    RunStatus,
    RunTiming,
)
from soccervision.schemas.detection import Detection, FrameRecord, InferenceStatus, ModelInfo
from soccervision.schemas.video import VideoProbe
from soccervision.vision.annotation import FrameAnnotator
from soccervision.vision.detection.base import DetectorFactory, ObjectDetector, RawDetection
from soccervision.vision.video.decoder import VideoDecoder
from soccervision.vision.video.encoder import PIXEL_FORMAT, AnnotatedVideoWriter
from soccervision.vision.video.probe import probe_video

logger = logging.getLogger(__name__)

PIPELINE_NAME = "person-detection"
PIPELINE_VERSION = "1.0.0"
PIPELINE_STAGES = ("decode", "detect", "annotate", "export")
_PROGRESS_STEPS = 200


@dataclass(frozen=True, slots=True)
class ProgressUpdate:
    processed_frames: int
    expected_frames: int | None

    @property
    def fraction(self) -> float:
        if not self.expected_frames:
            return 0.0
        return min(self.processed_frames / self.expected_frames, 1.0)


ProgressCallback = Callable[[ProgressUpdate], None]


class PipelineRunError(SoccerVisionError):
    """A run failed. ``metadata_path`` points at the failed-run metadata, if it could be written."""

    def __init__(self, code: str, message: str, metadata_path: Path | None) -> None:
        super().__init__(message)
        self.code = code
        self.metadata_path = metadata_path


@dataclass(frozen=True)
class PipelineResult:
    run_dir: Path
    metadata: RunMetadata
    annotated_video_path: Path
    detections_path: Path
    metadata_path: Path


@dataclass
class _RunState:
    current_frame: int | None = None
    decoded: int = 0
    analyzed: int = 0
    total_detections: int = 0
    max_detections: int = 0
    estimated_timestamps: int = 0
    clipped: int = 0
    discarded: int = 0
    inference_seconds: float = 0.0
    records: list[FrameRecord] = field(default_factory=list)

    def stats(self) -> RunStats:
        return RunStats(
            decoded_frames=self.decoded,
            analyzed_frames=self.analyzed,
            skipped_frames=self.decoded - self.analyzed,
            total_detections=self.total_detections,
            mean_detections_per_analyzed_frame=(
                self.total_detections / self.analyzed if self.analyzed else None
            ),
            max_detections_in_frame=self.max_detections,
            frames_with_estimated_timestamps=self.estimated_timestamps,
            clipped_boxes=self.clipped,
            discarded_degenerate_boxes=self.discarded,
        )


def run_detection_pipeline(
    *,
    input_path: Path,
    run_dir: Path,
    config: PipelineConfig,
    detector_factory: DetectorFactory,
    video_id: str,
    run_id: str,
    progress: ProgressCallback | None = None,
    require_mp4: bool = False,
) -> PipelineResult:
    """Run detection on every ``inference_stride``-th frame and write artifacts into ``run_dir``.

    On failure, a ``metadata.json`` with ``status: failed`` is written (best effort), partial
    outputs are removed, and :class:`PipelineRunError` is raised.
    """
    started_at = datetime.now(UTC)
    t_start = perf_counter()
    run_dir.mkdir(parents=True, exist_ok=True)
    partial_video = run_dir / f".{ANNOTATED_VIDEO_FILE}.partial"
    final_video = run_dir / ANNOTATED_VIDEO_FILE
    detections_path = run_dir / DETECTIONS_FILE
    metadata_path = run_dir / METADATA_FILE

    state = _RunState()
    warnings: list[str] = []
    probe: VideoProbe | None = None
    model_info: ModelInfo | None = None
    input_sha256: str | None = None
    model_load_seconds: float | None = None
    output_info: OutputVideoInfo | None = None

    def build_metadata(status: RunStatus, error: ErrorInfo | None) -> RunMetadata:
        completed_at = datetime.now(UTC)
        total = perf_counter() - t_start
        return RunMetadata(
            run_id=run_id,
            video_id=video_id,
            status=status,
            pipeline=PipelineInfo(
                name=PIPELINE_NAME,
                version=PIPELINE_VERSION,
                package_version=__version__,
                stages=list(PIPELINE_STAGES),
            ),
            configuration=config.model_dump(mode="json"),
            model=model_info,
            input_file=input_path.name,
            input_sha256=input_sha256,
            input_video=probe,
            output_video=output_info if status is RunStatus.COMPLETED else None,
            stats=state.stats() if state.decoded else None,
            timing=RunTiming(
                started_at=started_at,
                completed_at=completed_at,
                total_seconds=total,
                model_load_seconds=model_load_seconds,
                inference_seconds=state.inference_seconds,
                mean_inference_ms=(
                    1000 * state.inference_seconds / state.analyzed if state.analyzed else None
                ),
                processing_fps=state.decoded / total if total > 0 and state.decoded else None,
            ),
            environment=_environment(model_info),
            artifacts=(
                {
                    "annotated_video": ANNOTATED_VIDEO_FILE,
                    "detections": DETECTIONS_FILE,
                    "metadata": METADATA_FILE,
                }
                if status is RunStatus.COMPLETED
                else {"metadata": METADATA_FILE}
            ),
            warnings=warnings,
            error=error,
        )

    try:
        probe = probe_video(input_path, require_mp4=require_mp4)
        _check_limits(probe, config)
        if probe.frame_rate_mode == "variable":
            warnings.append(
                "Variable frame rate input: frame timestamps use real presentation timestamps."
            )
        input_sha256 = sha256_file(input_path)

        t_load = perf_counter()
        detector = detector_factory(config.detector)
        model_load_seconds = perf_counter() - t_load
        model_info = detector.info
        if model_info.is_synthetic:
            warnings.append(
                "SYNTHETIC detector in use: detections are test fixtures, not model output."
            )
        logger.info(
            "Loaded %s on %s in %.2fs", model_info.model_id, model_info.device, model_load_seconds
        )

        output_info = _process_frames(
            input_path=input_path,
            output_path=partial_video,
            probe=probe,
            config=config,
            detector=detector,
            video_id=video_id,
            state=state,
            progress=progress,
        )
        output_info = output_info.model_copy(update={"file": ANNOTATED_VIDEO_FILE})
        if not output_info.browser_playable:
            warnings.append(
                f"No H.264 encoder available; output uses {output_info.codec} and may not play in browsers."
            )
        if state.estimated_timestamps:
            warnings.append(
                f"{state.estimated_timestamps} frame(s) lacked PTS; timestamps were estimated."
            )

        write_json_atomic(
            detections_path,
            DetectionsArtifact(
                run_id=run_id,
                video_id=video_id,
                model=model_info,
                confidence_threshold=config.detector.confidence_threshold,
                target_classes=list(config.detector.target_classes),
                inference_stride=config.inference_stride,
                coordinate_system=CoordinateSystem(
                    frame_width=probe.width, frame_height=probe.height
                ),
                frames=state.records,
            ),
        )
        partial_video.replace(final_video)
        metadata = build_metadata(RunStatus.COMPLETED, None)
        write_json_atomic(metadata_path, metadata, indent=2)
        logger.info(
            "Run %s completed: %d frames decoded, %d analyzed, %d detections",
            run_id,
            state.decoded,
            state.analyzed,
            state.total_detections,
        )
        return PipelineResult(
            run_dir=run_dir,
            metadata=metadata,
            annotated_video_path=final_video,
            detections_path=detections_path,
            metadata_path=metadata_path,
        )
    except BaseException as exc:
        if isinstance(exc, SoccerVisionError):
            code, message = exc.code, exc.message
            logger.error("Run %s failed (%s): %s", run_id, code, message)
        elif isinstance(exc, Exception):
            code, message = "internal_error", f"{type(exc).__name__}: {exc}"
            logger.exception("Run %s failed with an unexpected error", run_id)
        else:
            code, message = "interrupted", f"Run interrupted ({type(exc).__name__})."
            logger.warning("Run %s interrupted", run_id)
        for path in (partial_video, final_video, detections_path):
            path.unlink(missing_ok=True)
        written: Path | None = None
        try:
            error = ErrorInfo(code=code, message=message, frame_index=state.current_frame)
            write_json_atomic(metadata_path, build_metadata(RunStatus.FAILED, error), indent=2)
            written = metadata_path
        except Exception:
            logger.exception("Could not write failure metadata for run %s", run_id)
        if not isinstance(exc, Exception):
            raise
        raise PipelineRunError(code, message, written) from exc


def _check_limits(probe: VideoProbe, config: PipelineConfig) -> None:
    limit = config.max_duration_seconds
    if limit is not None and probe.duration_seconds > limit:
        raise VideoLimitExceededError(
            f"Video is {probe.duration_seconds:.1f}s long; the configured maximum is {limit:.0f}s."
        )


def _process_frames(
    *,
    input_path: Path,
    output_path: Path,
    probe: VideoProbe,
    config: PipelineConfig,
    detector: ObjectDetector,
    video_id: str,
    state: _RunState,
    progress: ProgressCallback | None,
) -> OutputVideoInfo:
    info = detector.info
    prefix = "SYNTHETIC " if info.is_synthetic else ""
    label = f"{prefix}{info.model_id} (conf >= {config.detector.confidence_threshold:.2f})"
    annotator = FrameAnnotator(
        width=probe.width,
        height=probe.height,
        model_label=label,
        inference_stride=config.inference_stride,
        draw_hud=config.draw_hud,
    )
    expected = probe.frame_count
    report_every = max(1, (expected or 0) // _PROGRESS_STEPS) if expected else 10
    stride = config.inference_stride

    with (
        VideoDecoder(input_path) as decoder,
        AnnotatedVideoWriter(
            output_path,
            width=probe.width,
            height=probe.height,
            frame_rate=decoder.frame_rate,
            time_base=decoder.time_base,
        ) as writer,
    ):
        for frame in decoder.frames():
            state.current_frame = frame.index
            height, width = frame.image.shape[:2]
            if (width, height) != (probe.width, probe.height):
                raise InvalidVideoError(
                    f"Frame {frame.index} is {width}x{height}, but the stream is "
                    f"{probe.width}x{probe.height}; mid-stream resolution changes are unsupported."
                )

            if frame.index % stride == 0:
                t0 = perf_counter()
                raw = detector.detect(frame.image)
                state.inference_seconds += perf_counter() - t0
                detections = _postprocess(
                    raw, frame.index, frame.timestamp_seconds, video_id, info, config, probe, state
                )
                record = FrameRecord(
                    frame_index=frame.index,
                    timestamp_seconds=frame.timestamp_seconds,
                    timestamp_source=frame.timestamp_source,
                    inference_status=InferenceStatus.ANALYZED,
                    detections=detections,
                )
                state.analyzed += 1
                state.total_detections += len(detections)
                state.max_detections = max(state.max_detections, len(detections))
            else:
                record = FrameRecord(
                    frame_index=frame.index,
                    timestamp_seconds=frame.timestamp_seconds,
                    timestamp_source=frame.timestamp_source,
                    inference_status=InferenceStatus.SKIPPED,
                    detections=None,
                )

            if frame.timestamp_source == "estimated":
                state.estimated_timestamps += 1
            state.records.append(record)
            writer.write(annotator.annotate(frame.image, record), frame.pts or 0)
            state.decoded += 1

            if progress is not None and state.decoded % report_every == 0:
                progress(ProgressUpdate(state.decoded, expected))

    if state.decoded == 0:
        raise InvalidVideoError("No frames could be decoded from the video.")
    state.current_frame = None
    if progress is not None:
        progress(ProgressUpdate(state.decoded, state.decoded))

    return OutputVideoInfo(
        file=output_path.name,
        codec=writer.codec_name,
        pixel_format=PIXEL_FORMAT,
        width=writer.width,
        height=writer.height,
        browser_playable=writer.browser_playable,
        padded_to_even_dimensions=writer.padded,
    )


def _postprocess(
    raw: Sequence[RawDetection],
    frame_index: int,
    timestamp: float,
    video_id: str,
    info: ModelInfo,
    config: PipelineConfig,
    probe: VideoProbe,
    state: _RunState,
) -> list[Detection]:
    """Filter by class and threshold, clip to the frame, drop degenerate boxes, assign IDs."""
    wanted = set(config.detector.target_classes)
    threshold = config.detector.confidence_threshold
    out: list[Detection] = []
    for det in raw:
        if det.class_name not in wanted or det.confidence < threshold:
            continue
        if not all(math.isfinite(v) for v in det.bbox_xyxy):
            state.discarded += 1
            continue
        x1, y1, x2, y2 = det.bbox_xyxy
        cx1, cy1 = min(max(x1, 0.0), probe.width), min(max(y1, 0.0), probe.height)
        cx2, cy2 = min(max(x2, 0.0), probe.width), min(max(y2, 0.0), probe.height)
        if cx2 <= cx1 or cy2 <= cy1:
            state.discarded += 1
            continue
        if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
            state.clipped += 1
        out.append(
            Detection(
                video_id=video_id,
                frame_index=frame_index,
                timestamp_seconds=timestamp,
                detection_id=f"f{frame_index}-d{len(out)}",
                class_name=det.class_name,
                confidence=min(max(det.confidence, 0.0), 1.0),
                bbox_xyxy=(cx1, cy1, cx2, cy2),
                model_id=info.model_id,
            )
        )
    return out


def _environment(model_info: ModelInfo | None) -> EnvironmentInfo:
    packages: dict[str, str] = {"soccervision": __version__}
    for name in ("av", "numpy", "pillow", "pydantic"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    if model_info is not None:
        packages.update(model_info.framework_versions)
    return EnvironmentInfo(
        python=sys.version.split()[0], platform=platform.platform(), packages=packages
    )
