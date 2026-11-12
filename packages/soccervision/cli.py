"""Command-line interface.

python -m soccervision.cli analyze --input clip.mp4 --output out/
python -m soccervision.cli devices
"""

import argparse
import json
import logging
import re
import sys
import uuid
from pathlib import Path
from typing import get_args

from pydantic import ValidationError

from soccervision.config import DetectorConfig, DetectorName, DeviceName, PipelineConfig
from soccervision.devices import detect_available_devices
from soccervision.pipeline import (
    METADATA_FILE,
    PipelineRunError,
    ProgressUpdate,
    run_detection_pipeline,
)
from soccervision.vision.detection import ObjectDetector, create_detector

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soccervision", description="SoccerVision 3D command-line tools."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Run person detection on a local video.")
    analyze.add_argument("--input", "-i", type=Path, required=True, help="Input video file.")
    analyze.add_argument("--output", "-o", type=Path, required=True, help="Output run directory.")
    analyze.add_argument("--model", choices=get_args(DetectorName), default="rfdetr-nano")
    analyze.add_argument(
        "--confidence", type=float, default=0.5, help="Confidence threshold (0-1)."
    )
    analyze.add_argument("--stride", type=int, default=1, help="Run inference on every Nth frame.")
    analyze.add_argument("--resolution", type=int, default=None, help="Square model input size.")
    analyze.add_argument("--device", choices=get_args(DeviceName), default="auto")
    analyze.add_argument(
        "--max-duration", type=float, default=600.0, help="Reject longer inputs (s)."
    )
    analyze.add_argument("--weights-dir", type=Path, default=Path("models"))
    analyze.add_argument(
        "--no-download", action="store_true", help="Fail instead of downloading weights."
    )
    analyze.add_argument("--no-hud", action="store_true", help="Do not draw the status overlay.")
    analyze.add_argument("--require-mp4", action="store_true", help="Reject non-MP4 containers.")
    analyze.add_argument(
        "--overwrite", action="store_true", help="Replace an existing run in --output."
    )
    analyze.add_argument("--video-id", default=None, help="Identifier recorded in artifacts.")

    sub.add_parser("devices", help="Show available compute devices.")
    return parser


class _ProgressPrinter:
    """Redraws one line on a TTY; otherwise prints at most one line per 10%."""

    def __init__(self) -> None:
        self._interactive = sys.stderr.isatty()
        self._last_decile = -1

    def __call__(self, update: ProgressUpdate) -> None:
        total = update.expected_frames
        text = f"processed {update.processed_frames}" + (
            f"/{total} frames ({100 * update.fraction:5.1f}%)" if total else " frames"
        )
        if self._interactive:
            sys.stderr.write(f"\r  {text}")
            sys.stderr.flush()
            return
        decile = int(update.fraction * 10)
        if decile != self._last_decile:
            self._last_decile = decile
            print(f"  {text}", file=sys.stderr)

    def finish(self) -> None:
        if self._interactive:
            sys.stderr.write("\n")


def _cmd_analyze(args: argparse.Namespace) -> int:
    try:
        config = PipelineConfig(
            detector=DetectorConfig(
                name=args.model,
                confidence_threshold=args.confidence,
                inference_resolution=args.resolution,
                device=args.device,
            ),
            inference_stride=args.stride,
            max_duration_seconds=args.max_duration,
            draw_hud=not args.no_hud,
        )
    except ValidationError as exc:
        print(f"Invalid configuration:\n{exc}", file=sys.stderr)
        return EXIT_USAGE

    output: Path = args.output
    if (output / METADATA_FILE).exists() and not args.overwrite:
        print(f"{output} already contains a run; pass --overwrite to replace it.", file=sys.stderr)
        return EXIT_USAGE

    video_id = (
        args.video_id or re.sub(r"[^A-Za-z0-9_-]+", "-", args.input.stem).strip("-") or "video"
    )
    run_id = uuid.uuid4().hex
    weights_dir: Path = args.weights_dir
    allow_download = not args.no_download

    def factory(detector_config: DetectorConfig) -> ObjectDetector:
        return create_detector(
            detector_config, weights_dir=weights_dir, allow_download=allow_download
        )

    print(f"Analyzing {args.input} -> {output}", file=sys.stderr)
    progress = _ProgressPrinter()
    try:
        result = run_detection_pipeline(
            input_path=args.input,
            run_dir=output,
            config=config,
            detector_factory=factory,
            video_id=video_id,
            run_id=run_id,
            progress=progress,
            require_mp4=args.require_mp4,
        )
    except PipelineRunError as exc:
        progress.finish()
        print(f"Analysis failed [{exc.code}]: {exc.message}", file=sys.stderr)
        if exc.metadata_path:
            print(f"Failure metadata: {exc.metadata_path}", file=sys.stderr)
        return EXIT_FAILED
    progress.finish()

    meta = result.metadata
    summary = {
        "run_dir": str(result.run_dir),
        "model": meta.model.model_id if meta.model else None,
        "device": meta.model.device if meta.model else None,
        "stats": meta.stats.model_dump() if meta.stats else None,
        "processing_seconds": round(meta.timing.total_seconds, 2),
        "processing_fps": round(meta.timing.processing_fps or 0.0, 2),
        "output_codec": meta.output_video.codec if meta.output_video else None,
        "warnings": meta.warnings,
    }
    print(json.dumps(summary, indent=2))
    return EXIT_OK


def _cmd_devices() -> int:
    availability = detect_available_devices()
    print(
        json.dumps(
            {
                "torch_installed": availability.torch_installed,
                "cuda": availability.cuda,
                "mps": availability.mps,
                "cpu": availability.cpu,
            },
            indent=2,
        )
    )
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "analyze":
        return _cmd_analyze(args)
    return _cmd_devices()


if __name__ == "__main__":
    sys.exit(main())
