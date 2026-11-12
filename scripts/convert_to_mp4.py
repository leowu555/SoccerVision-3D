"""Convert any FFmpeg-readable video to an H.264 MP4 the upload endpoint accepts.

    python scripts/convert_to_mp4.py input.webm output.mp4 [--max-seconds 30]

Video only (audio is dropped). Frame timestamps are preserved.
"""

import argparse
import sys
from pathlib import Path

from soccervision.errors import SoccerVisionError
from soccervision.vision.video import AnnotatedVideoWriter, VideoDecoder, probe_video


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--max-seconds", type=float, default=None, help="Keep only the first N seconds."
    )
    args = parser.parse_args()

    try:
        probe = probe_video(args.input, require_mp4=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with (
            VideoDecoder(args.input) as decoder,
            AnnotatedVideoWriter(
                args.output,
                width=probe.width,
                height=probe.height,
                frame_rate=decoder.frame_rate,
                time_base=decoder.time_base,
            ) as writer,
        ):
            for frame in decoder.frames():
                if args.max_seconds is not None and frame.timestamp_seconds >= args.max_seconds:
                    break
                writer.write(frame.image, frame.pts or 0)
    except SoccerVisionError as exc:
        print(f"Conversion failed: {exc.message}", file=sys.stderr)
        return 1
    print(f"Wrote {writer.frames_written} frames ({writer.codec_name}) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
