"""Download and verify pretrained RF-DETR COCO weights into the weights directory.

    python scripts/download_weights.py --model rfdetr-nano --weights-dir models

The rfdetr package downloads from its published URL and validates the published MD5. Loading
the model once here also confirms the checkpoint works on this machine.
"""

import argparse
import sys
from pathlib import Path
from typing import get_args

from soccervision.config import DetectorConfig, DetectorName
from soccervision.errors import SoccerVisionError
from soccervision.vision.detection import create_detector, weights_path_for


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", choices=get_args(DetectorName), default="rfdetr-nano")
    parser.add_argument("--weights-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    target = weights_path_for(args.model, args.weights_dir)
    print(f"Ensuring {args.model} weights at {target} ...")
    try:
        detector = create_detector(
            DetectorConfig(name=args.model, device="cpu"),
            weights_dir=args.weights_dir,
            allow_download=True,
        )
    except SoccerVisionError as exc:
        print(f"Failed: {exc.message}", file=sys.stderr)
        return 1
    info = detector.info
    print(
        f"OK: {info.model_id} ({info.license}), md5={info.weights_md5}, input={info.input_resolution}px"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
