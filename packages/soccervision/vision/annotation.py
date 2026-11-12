"""Frame annotation with Pillow.

Pillow is used instead of OpenCV deliberately: opencv-python wheels bundle their own FFmpeg, which
conflicts with PyAV's on macOS (duplicate Objective-C classes).
"""

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont

from soccervision.schemas.detection import FrameRecord, InferenceStatus

BOX_COLOR = (34, 211, 238)  # cyan-400
LABEL_TEXT_COLOR = (8, 15, 25)
HUD_TEXT_COLOR = (226, 232, 240)
HUD_MUTED_COLOR = (148, 163, 184)
HUD_WARNING_COLOR = (251, 191, 36)
HUD_BACKGROUND_ALPHA = 0.65


class FrameAnnotator:
    def __init__(
        self, *, width: int, height: int, model_label: str, inference_stride: int, draw_hud: bool
    ):
        self._model_label = model_label
        self._stride = inference_stride
        self._draw_hud = draw_hud
        scale = max(height, 360) / 720
        self._line_width = max(1, round(2 * scale))
        self._label_font = ImageFont.load_default(size=max(10, round(14 * scale)))
        self._hud_font = ImageFont.load_default(size=max(11, round(16 * scale)))
        self._pad = max(4, round(8 * scale))
        self._width = width
        self._height = height

    def annotate(
        self, image_rgb: npt.NDArray[np.uint8], record: FrameRecord
    ) -> npt.NDArray[np.uint8]:
        out = image_rgb.copy()
        lines = self._hud_lines(record)
        if self._draw_hud:
            self._shade_hud_region(out, lines)

        canvas = Image.fromarray(out)
        draw = ImageDraw.Draw(canvas)
        for det in record.detections or []:
            x1, y1, x2, y2 = det.bbox_xyxy
            draw.rectangle((x1, y1, x2, y2), outline=BOX_COLOR, width=self._line_width)
            label = f"{det.class_name} {det.confidence:.2f}"
            tx0, ty0, tx1, ty1 = draw.textbbox((0, 0), label, font=self._label_font)
            text_w, text_h = tx1 - tx0, ty1 - ty0
            top = y1 - text_h - 4 if y1 - text_h - 4 >= 0 else y1
            draw.rectangle((x1, top, x1 + text_w + 6, top + text_h + 4), fill=BOX_COLOR)
            draw.text(
                (x1 + 3 - tx0, top + 2 - ty0), label, fill=LABEL_TEXT_COLOR, font=self._label_font
            )

        if self._draw_hud:
            y = self._pad
            for text, color in lines:
                draw.text((self._pad * 2, y), text, fill=color, font=self._hud_font)
                y += self._line_height(draw)
        return np.asarray(canvas)

    def _hud_lines(self, record: FrameRecord) -> list[tuple[str, tuple[int, int, int]]]:
        header = f"SoccerVision 3D  |  {self._model_label}"
        position = f"frame {record.frame_index}  |  t={record.timestamp_seconds:.3f}s"
        if record.inference_status is InferenceStatus.SKIPPED:
            status = (f"NOT ANALYZED (inference every {self._stride} frames)", HUD_WARNING_COLOR)
        else:
            count = len(record.detections or [])
            status = (f"{count} detection{'s' if count != 1 else ''}", HUD_TEXT_COLOR)
        return [(header, HUD_MUTED_COLOR), (position, HUD_TEXT_COLOR), status]

    def _line_height(self, draw: ImageDraw.ImageDraw) -> int:
        _, top, _, bottom = draw.textbbox((0, 0), "Ag", font=self._hud_font)
        return int(bottom - top) + self._pad // 2 + 2

    def _shade_hud_region(
        self, image: npt.NDArray[np.uint8], lines: list[tuple[str, tuple[int, int, int]]]
    ) -> None:
        probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        text_width = max(probe.textlength(text, font=self._hud_font) for text, _ in lines)
        box_w = min(self._width, int(text_width) + self._pad * 4)
        box_h = min(self._height, self._line_height(probe) * len(lines) + self._pad * 2)
        region = image[:box_h, :box_w].astype(np.float32)
        image[:box_h, :box_w] = (
            region * (1 - HUD_BACKGROUND_ALPHA) + 10 * HUD_BACKGROUND_ALPHA
        ).astype(np.uint8)
