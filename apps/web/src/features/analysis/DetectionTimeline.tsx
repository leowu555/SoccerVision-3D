"use client";

import { useMemo, type MouseEvent } from "react";
import { Card, CardHeader } from "@/components/ui/primitives";
import { formatDuration } from "@/lib/format";
import { downsampleMax, frameCountAt } from "@/lib/stats";
import type { FrameCount } from "@/types/api";

const WIDTH = 1000;
const HEIGHT = 140;
const PAD_TOP = 10;
const PAD_BOTTOM = 20;
const MAX_POINTS = 500;

interface Props {
  counts: FrameCount[];
  duration: number;
  currentTime: number;
  stride: number;
  onSeek: (time: number) => void;
}

export function DetectionTimeline({ counts, duration, currentTime, stride, onSeek }: Props) {
  const span = Math.max(duration, counts.at(-1)?.timestamp_seconds ?? 0, 1e-3);
  const { path, area, maxY } = useMemo(() => {
    const points = downsampleMax(counts, MAX_POINTS);
    const peak = Math.max(1, ...points.map((p) => p.y));
    const plotH = HEIGHT - PAD_TOP - PAD_BOTTOM;
    const toX = (t: number) => (t / span) * WIDTH;
    const toY = (y: number) => PAD_TOP + plotH - (y / peak) * plotH;
    const line = points.map((p, i) => `${i === 0 ? "M" : "L"}${toX(p.x).toFixed(1)},${toY(p.y).toFixed(1)}`).join("");
    const fill = points.length
      ? `${line}L${toX(points.at(-1)!.x).toFixed(1)},${HEIGHT - PAD_BOTTOM}L${toX(points[0].x).toFixed(1)},${HEIGHT - PAD_BOTTOM}Z`
      : "";
    return { path: line, area: fill, maxY: peak };
  }, [counts, span]);

  const current = frameCountAt(counts, currentTime);
  const playheadX = (Math.min(currentTime, span) / span) * WIDTH;

  function handleClick(event: MouseEvent<SVGSVGElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    const fraction = (event.clientX - rect.left) / rect.width;
    onSeek(Math.min(Math.max(fraction, 0), 1) * span);
  }

  return (
    <Card>
      <CardHeader
        title="Person detections over time"
        subtitle={`One point per analyzed frame${stride > 1 ? ` (every ${stride} frames)` : ""}. Click to seek.`}
        action={
          <div className="text-right font-mono text-xs text-slate-400">
            <div>t = {formatDuration(currentTime)}</div>
            <div className="text-slate-200">
              {current ? `${current.detections} in frame #${current.frame_index}` : "no analyzed frame yet"}
            </div>
          </div>
        }
      />
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        preserveAspectRatio="none"
        className="h-36 w-full cursor-crosshair"
        onClick={handleClick}
        role="img"
        aria-label="Detections per analyzed frame over time"
      >
        <line x1={0} x2={WIDTH} y1={HEIGHT - PAD_BOTTOM} y2={HEIGHT - PAD_BOTTOM} stroke="rgb(51 65 85)" strokeWidth={1} />
        <line x1={0} x2={WIDTH} y1={PAD_TOP} y2={PAD_TOP} stroke="rgb(30 41 59)" strokeDasharray="4 4" strokeWidth={1} />
        <path d={area} fill="rgb(34 211 238 / 0.12)" />
        <path d={path} fill="none" stroke="rgb(34 211 238)" strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
        <line
          x1={playheadX}
          x2={playheadX}
          y1={0}
          y2={HEIGHT - PAD_BOTTOM}
          stroke="rgb(251 191 36)"
          strokeWidth={1.5}
          vectorEffect="non-scaling-stroke"
        />
      </svg>
      <div className="mt-1 flex justify-between font-mono text-[10px] text-slate-500">
        <span>0:00.0</span>
        <span>peak {maxY}</span>
        <span>{formatDuration(span)}</span>
      </div>
    </Card>
  );
}
