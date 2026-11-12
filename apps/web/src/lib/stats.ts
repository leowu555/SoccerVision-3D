import type { FrameCount, JobStatus, RunMetadata, VideoProcessingStatus } from "@/types/api";

export function isActiveJob(status: JobStatus | null | undefined): boolean {
  return status === "pending" || status === "processing";
}

export function isActiveVideo(status: VideoProcessingStatus): boolean {
  return status === "queued" || status === "processing";
}

/** Fraction of decoded frames the detector actually ran on. */
export function analysisCoverage(run: RunMetadata): number | null {
  const stats = run.stats;
  if (!stats || stats.decoded_frames === 0) return null;
  return stats.analyzed_frames / stats.decoded_frames;
}

/**
 * Detection count for the analyzed frame shown at `time`: the latest analyzed frame whose
 * timestamp is <= time. Returns null before the first analyzed frame. When inference ran on
 * sampled frames, the result belongs to `frame_index`, not necessarily the displayed frame.
 */
export function frameCountAt(counts: FrameCount[], time: number): FrameCount | null {
  let lo = 0;
  let hi = counts.length - 1;
  let found: FrameCount | null = null;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (counts[mid].timestamp_seconds <= time + 1e-6) {
      found = counts[mid];
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return found;
}

export interface SeriesPoint {
  x: number;
  y: number;
}

/** Downsample to at most `maxPoints` buckets, keeping each bucket's maximum so peaks survive. */
export function downsampleMax(counts: FrameCount[], maxPoints: number): SeriesPoint[] {
  if (counts.length <= maxPoints) {
    return counts.map((c) => ({ x: c.timestamp_seconds, y: c.detections }));
  }
  const bucketSize = counts.length / maxPoints;
  const points: SeriesPoint[] = [];
  for (let b = 0; b < maxPoints; b++) {
    const start = Math.floor(b * bucketSize);
    const end = Math.min(counts.length, Math.floor((b + 1) * bucketSize));
    let best = counts[start];
    for (let i = start + 1; i < end; i++) if (counts[i].detections > best.detections) best = counts[i];
    points.push({ x: counts[start].timestamp_seconds, y: best.detections });
  }
  return points;
}
