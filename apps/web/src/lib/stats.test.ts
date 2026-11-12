import { describe, expect, it } from "vitest";
import { analysisCoverage, downsampleMax, frameCountAt, isActiveJob } from "@/lib/stats";
import type { FrameCount, RunMetadata } from "@/types/api";

const counts: FrameCount[] = [0, 3, 6, 9].map((frame, i) => ({
  frame_index: frame,
  timestamp_seconds: frame / 30,
  detections: [4, 7, 2, 5][i],
}));

describe("frameCountAt", () => {
  it("returns the latest analyzed frame at or before t", () => {
    expect(frameCountAt(counts, 0)?.frame_index).toBe(0);
    expect(frameCountAt(counts, 0.11)?.frame_index).toBe(3);
    expect(frameCountAt(counts, 6 / 30)?.frame_index).toBe(6);
    expect(frameCountAt(counts, 99)?.frame_index).toBe(9);
  });

  it("returns null before the first analyzed frame or with no data", () => {
    expect(frameCountAt(counts.slice(1), 0.01)).toBeNull();
    expect(frameCountAt([], 1)).toBeNull();
  });
});

describe("downsampleMax", () => {
  it("keeps all points when under the limit", () => {
    expect(downsampleMax(counts, 10)).toHaveLength(4);
  });

  it("preserves bucket peaks", () => {
    const points = downsampleMax(counts, 2);
    expect(points.map((p) => p.y)).toEqual([7, 5]);
  });
});

describe("analysisCoverage", () => {
  it("is analyzed / decoded", () => {
    const run = { stats: { decoded_frames: 12, analyzed_frames: 4 } } as RunMetadata;
    expect(analysisCoverage(run)).toBeCloseTo(1 / 3);
    expect(analysisCoverage({ stats: null } as RunMetadata)).toBeNull();
  });
});

describe("isActiveJob", () => {
  it("treats pending and processing as active", () => {
    expect(isActiveJob("pending")).toBe(true);
    expect(isActiveJob("processing")).toBe(true);
    expect(isActiveJob("completed")).toBe(false);
    expect(isActiveJob(null)).toBe(false);
  });
});
