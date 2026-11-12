import { describe, expect, it } from "vitest";
import { formatBytes, formatDuration, formatNumber, formatPercent, formatRelativeTime } from "@/lib/format";

describe("formatDuration", () => {
  it.each([
    [0, "0:00.0"],
    [13.966, "0:14.0"],
    [65.25, "1:05.3"],
    [3725, "1:02:05.0"],
  ])("%s -> %s", (input, expected) => {
    expect(formatDuration(input)).toBe(expected);
  });

  it("handles missing values", () => {
    expect(formatDuration(null)).toBe("—");
    expect(formatDuration(Number.NaN)).toBe("—");
    expect(formatDuration(-1)).toBe("—");
  });
});

describe("formatBytes", () => {
  it("scales units", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(formatBytes(500 * 1024 * 1024)).toBe("500.0 MB");
  });
});

describe("formatPercent", () => {
  it("clamps and rounds", () => {
    expect(formatPercent(0.456)).toBe("46%");
    expect(formatPercent(1.2)).toBe("100%");
    expect(formatPercent(0.4567, 1)).toBe("45.7%");
    expect(formatPercent(undefined)).toBe("—");
  });
});

describe("formatNumber", () => {
  it("formats with grouping", () => {
    expect(formatNumber(3520)).toBe("3,520");
    expect(formatNumber(8.4009, 1)).toBe("8.4");
    expect(formatNumber(null)).toBe("—");
  });
});

describe("formatRelativeTime", () => {
  const now = new Date("2026-11-05T12:00:00Z");
  it("describes recent times", () => {
    expect(formatRelativeTime("2026-11-05T11:59:50Z", now)).toBe("just now");
    expect(formatRelativeTime("2026-11-05T11:50:00Z", now)).toBe("10 min ago");
    expect(formatRelativeTime("2026-11-05T09:00:00Z", now)).toBe("3 h ago");
    expect(formatRelativeTime("2026-11-02T12:00:00Z", now)).toBe("3 d ago");
  });
});
