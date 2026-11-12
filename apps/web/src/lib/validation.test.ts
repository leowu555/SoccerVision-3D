import { describe, expect, it } from "vitest";
import { validateVideoFile } from "@/lib/validation";

const MB = 1024 * 1024;

describe("validateVideoFile", () => {
  it("accepts MP4 within limits", () => {
    expect(validateVideoFile({ name: "match.MP4", size: 10 * MB, type: "video/mp4" }, 500 * MB)).toEqual({ ok: true });
    expect(validateVideoFile({ name: "match.mp4", size: 10, type: "" }, 500 * MB)).toEqual({ ok: true });
  });

  it("rejects other formats", () => {
    expect(validateVideoFile({ name: "match.mov", size: 10, type: "video/quicktime" }, MB).ok).toBe(false);
    expect(validateVideoFile({ name: "match.mp4", size: 10, type: "text/plain" }, MB).ok).toBe(false);
  });

  it("rejects empty and oversized files", () => {
    expect(validateVideoFile({ name: "a.mp4", size: 0, type: "video/mp4" }, MB)).toEqual({
      ok: false,
      reason: "The file is empty.",
    });
    const result = validateVideoFile({ name: "a.mp4", size: 2 * MB, type: "video/mp4" }, MB);
    expect(result).toEqual({ ok: false, reason: "The file exceeds the 1 MiB upload limit." });
  });
});
