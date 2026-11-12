import { describe, expect, it, vi } from "vitest";
import { __testing, ApiClientError, apiUrl, toApiError } from "@/lib/api";

describe("apiUrl", () => {
  it("joins relative API paths with the base", () => {
    expect(apiUrl("/api/v1/health", "http://api:8000")).toBe("http://api:8000/api/v1/health");
    expect(apiUrl("api/v1/health", "http://api:8000")).toBe("http://api:8000/api/v1/health");
  });

  it("passes absolute URLs through", () => {
    expect(apiUrl("https://cdn.example/x.mp4", "http://api:8000")).toBe("https://cdn.example/x.mp4");
  });
});

describe("toApiError", () => {
  it("uses the structured error body", () => {
    const err = toApiError(422, { error: { code: "invalid_video", message: "Not an MP4", details: { a: 1 } } });
    expect(err).toBeInstanceOf(ApiClientError);
    expect(err.status).toBe(422);
    expect(err.code).toBe("invalid_video");
    expect(err.message).toBe("Not an MP4");
    expect(err.details).toEqual({ a: 1 });
  });

  it("falls back for unstructured bodies", () => {
    const err = toApiError(502, "<html>bad gateway</html>");
    expect(err.code).toBe("http_error");
    expect(err.message).toContain("502");
  });
});

describe("request", () => {
  it("returns parsed JSON on success", async () => {
    const fetchImpl = vi.fn(async () => new Response(JSON.stringify({ ok: 1 }), { status: 200 }));
    await expect(__testing.request<{ ok: number }>("/x", undefined, fetchImpl)).resolves.toEqual({ ok: 1 });
    expect(fetchImpl).toHaveBeenCalledOnce();
  });

  it("raises ApiClientError with server code on failure", async () => {
    const body = { error: { code: "job_already_active", message: "busy" } };
    const fetchImpl = vi.fn(async () => new Response(JSON.stringify(body), { status: 409 }));
    await expect(__testing.request("/x", undefined, fetchImpl)).rejects.toMatchObject({
      status: 409,
      code: "job_already_active",
    });
  });

  it("reports network failures distinctly", async () => {
    const fetchImpl = vi.fn(async () => {
      throw new TypeError("fetch failed");
    });
    await expect(__testing.request("/x", undefined, fetchImpl)).rejects.toMatchObject({ code: "network_error" });
  });
});
