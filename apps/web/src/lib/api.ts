import type {
  AnalyzeRequest,
  ApiErrorBody,
  Health,
  Job,
  Results,
  UploadResponse,
  Video,
} from "@/types/api";

export const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000").replace(/\/+$/, "");

export class ApiClientError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code: string,
    readonly details?: unknown,
  ) {
    super(message);
    this.name = "ApiClientError";
  }
}

/** Resolve an API-relative URL (e.g. `/api/v1/jobs/…/artifacts/annotated.mp4`) against the API origin. */
export function apiUrl(path: string, base: string = API_BASE_URL): string {
  if (/^https?:\/\//.test(path)) return path;
  return `${base}${path.startsWith("/") ? "" : "/"}${path}`;
}

function isApiErrorBody(value: unknown): value is ApiErrorBody {
  if (typeof value !== "object" || value === null || !("error" in value)) return false;
  const error = (value as { error: unknown }).error;
  return typeof error === "object" && error !== null && "code" in error && "message" in error;
}

export function toApiError(status: number, body: unknown, fallback = "Request failed"): ApiClientError {
  if (isApiErrorBody(body)) {
    return new ApiClientError(body.error.message, status, body.error.code, body.error.details);
  }
  return new ApiClientError(`${fallback} (HTTP ${status})`, status, "http_error");
}

async function request<T>(path: string, init?: RequestInit, fetchImpl: typeof fetch = fetch): Promise<T> {
  let response: Response;
  try {
    response = await fetchImpl(apiUrl(path), {
      ...init,
      headers: { Accept: "application/json", ...init?.headers },
      cache: "no-store",
    });
  } catch {
    throw new ApiClientError(
      `Cannot reach the SoccerVision API at ${API_BASE_URL}. Is the backend running?`,
      0,
      "network_error",
    );
  }
  const text = await response.text();
  let body: unknown = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = null;
    }
  }
  if (!response.ok) throw toApiError(response.status, body);
  return body as T;
}

export const api = {
  health: () => request<Health>("/api/v1/health"),
  listVideos: (limit = 50) => request<{ videos: Video[] }>(`/api/v1/videos?limit=${limit}`).then((r) => r.videos),
  getVideo: (videoId: string) => request<Video>(`/api/v1/videos/${encodeURIComponent(videoId)}`),
  getJob: (jobId: string) => request<Job>(`/api/v1/jobs/${encodeURIComponent(jobId)}`),
  getResults: (videoId: string) => request<Results>(`/api/v1/videos/${encodeURIComponent(videoId)}/results`),
  analyze: (videoId: string, body: AnalyzeRequest = {}) =>
    request<Job>(`/api/v1/videos/${encodeURIComponent(videoId)}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
};

export interface UploadHandle {
  promise: Promise<UploadResponse>;
  abort: () => void;
}

/** Multipart upload via XHR, which (unlike fetch) reports upload progress. */
export function uploadVideo(file: File, onProgress?: (fraction: number) => void): UploadHandle {
  const xhr = new XMLHttpRequest();
  const promise = new Promise<UploadResponse>((resolve, reject) => {
    xhr.open("POST", apiUrl("/api/v1/videos"));
    xhr.responseType = "json";
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) onProgress(event.loaded / event.total);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(xhr.response as UploadResponse);
      else reject(toApiError(xhr.status, xhr.response, "Upload failed"));
    };
    xhr.onerror = () =>
      reject(new ApiClientError(`Upload failed: cannot reach the API at ${API_BASE_URL}.`, 0, "network_error"));
    xhr.onabort = () => reject(new ApiClientError("Upload cancelled.", 0, "aborted"));
    const form = new FormData();
    form.append("file", file, file.name);
    xhr.send(form);
  });
  return { promise, abort: () => xhr.abort() };
}

export const __testing = { request };
