"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef, useState, type DragEvent } from "react";
import { Alert, Button, Card, CardHeader, DefinitionList, ProgressBar, Spinner } from "@/components/ui/primitives";
import { AnalysisOptions, type AnalysisOptionsValue } from "@/features/upload/AnalysisOptions";
import { usePolling } from "@/hooks/usePolling";
import { api, ApiClientError, uploadVideo, type UploadHandle } from "@/lib/api";
import { formatBytes, formatDuration, formatPercent, formatResolution } from "@/lib/format";
import { validateVideoFile } from "@/lib/validation";
import type { Video } from "@/types/api";

const FALLBACK_MAX_BYTES = 500 * 1024 * 1024;

type Phase =
  | { kind: "idle" }
  | { kind: "selected"; file: File }
  | { kind: "uploading"; file: File; progress: number }
  | { kind: "uploaded"; file: File; video: Video }
  | { kind: "submitting"; file: File; video: Video };

export function UploadForm() {
  const router = useRouter();
  const health = usePolling("health", api.health);
  const maxBytes = health.data?.limits.max_upload_bytes ?? FALLBACK_MAX_BYTES;
  const maxDuration = health.data?.limits.max_video_duration_seconds;

  const [phase, setPhase] = useState<Phase>({ kind: "idle" });
  const [error, setError] = useState<string>();
  const [dragging, setDragging] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string>();
  const [options, setOptions] = useState<AnalysisOptionsValue>();
  const inputRef = useRef<HTMLInputElement>(null);
  const uploadRef = useRef<UploadHandle | null>(null);
  const previewUrlRef = useRef<string | undefined>(undefined);

  const file = phase.kind === "idle" ? undefined : phase.file;

  function replacePreview(next: File | undefined) {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = next ? URL.createObjectURL(next) : undefined;
    setPreviewUrl(previewUrlRef.current);
  }

  useEffect(
    () => () => {
      uploadRef.current?.abort();
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    },
    [],
  );

  function selectFile(candidate: File | undefined) {
    if (!candidate) return;
    setError(undefined);
    const check = validateVideoFile(candidate, maxBytes);
    if (!check.ok) {
      setError(check.reason);
      return;
    }
    replacePreview(candidate);
    setPhase({ kind: "selected", file: candidate });
  }

  async function startUpload() {
    if (phase.kind !== "selected") return;
    const selected = phase.file;
    setError(undefined);
    setPhase({ kind: "uploading", file: selected, progress: 0 });
    const handle = uploadVideo(selected, (progress) =>
      setPhase((current) => (current.kind === "uploading" ? { ...current, progress } : current)),
    );
    uploadRef.current = handle;
    try {
      const response = await handle.promise;
      setPhase({ kind: "uploaded", file: selected, video: response.metadata });
    } catch (err) {
      setError(err instanceof ApiClientError ? err.message : "Upload failed.");
      setPhase({ kind: "selected", file: selected });
    } finally {
      uploadRef.current = null;
    }
  }

  async function startAnalysis() {
    if (phase.kind !== "uploaded") return;
    setError(undefined);
    setPhase({ kind: "submitting", file: phase.file, video: phase.video });
    try {
      await api.analyze(phase.video.video_id, options ?? {});
      router.push(`/videos/${phase.video.video_id}`);
    } catch (err) {
      setError(err instanceof ApiClientError ? err.message : "Could not start analysis.");
      setPhase({ kind: "uploaded", file: phase.file, video: phase.video });
    }
  }

  function reset() {
    uploadRef.current?.abort();
    setPhase({ kind: "idle" });
    replacePreview(undefined);
    setError(undefined);
    if (inputRef.current) inputRef.current.value = "";
  }

  function onDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setDragging(false);
    selectFile(event.dataTransfer.files[0]);
  }

  const video = phase.kind === "uploaded" || phase.kind === "submitting" ? phase.video : undefined;

  return (
    <div className="space-y-6">
      {error ? <Alert title="Something went wrong">{error}</Alert> : null}

      {phase.kind === "idle" ? (
        <label
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className={`flex cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed px-6 py-16 text-center transition-colors ${
            dragging ? "border-cyan-400 bg-cyan-400/5" : "border-slate-700 bg-slate-900/40 hover:border-slate-500"
          }`}
        >
          <svg viewBox="0 0 24 24" className="size-10 text-slate-500" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden>
            <path d="M12 16V4m0 0-4 4m4-4 4 4M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
          </svg>
          <span className="text-sm text-slate-200">Drag an MP4 here, or click to browse</span>
          <span className="text-xs text-slate-500">
            Up to {formatBytes(maxBytes)}
            {maxDuration ? ` · max ${formatDuration(maxDuration)} long` : ""}
          </span>
          <input
            ref={inputRef}
            type="file"
            accept="video/mp4,.mp4"
            className="sr-only"
            onChange={(e) => selectFile(e.target.files?.[0])}
          />
        </label>
      ) : (
        <div className="grid gap-6 lg:grid-cols-[3fr_2fr]">
          <Card className="p-3">
            {previewUrl ? (
              <video src={previewUrl} controls muted playsInline className="aspect-video w-full rounded-lg" />
            ) : null}
            <p className="mt-2 truncate px-1 text-xs text-slate-500">
              Local preview · {file?.name} · {formatBytes(file?.size)}
            </p>
          </Card>

          <Card>
            {video ? (
              <>
                <CardHeader title="Validated by server" subtitle="Metadata extracted from the media stream." />
                <DefinitionList
                  items={[
                    ["Duration", formatDuration(video.duration_seconds)],
                    ["Resolution", formatResolution(video.width, video.height)],
                    ["Frame rate", `${video.fps.toFixed(3)} fps (${video.frame_rate_mode})`],
                    ["Frames", video.frame_count ?? "unknown"],
                    ["Codec", video.video_codec],
                  ]}
                />
                <div className="mt-6 border-t border-slate-800 pt-5">
                  <AnalysisOptions
                    key={health.data ? "server-defaults" : "fallback-defaults"}
                    defaults={health.data?.defaults}
                    onChange={setOptions}
                  />
                </div>
                <div className="mt-6 flex gap-3">
                  <Button onClick={startAnalysis} disabled={phase.kind === "submitting"}>
                    {phase.kind === "submitting" ? <Spinner /> : null}
                    Analyze video
                  </Button>
                  <Button variant="ghost" onClick={reset} disabled={phase.kind === "submitting"}>
                    Choose another
                  </Button>
                </div>
              </>
            ) : (
              <>
                <CardHeader title="Ready to upload" subtitle="The server inspects the actual media before accepting it." />
                {phase.kind === "uploading" ? (
                  <div className="space-y-2">
                    <ProgressBar value={phase.progress} label="Upload progress" />
                    <p className="font-mono text-xs text-slate-400">
                      {phase.progress >= 1 ? "Validating media on server…" : `Uploading ${formatPercent(phase.progress)}`}
                    </p>
                  </div>
                ) : null}
                <div className="mt-6 flex gap-3">
                  <Button onClick={startUpload} disabled={phase.kind === "uploading"}>
                    {phase.kind === "uploading" ? <Spinner /> : null}
                    Upload
                  </Button>
                  <Button variant="ghost" onClick={reset}>
                    {phase.kind === "uploading" ? "Cancel" : "Choose another"}
                  </Button>
                </div>
              </>
            )}
          </Card>
        </div>
      )}
    </div>
  );
}
