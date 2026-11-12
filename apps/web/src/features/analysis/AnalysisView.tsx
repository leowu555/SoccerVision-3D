"use client";

import { useRouter } from "next/navigation";
import { useCallback, useState } from "react";
import { Alert, Button, ButtonLink, Card, Skeleton, Spinner, StatusBadge } from "@/components/ui/primitives";
import { DetectionStats } from "@/features/analysis/DetectionStats";
import { DetectionTimeline } from "@/features/analysis/DetectionTimeline";
import { JobStatusCard } from "@/features/analysis/JobStatusCard";
import { RunDetails } from "@/features/analysis/RunDetails";
import { VideoComparison } from "@/features/analysis/VideoComparison";
import { VideoMetadataCard } from "@/features/analysis/VideoMetadataCard";
import { AnalysisOptions, type AnalysisOptionsValue } from "@/features/upload/AnalysisOptions";
import { usePolling } from "@/hooks/usePolling";
import { api, ApiClientError } from "@/lib/api";
import { formatRelativeTime } from "@/lib/format";
import { isActiveJob } from "@/lib/stats";
import type { Results } from "@/types/api";

async function fetchResultsOrNull(videoId: string): Promise<Results | null> {
  try {
    return await api.getResults(videoId);
  } catch (err) {
    if (err instanceof ApiClientError && err.code === "results_not_found") return null;
    throw err;
  }
}

export function AnalysisView({ videoId }: { videoId: string }) {
  const video = usePolling(`video:${videoId}`, () => api.getVideo(videoId), {
    intervalMs: 1000,
    shouldPoll: (v) => isActiveJob(v.latest_job?.status),
  });
  const latestJob = video.data?.latest_job ?? null;
  // Refetch results whenever the latest job reaches a new terminal state.
  const resultsKey = `results:${videoId}:${latestJob?.job_id ?? "none"}:${isActiveJob(latestJob?.status) ? "active" : latestJob?.status}`;
  const results = usePolling(resultsKey, () => fetchResultsOrNull(videoId));

  const [currentTime, setCurrentTime] = useState(0);
  const [seekRequest, setSeekRequest] = useState<{ time: number; nonce: number }>();
  const [showRerun, setShowRerun] = useState(false);
  const [rerunOptions, setRerunOptions] = useState<AnalysisOptionsValue>({});
  const [submitting, setSubmitting] = useState(false);
  const [actionError, setActionError] = useState<string>();
  const router = useRouter();

  const onSeek = useCallback((time: number) => setSeekRequest({ time, nonce: Date.now() }), []);

  async function remove() {
    if (!window.confirm("Delete this video, its analysis runs and all generated artifacts?")) return;
    setSubmitting(true);
    setActionError(undefined);
    try {
      await api.deleteVideo(videoId);
      router.push("/");
    } catch (err) {
      setActionError(err instanceof ApiClientError ? err.message : "Could not delete the video.");
      setSubmitting(false);
    }
  }

  async function rerun() {
    setSubmitting(true);
    setActionError(undefined);
    try {
      await api.analyze(videoId, rerunOptions);
      setShowRerun(false);
      await video.refresh();
    } catch (err) {
      setActionError(err instanceof ApiClientError ? err.message : "Could not start analysis.");
    } finally {
      setSubmitting(false);
    }
  }

  if (video.error && !video.data) {
    const notFound = video.error instanceof ApiClientError && video.error.status === 404;
    return (
      <div className="mx-auto max-w-xl space-y-4">
        <Alert title={notFound ? "Video not found" : "Could not load video"}>{video.error.message}</Alert>
        <ButtonLink href="/" variant="secondary">
          Back to dashboard
        </ButtonLink>
      </div>
    );
  }
  if (!video.data) {
    return <Skeleton className="h-96 w-full" />;
  }

  const v = video.data;
  const result = results.data ?? null;
  const active = isActiveJob(latestJob?.status);
  const showingOlderRun = result && latestJob && result.job.job_id !== latestJob.job_id;

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <div className="min-w-0">
          <div className="flex items-center gap-3">
            <h1 className="truncate text-2xl font-semibold tracking-tight text-slate-50">{v.filename}</h1>
            <StatusBadge status={v.processing_status} />
          </div>
          <p className="mt-1 font-mono text-xs text-slate-500">
            video {v.video_id} · uploaded {formatRelativeTime(v.created_at)}
          </p>
        </div>
        <div className="flex gap-2">
          {latestJob === null ? (
            <Button onClick={rerun} disabled={submitting}>
              {submitting ? <Spinner /> : null}
              Analyze
            </Button>
          ) : !active ? (
            <Button variant="secondary" onClick={() => setShowRerun((s) => !s)}>
              {showRerun ? "Close" : "Run again"}
            </Button>
          ) : null}
          {!active ? (
            <Button variant="ghost" onClick={remove} disabled={submitting}>
              Delete
            </Button>
          ) : null}
        </div>
      </div>

      {actionError ? <Alert title="Request failed">{actionError}</Alert> : null}

      {showRerun && !active ? (
        <Card>
          <AnalysisOptions onChange={setRerunOptions} />
          <Button className="mt-4" onClick={rerun} disabled={submitting}>
            {submitting ? <Spinner /> : null}
            Start new analysis
          </Button>
        </Card>
      ) : null}

      {latestJob && (active || latestJob.status === "failed") ? <JobStatusCard job={latestJob} video={v} /> : null}

      {showingOlderRun ? (
        <Alert tone="warning" title="Showing an earlier run">
          The latest job did not complete. Results below are from job {result.job.job_id.slice(0, 8)}, finished{" "}
          {result.job.completed_at ? formatRelativeTime(result.job.completed_at) : "earlier"}.
        </Alert>
      ) : null}

      {results.error ? <Alert title="Could not load results">{results.error.message}</Alert> : null}

      <VideoComparison
        video={v}
        results={result}
        onTimeUpdate={setCurrentTime}
        seekRequest={seekRequest}
      />

      {result ? (
        <>
          <DetectionStats video={v} run={result.run} />
          <DetectionTimeline
            counts={result.frame_counts}
            duration={v.duration_seconds}
            currentTime={currentTime}
            stride={result.run.configuration.inference_stride}
            onSeek={onSeek}
          />
          <div className="grid gap-4 lg:grid-cols-2">
            <VideoMetadataCard video={v} />
            <RunDetails results={result} />
          </div>
        </>
      ) : (
        <VideoMetadataCard video={v} />
      )}
    </div>
  );
}
