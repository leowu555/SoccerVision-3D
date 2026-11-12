import { Alert, Card, ProgressBar, Spinner } from "@/components/ui/primitives";
import { formatNumber, formatPercent } from "@/lib/format";
import type { Job, Video } from "@/types/api";

const ERROR_HINTS: Record<string, string> = {
  model_unavailable:
    "The detector could not be loaded. Install ML extras (pip install -e '.[ml]') and download weights (python scripts/download_weights.py).",
  device_unavailable: "The requested compute device is not available on the server. Try device “auto” or “cpu”.",
  interrupted: "The server restarted or stopped while the job was running. Run the analysis again.",
  worker_crashed: "The worker process exited unexpectedly, possibly out of memory. Check worker.log in the run directory.",
  video_limit_exceeded: "The video exceeds the configured maximum duration.",
};

export function JobStatusCard({ job, video }: { job: Job; video: Video }) {
  if (job.status === "failed") {
    return (
      <Alert title={`Analysis failed${job.error_code ? ` · ${job.error_code}` : ""}`}>
        <p>{job.error_message ?? "No error message was recorded."}</p>
        {job.error_code && ERROR_HINTS[job.error_code] ? (
          <p className="mt-2 text-xs text-slate-400">{ERROR_HINTS[job.error_code]}</p>
        ) : null}
      </Alert>
    );
  }

  const processedFrames = video.frame_count ? Math.round(job.progress * video.frame_count) : null;
  return (
    <Card>
      <div className="flex items-center gap-3">
        <Spinner />
        <div className="flex-1">
          <p className="text-sm font-medium text-slate-100">
            {job.status === "pending" ? "Queued, waiting for a worker" : "Running detection"}
          </p>
          <p className="text-xs text-slate-500">
            {job.status === "processing" && job.progress === 0
              ? "Loading model…"
              : processedFrames != null
                ? `${formatNumber(processedFrames)} / ${formatNumber(video.frame_count)} frames decoded`
                : "Processing frames"}
          </p>
        </div>
        <span className="font-mono text-sm tabular-nums text-slate-300">{formatPercent(job.progress)}</span>
      </div>
      <div className="mt-4">
        <ProgressBar value={job.progress} label="Analysis progress" />
      </div>
    </Card>
  );
}
