"use client";

import Link from "next/link";
import { Alert, ButtonLink, Card, CardHeader, ProgressBar, Skeleton, StatusBadge } from "@/components/ui/primitives";
import { usePolling } from "@/hooks/usePolling";
import { api } from "@/lib/api";
import { formatBytes, formatDuration, formatPercent, formatRelativeTime, formatResolution } from "@/lib/format";
import { isActiveVideo } from "@/lib/stats";
import type { Video } from "@/types/api";

function VideoRow({ video }: { video: Video }) {
  const job = video.latest_job;
  const active = isActiveVideo(video.processing_status);
  return (
    <tr className="border-t border-slate-800/80 hover:bg-slate-800/30">
      <td className="max-w-xs py-3 pr-4">
        <Link href={`/videos/${video.video_id}`} className="block truncate font-medium text-slate-100 hover:text-cyan-300">
          {video.filename}
        </Link>
        <span className="font-mono text-xs text-slate-500">{video.video_id.slice(0, 8)}</span>
      </td>
      <td className="py-3 pr-4 font-mono text-xs text-slate-400">{formatDuration(video.duration_seconds)}</td>
      <td className="hidden py-3 pr-4 font-mono text-xs text-slate-400 md:table-cell">
        {formatResolution(video.width, video.height)} · {video.fps.toFixed(2)} fps
      </td>
      <td className="hidden py-3 pr-4 font-mono text-xs text-slate-400 lg:table-cell">{formatBytes(video.size_bytes)}</td>
      <td className="py-3 pr-4">
        <div className="flex flex-col gap-1.5">
          <StatusBadge status={video.processing_status} />
          {active && job ? (
            <div className="flex w-32 items-center gap-2">
              <ProgressBar value={job.progress} label="Analysis progress" />
              <span className="font-mono text-[10px] text-slate-500">{formatPercent(job.progress)}</span>
            </div>
          ) : null}
        </div>
      </td>
      <td className="py-3 text-right text-xs text-slate-500">{formatRelativeTime(video.created_at)}</td>
    </tr>
  );
}

export function RecentVideos() {
  const { data, error, loading } = usePolling("videos", () => api.listVideos(25), {
    intervalMs: 2000,
    shouldPoll: (videos) => videos.some((v) => isActiveVideo(v.processing_status)),
  });

  return (
    <Card>
      <CardHeader title="Recent analyses" subtitle="Uploaded videos and the status of their latest analysis job." />
      {error && !data ? (
        <Alert title="Could not load videos">{error.message}</Alert>
      ) : loading && !data ? (
        <div className="space-y-2">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-12 w-full" />
          ))}
        </div>
      ) : data && data.length === 0 ? (
        <div className="flex flex-col items-center gap-3 rounded-lg border border-dashed border-slate-800 py-12 text-center">
          <p className="text-sm text-slate-400">No videos yet.</p>
          <ButtonLink href="/upload" variant="secondary">
            Upload your first clip
          </ButtonLink>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-xs uppercase tracking-wider text-slate-500">
              <tr>
                <th className="pb-2 font-medium">Video</th>
                <th className="pb-2 font-medium">Duration</th>
                <th className="hidden pb-2 font-medium md:table-cell">Format</th>
                <th className="hidden pb-2 font-medium lg:table-cell">Size</th>
                <th className="pb-2 font-medium">Status</th>
                <th className="pb-2 text-right font-medium">Uploaded</th>
              </tr>
            </thead>
            <tbody>{data?.map((video) => <VideoRow key={video.video_id} video={video} />)}</tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
