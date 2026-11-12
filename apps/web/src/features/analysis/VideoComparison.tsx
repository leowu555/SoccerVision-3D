"use client";

import { useEffect, useRef, useState } from "react";
import { Badge, Card, CardHeader } from "@/components/ui/primitives";
import { apiUrl } from "@/lib/api";
import type { Results, Video } from "@/types/api";

interface Props {
  video: Video;
  results: Results | null;
  onTimeUpdate: (time: number) => void;
  seekRequest?: { time: number; nonce: number };
}

const SYNC_TOLERANCE_SECONDS = 0.15;

export function VideoComparison({ video, results, onTimeUpdate, seekRequest }: Props) {
  const originalRef = useRef<HTMLVideoElement>(null);
  const annotatedRef = useRef<HTMLVideoElement>(null);
  const [synced, setSynced] = useState(true);
  const [originalError, setOriginalError] = useState(false);
  const [annotatedError, setAnnotatedError] = useState(false);

  const annotatedUrl = results?.job.artifact_urls.annotated_video;
  const output = results?.run.output_video;

  useEffect(() => {
    if (!seekRequest) return;
    const target = annotatedRef.current ?? originalRef.current;
    if (target) target.currentTime = seekRequest.time;
  }, [seekRequest]);

  useEffect(() => {
    const leader = annotatedRef.current;
    const follower = originalRef.current;
    if (!leader) return;

    const report = () => onTimeUpdate(leader.currentTime);
    const align = () => {
      if (!synced || !follower) return;
      if (Math.abs(follower.currentTime - leader.currentTime) > SYNC_TOLERANCE_SECONDS) {
        follower.currentTime = leader.currentTime;
      }
    };
    const onPlay = () => {
      align();
      if (synced && follower) void follower.play().catch(() => undefined);
    };
    const onPause = () => {
      if (synced && follower) follower.pause();
      align();
    };
    const onRate = () => {
      if (synced && follower) follower.playbackRate = leader.playbackRate;
    };

    leader.addEventListener("timeupdate", report);
    leader.addEventListener("timeupdate", align);
    leader.addEventListener("seeked", align);
    leader.addEventListener("seeked", report);
    leader.addEventListener("play", onPlay);
    leader.addEventListener("pause", onPause);
    leader.addEventListener("ratechange", onRate);
    return () => {
      leader.removeEventListener("timeupdate", report);
      leader.removeEventListener("timeupdate", align);
      leader.removeEventListener("seeked", align);
      leader.removeEventListener("seeked", report);
      leader.removeEventListener("play", onPlay);
      leader.removeEventListener("pause", onPause);
      leader.removeEventListener("ratechange", onRate);
    };
  }, [synced, onTimeUpdate, annotatedUrl]);

  return (
    <Card>
      <CardHeader
        title="Playback"
        subtitle="Original upload and model-annotated output. Boxes are drawn only on analyzed frames."
        action={
          annotatedUrl ? (
            <label className="flex items-center gap-2 text-xs text-slate-400">
              <input
                type="checkbox"
                checked={synced}
                onChange={(e) => setSynced(e.target.checked)}
                className="accent-cyan-400"
              />
              Sync original to annotated
            </label>
          ) : null
        }
      />
      <div className="grid gap-4 lg:grid-cols-2">
        <figure>
          <video
            ref={originalRef}
            src={apiUrl(video.source_url)}
            controls
            muted
            playsInline
            preload="metadata"
            onError={() => setOriginalError(true)}
            className="aspect-video w-full rounded-lg"
          />
          <figcaption className="mt-2 flex items-center justify-between text-xs text-slate-500">
            <span>Original · {video.video_codec}</span>
            {originalError ? <Badge tone="warning">Browser cannot play this codec</Badge> : null}
          </figcaption>
        </figure>
        <figure>
          {annotatedUrl ? (
            <video
              ref={annotatedRef}
              src={apiUrl(annotatedUrl)}
              controls
              playsInline
              preload="metadata"
              onError={() => setAnnotatedError(true)}
              className="aspect-video w-full rounded-lg"
            />
          ) : (
            <div className="flex aspect-video w-full items-center justify-center rounded-lg border border-dashed border-slate-800 bg-slate-950 text-sm text-slate-500">
              Annotated video appears when an analysis completes.
            </div>
          )}
          <figcaption className="mt-2 flex items-center justify-between text-xs text-slate-500">
            <span>
              Annotated{output ? ` · ${output.codec}` : ""}
              {results?.run.model ? ` · ${results.run.model.model_id}` : ""}
            </span>
            {annotatedError || (output && !output.browser_playable) ? (
              <Badge tone="warning">Not browser-playable; download the file</Badge>
            ) : results?.run.model?.is_synthetic ? (
              <Badge tone="danger">Synthetic test detector</Badge>
            ) : null}
          </figcaption>
        </figure>
      </div>
    </Card>
  );
}
