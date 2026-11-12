import { Card, CardHeader, Stat } from "@/components/ui/primitives";
import { formatDuration, formatNumber, formatPercent, formatResolution } from "@/lib/format";
import { analysisCoverage } from "@/lib/stats";
import type { RunMetadata, Video } from "@/types/api";

export function DetectionStats({ video, run }: { video: Video; run: RunMetadata }) {
  const stats = run.stats;
  const timing = run.timing;
  const coverage = analysisCoverage(run);
  const threshold = run.configuration.detector.confidence_threshold;

  return (
    <Card>
      <CardHeader
        title="Detection statistics"
        subtitle={`Computed from this run's artifacts. Class “person”, confidence ≥ ${threshold.toFixed(2)}.`}
      />
      <dl className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <Stat label="Video duration" value={formatDuration(video.duration_seconds)} />
        <Stat label="Resolution" value={formatResolution(video.width, video.height)} />
        <Stat
          label="Analyzed frames"
          value={formatNumber(stats?.analyzed_frames)}
          hint={
            stats && stats.skipped_frames > 0
              ? `${formatPercent(coverage)} of ${formatNumber(stats.decoded_frames)} decoded`
              : `all ${formatNumber(stats?.decoded_frames)} decoded frames`
          }
        />
        <Stat
          label="Person detections"
          value={formatNumber(stats?.total_detections)}
          hint="Summed over frames. Not a count of unique people."
        />
        <Stat
          label="Mean per analyzed frame"
          value={formatNumber(stats?.mean_detections_per_analyzed_frame, 1)}
          hint={`Peak ${formatNumber(stats?.max_detections_in_frame)} in one frame`}
        />
        <Stat
          label="Processing time"
          value={`${formatNumber(timing.total_seconds, 1)} s`}
          hint={`${formatNumber(timing.processing_fps, 1)} fps · ${formatNumber(timing.mean_inference_ms, 0)} ms/inference`}
        />
      </dl>
    </Card>
  );
}
