import { Card, CardHeader, DefinitionList } from "@/components/ui/primitives";
import { formatBytes, formatDuration, formatNumber, formatResolution } from "@/lib/format";
import type { Video } from "@/types/api";

export function VideoMetadataCard({ video }: { video: Video }) {
  return (
    <Card>
      <CardHeader title="Source video" subtitle="Extracted from the media stream at upload." />
      <DefinitionList
        items={[
          ["Duration", formatDuration(video.duration_seconds)],
          ["Resolution", formatResolution(video.width, video.height)],
          ["Frame rate", `${video.fps.toFixed(3)} fps · ${video.frame_rate_mode}`],
          ["Frames", video.frame_count ? formatNumber(video.frame_count) : "unknown"],
          ["Codec", video.video_codec],
          ["Container", video.container_format],
          ["Audio", video.has_audio ? "yes (not used)" : "none"],
          ["Size", formatBytes(video.size_bytes)],
        ]}
      />
    </Card>
  );
}
