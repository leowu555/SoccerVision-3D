import { Alert, Card, CardHeader, DefinitionList } from "@/components/ui/primitives";
import { apiUrl } from "@/lib/api";
import { formatNumber, formatRelativeTime } from "@/lib/format";
import type { Results } from "@/types/api";

export function RunDetails({ results }: { results: Results }) {
  const { run, job } = results;
  const model = run.model;
  const detector = run.configuration.detector;
  const links: Array<[string, string | undefined]> = [
    ["annotated.mp4", job.artifact_urls.annotated_video],
    ["detections.json", job.artifact_urls.detections],
    ["metadata.json", job.artifact_urls.metadata],
  ];

  return (
    <Card>
      <CardHeader
        title="Run details"
        subtitle={`Job ${job.job_id.slice(0, 8)} · completed ${job.completed_at ? formatRelativeTime(job.completed_at) : "—"}`}
      />
      {model?.is_synthetic ? (
        <div className="mb-4">
          <Alert title="Synthetic detector">These detections come from a test double, not a trained model.</Alert>
        </div>
      ) : null}
      <DefinitionList
        items={[
          ["Model", model ? `${model.model_id} (${model.license})` : "—"],
          ["Weights MD5", model?.weights_md5 ?? "—"],
          ["Input size", model?.input_resolution ? `${model.input_resolution}px` : "—"],
          ["Device", model?.device ?? "—"],
          ["Classes kept", detector.target_classes.join(", ")],
          ["Threshold", detector.confidence_threshold.toFixed(2)],
          ["Stride", run.configuration.inference_stride === 1 ? "every frame" : `every ${run.configuration.inference_stride} frames`],
          ["Model load", run.timing.model_load_seconds != null ? `${formatNumber(run.timing.model_load_seconds, 1)} s` : "—"],
          ["Pipeline", `${run.pipeline.name} v${run.pipeline.version}`],
          ["Schema", `v${run.schema_version}`],
        ]}
      />
      {run.warnings.length ? (
        <ul className="mt-4 space-y-1 text-xs text-amber-300/90">
          {run.warnings.map((w) => (
            <li key={w}>⚠ {w}</li>
          ))}
        </ul>
      ) : null}
      <div className="mt-5 flex flex-wrap gap-2 border-t border-slate-800 pt-4">
        {links.map(([label, url]) =>
          url ? (
            <a
              key={label}
              href={apiUrl(url)}
              target="_blank"
              rel="noreferrer"
              className="rounded-md border border-slate-700 px-3 py-1.5 font-mono text-xs text-slate-300 hover:border-cyan-400 hover:text-cyan-300"
            >
              {label}
            </a>
          ) : null,
        )}
      </div>
    </Card>
  );
}
