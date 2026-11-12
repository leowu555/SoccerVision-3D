"use client";

import { usePolling } from "@/hooks/usePolling";
import { api } from "@/lib/api";
import type { Health } from "@/types/api";

export function describeAccelerator(health: Health): string {
  if (health.devices.cuda) return "CUDA";
  if (health.devices.mps) return "Apple MPS";
  return "CPU";
}

export function BackendStatus() {
  const { data, error, loading } = usePolling("health", api.health, { intervalMs: 15000, shouldPoll: () => true });

  if (loading && !data) {
    return <span className="text-xs text-slate-500">Checking API…</span>;
  }
  if (error || !data) {
    return (
      <span className="flex items-center gap-2 text-xs text-rose-300" title={error?.message}>
        <span className="size-2 rounded-full bg-rose-400" aria-hidden />
        API offline
      </span>
    );
  }
  const modelReady = data.model.ml_dependencies_installed && (data.model.weights_present || data.model.weights_download_allowed);
  return (
    <span
      className="flex items-center gap-2 text-xs text-slate-400"
      title={`API v${data.version} · pipeline ${data.pipeline_name} v${data.pipeline_version}`}
    >
      <span className={`size-2 rounded-full ${modelReady ? "bg-emerald-400" : "bg-amber-400"}`} aria-hidden />
      <span className="hidden sm:inline">
        {data.model.detector} · {describeAccelerator(data)}
        {!data.model.ml_dependencies_installed ? " · ML deps missing" : ""}
        {data.model.ml_dependencies_installed && !data.model.weights_present ? " · weights not downloaded" : ""}
      </span>
    </span>
  );
}
