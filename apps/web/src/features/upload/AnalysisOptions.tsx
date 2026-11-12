"use client";

import { useEffect, useState } from "react";
import type { AnalyzeRequest, DetectorName, DeviceName, Health } from "@/types/api";

export type AnalysisOptionsValue = AnalyzeRequest;

const DETECTORS: Array<{ value: DetectorName; label: string }> = [
  { value: "rfdetr-nano", label: "RF-DETR Nano (fastest)" },
  { value: "rfdetr-small", label: "RF-DETR Small" },
  { value: "rfdetr-medium", label: "RF-DETR Medium (most accurate)" },
];
const DEVICES: DeviceName[] = ["auto", "cpu", "mps", "cuda"];
const STRIDES = [1, 2, 3, 5, 10];

const fieldClass =
  "mt-1 w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 focus:border-cyan-400 focus:outline-none";

export function AnalysisOptions({
  defaults,
  onChange,
}: {
  defaults?: Health["defaults"];
  onChange: (value: AnalysisOptionsValue) => void;
}) {
  const [detector, setDetector] = useState<DetectorName>(defaults?.detector ?? "rfdetr-nano");
  const [device, setDevice] = useState<DeviceName>(defaults?.device ?? "auto");
  const [confidence, setConfidence] = useState(defaults?.confidence_threshold ?? 0.5);
  const [stride, setStride] = useState(defaults?.inference_stride ?? 1);

  useEffect(() => {
    onChange({ detector, device, confidence_threshold: confidence, inference_stride: stride });
  }, [detector, device, confidence, stride, onChange]);

  return (
    <fieldset className="space-y-4">
      <legend className="text-sm font-semibold text-slate-100">Analysis settings</legend>
      <label className="block text-xs text-slate-400">
        Detector
        <select className={fieldClass} value={detector} onChange={(e) => setDetector(e.target.value as DetectorName)}>
          {DETECTORS.map((d) => (
            <option key={d.value} value={d.value}>
              {d.label}
            </option>
          ))}
        </select>
      </label>
      <label className="block text-xs text-slate-400">
        Confidence threshold · <span className="font-mono text-slate-200">{confidence.toFixed(2)}</span>
        <input
          type="range"
          min={0.1}
          max={0.9}
          step={0.05}
          value={confidence}
          onChange={(e) => setConfidence(Number(e.target.value))}
          className="mt-2 w-full accent-cyan-400"
        />
      </label>
      <div className="grid grid-cols-2 gap-3">
        <label className="block text-xs text-slate-400">
          Analyze every
          <select className={fieldClass} value={stride} onChange={(e) => setStride(Number(e.target.value))}>
            {STRIDES.map((s) => (
              <option key={s} value={s}>
                {s === 1 ? "frame" : `${s} frames`}
              </option>
            ))}
          </select>
        </label>
        <label className="block text-xs text-slate-400">
          Device
          <select className={fieldClass} value={device} onChange={(e) => setDevice(e.target.value as DeviceName)}>
            {DEVICES.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </label>
      </div>
      {stride > 1 ? (
        <p className="text-xs text-amber-300/80">
          Frames between analyzed frames are marked “not analyzed” in the output; no boxes are drawn on them.
        </p>
      ) : null}
    </fieldset>
  );
}
