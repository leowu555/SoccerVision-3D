import { ButtonLink, Card } from "@/components/ui/primitives";
import { RecentVideos } from "@/features/videos/RecentVideos";

const IMPLEMENTED = [
  "MP4 upload with server-side media validation",
  "Pretrained RF-DETR person detection (COCO “person” class)",
  "Annotated H.264 output with per-frame analysis status",
  "Versioned detection + run-metadata artifacts",
];

const PLANNED = [
  "Multi-object tracking",
  "Team classification",
  "Pitch calibration & 2D tactical view",
  "Ball tracking & events",
  "3D scene replay",
];

export default function DashboardPage() {
  return (
    <div className="space-y-8">
      <section className="flex flex-col gap-6 md:flex-row md:items-end md:justify-between">
        <div className="max-w-2xl">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">Phase 1 · Detection</p>
          <h1 className="mt-2 text-3xl font-semibold tracking-tight text-slate-50">Soccer video analysis</h1>
          <p className="mt-3 text-sm leading-relaxed text-slate-400">
            Upload broadcast-style footage and run a pretrained detector on every frame. Results are generic person
            detections: no player identities, roles, teams, or pitch coordinates yet.
          </p>
        </div>
        <ButtonLink href="/upload" className="self-start md:self-auto">
          Upload video
        </ButtonLink>
      </section>

      <RecentVideos />

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <h2 className="text-sm font-semibold text-slate-100">Available now</h2>
          <ul className="mt-3 space-y-2 text-sm text-slate-400">
            {IMPLEMENTED.map((item) => (
              <li key={item} className="flex gap-2">
                <span className="mt-1.5 size-1.5 shrink-0 rounded-full bg-emerald-400" aria-hidden />
                {item}
              </li>
            ))}
          </ul>
        </Card>
        <Card>
          <h2 className="text-sm font-semibold text-slate-100">On the roadmap (not implemented)</h2>
          <ul className="mt-3 flex flex-wrap gap-2">
            {PLANNED.map((item) => (
              <li key={item} className="rounded-full border border-dashed border-slate-700 px-3 py-1 text-xs text-slate-500">
                {item}
              </li>
            ))}
          </ul>
        </Card>
      </div>
    </div>
  );
}
