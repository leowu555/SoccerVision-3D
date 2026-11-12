import type { Metadata } from "next";
import { Suspense } from "react";
import { Skeleton } from "@/components/ui/primitives";
import { AnalysisView } from "@/features/analysis/AnalysisView";

export const metadata: Metadata = { title: "Analysis" };

function AnalysisSkeleton() {
  return (
    <div className="space-y-6">
      <Skeleton className="h-8 w-72" />
      <div className="grid gap-4 lg:grid-cols-2">
        <Skeleton className="aspect-video w-full" />
        <Skeleton className="aspect-video w-full" />
      </div>
    </div>
  );
}

export default function AnalysisPage({ params }: PageProps<"/videos/[id]">) {
  return (
    <Suspense fallback={<AnalysisSkeleton />}>
      {params.then(({ id }) => (
        <AnalysisView videoId={id} />
      ))}
    </Suspense>
  );
}
