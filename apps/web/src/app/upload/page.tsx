import type { Metadata } from "next";
import { UploadForm } from "@/features/upload/UploadForm";

export const metadata: Metadata = { title: "Upload" };

export default function UploadPage() {
  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-slate-50">Upload a video</h1>
        <p className="mt-2 text-sm text-slate-400">
          MP4 only. Use footage you own or are licensed to analyze. The server validates the actual media content,
          extracts metadata, and stores the file privately.
        </p>
      </div>
      <UploadForm />
    </div>
  );
}
