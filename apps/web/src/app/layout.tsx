import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { AppHeader } from "@/components/layout/AppHeader";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: { default: "SoccerVision 3D", template: "%s · SoccerVision 3D" },
  description: "Soccer video analysis: pretrained person detection with reproducible artifacts.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col font-sans">
        <AppHeader />
        <main className="mx-auto w-full max-w-7xl flex-1 px-6 py-8">{children}</main>
        <footer className="border-t border-slate-900 px-6 py-4 text-center text-xs text-slate-600">
          SoccerVision 3D · Phase 1 · Detections are generic COCO “person” boxes from a pretrained model.
        </footer>
      </body>
    </html>
  );
}
