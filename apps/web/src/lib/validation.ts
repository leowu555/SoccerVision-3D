export interface FileLike {
  name: string;
  size: number;
  type: string;
}

export type FileValidation = { ok: true } | { ok: false; reason: string };

/**
 * Client-side pre-check only, for fast feedback. The server re-validates the actual media
 * content (container signature, demuxer, decodable frames, duration).
 */
export function validateVideoFile(file: FileLike, maxBytes: number): FileValidation {
  const isMp4Name = file.name.toLowerCase().endsWith(".mp4");
  const typeOk = file.type === "" || file.type === "video/mp4" || file.type === "application/octet-stream";
  if (!isMp4Name || !typeOk) return { ok: false, reason: "Only MP4 video files are supported." };
  if (file.size === 0) return { ok: false, reason: "The file is empty." };
  if (file.size > maxBytes) {
    const limitMiB = Math.floor(maxBytes / (1024 * 1024));
    return { ok: false, reason: `The file exceeds the ${limitMiB} MiB upload limit.` };
  }
  return { ok: true };
}
