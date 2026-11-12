// Mirrors the FastAPI response models in apps/api/soccervision_api/schemas.py and the
// artifact schemas in packages/soccervision/schemas. Keep in sync with docs/DATA_CONTRACTS.md.

export type JobStatus = "pending" | "processing" | "completed" | "failed";
export type VideoProcessingStatus = "uploaded" | "queued" | "processing" | "completed" | "failed";
export type DetectorName = "rfdetr-nano" | "rfdetr-small" | "rfdetr-medium";
export type DeviceName = "auto" | "cpu" | "mps" | "cuda";

export interface ApiErrorBody {
  error: { code: string; message: string; details?: unknown };
}

export interface Job {
  job_id: string;
  video_id: string;
  status: JobStatus;
  progress: number;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  error_code: string | null;
  error_message: string | null;
  pipeline_version: string;
  configuration: Record<string, unknown>;
  artifact_urls: Partial<Record<"annotated_video" | "detections" | "metadata", string>>;
}

export interface Video {
  video_id: string;
  filename: string;
  duration_seconds: number;
  width: number;
  height: number;
  fps: number;
  frame_count: number | null;
  created_at: string;
  processing_status: VideoProcessingStatus;
  container_format: string;
  video_codec: string;
  size_bytes: number;
  frame_rate_mode: "constant" | "variable" | "unknown";
  has_audio: boolean;
  source_url: string;
  latest_job: Job | null;
}

export interface UploadResponse {
  video_id: string;
  original_filename: string;
  upload_status: "stored";
  metadata: Video;
}

export interface AnalyzeRequest {
  detector?: DetectorName;
  confidence_threshold?: number;
  inference_stride?: number;
  inference_resolution?: number;
  device?: DeviceName;
}

export interface ModelInfo {
  model_id: string;
  family: string;
  variant: string;
  label_space: string;
  license: string;
  weights_file: string | null;
  weights_md5: string | null;
  input_resolution: number | null;
  device: string;
  framework_versions: Record<string, string>;
  is_synthetic: boolean;
}

export interface RunStats {
  decoded_frames: number;
  analyzed_frames: number;
  skipped_frames: number;
  total_detections: number;
  mean_detections_per_analyzed_frame: number | null;
  max_detections_in_frame: number;
  frames_with_estimated_timestamps: number;
  clipped_boxes: number;
  discarded_degenerate_boxes: number;
}

export interface RunTiming {
  started_at: string;
  completed_at: string;
  total_seconds: number;
  model_load_seconds: number | null;
  inference_seconds: number;
  mean_inference_ms: number | null;
  processing_fps: number | null;
}

export interface OutputVideoInfo {
  file: string;
  codec: string;
  pixel_format: string;
  width: number;
  height: number;
  browser_playable: boolean;
  padded_to_even_dimensions: boolean;
}

export interface RunMetadata {
  schema_version: string;
  run_id: string;
  video_id: string;
  status: "completed" | "failed";
  pipeline: { name: string; version: string; package_version: string; stages: string[] };
  configuration: {
    detector: {
      name: DetectorName;
      confidence_threshold: number;
      target_classes: string[];
      inference_resolution: number | null;
      device: DeviceName;
    };
    inference_stride: number;
    max_duration_seconds: number | null;
    draw_hud: boolean;
  };
  model: ModelInfo | null;
  input_sha256: string | null;
  output_video: OutputVideoInfo | null;
  stats: RunStats | null;
  timing: RunTiming;
  environment: { python: string; platform: string; packages: Record<string, string> };
  warnings: string[];
  error: { code: string; message: string; frame_index: number | null } | null;
}

export interface FrameCount {
  frame_index: number;
  timestamp_seconds: number;
  detections: number;
}

export interface Results {
  video: Video;
  job: Job;
  run: RunMetadata;
  frame_counts: FrameCount[];
}

export interface Health {
  status: "ok";
  version: string;
  pipeline_name: string;
  pipeline_version: string;
  devices: { torch_installed: boolean; cuda: boolean; mps: boolean; cpu: boolean };
  model: {
    detector: DetectorName;
    ml_dependencies_installed: boolean;
    weights_present: boolean;
    weights_download_allowed: boolean;
  };
  limits: { max_upload_bytes: number; max_video_duration_seconds: number; max_concurrent_jobs: number };
  defaults: {
    detector: DetectorName;
    device: DeviceName;
    confidence_threshold: number;
    inference_stride: number;
  };
}
