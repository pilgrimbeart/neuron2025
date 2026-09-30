from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

import pygame


class VideoRecorder:
    def __init__(self, fps: int = 30):
        self.fps = fps
        self.process: subprocess.Popen | None = None
        self.output_path: Path | None = None
        self.pending_s = 0.0
        self.first_frame_written = False
        self.frame_duration_s = 1.0 / fps

    @property
    def active(self) -> bool:
        return self.process is not None

    def default_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return Path("videos") / f"recording-{timestamp}.mp4"

    def start(self, surface_size: tuple[int, int], output_path: str | None = None) -> Path:
        if self.active:
            raise RuntimeError("Recorder already active")

        width, height = surface_size
        self.output_path = Path(output_path) if output_path else self.default_path()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.pending_s = 0.0
        self.first_frame_written = False

        command = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.output_path),
        ]

        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return self.output_path

    def _write_surface(self, surface: pygame.Surface) -> None:
        if not self.active or self.process is None or self.process.stdin is None:
            return
        self.process.stdin.write(pygame.image.tobytes(surface, "RGB"))

    def add_frame(self, surface: pygame.Surface, delta_s: float) -> None:
        if not self.active:
            return

        if not self.first_frame_written:
            self._write_surface(surface)
            self.first_frame_written = True

        self.pending_s += delta_s
        while self.pending_s >= self.frame_duration_s:
            self._write_surface(surface)
            self.pending_s -= self.frame_duration_s

    def stop(self) -> Path | None:
        if not self.active or self.process is None:
            return self.output_path

        if self.process.stdin is not None:
            self.process.stdin.close()
        self.process.wait()
        path = self.output_path
        self.process = None
        self.output_path = None
        self.pending_s = 0.0
        self.first_frame_written = False
        return path
