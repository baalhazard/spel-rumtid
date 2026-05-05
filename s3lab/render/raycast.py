from __future__ import annotations

from dataclasses import dataclass
import math
import time
import numpy as np

from s3lab.geometry import s3
from s3lab.core.camera import Camera
from s3lab.core.scene import Scene, GeodesicSphere


@dataclass
class Hit:
    t: float
    point: np.ndarray
    normal: np.ndarray
    color: np.ndarray


def sphere_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    sphere: GeodesicSphere,
    t_min: float,
    t_max: float,
) -> Hit | None:
    c = sphere.center
    rhs = math.cos(sphere.radius)

    a = s3.dot(origin, c)
    b = s3.dot(direction, c)

    amplitude = math.sqrt(a * a + b * b)
    if amplitude < s3.EPS:
        return None

    quotient = rhs / amplitude
    if quotient < -1.0 or quotient > 1.0:
        return None

    phase = math.atan2(b, a)
    delta = math.acos(s3.clamp(quotient, -1.0, 1.0))

    candidates = []
    for base in (phase + delta, phase - delta):
        for k in range(-2, 4):
            t = base + 2.0 * math.pi * k
            if t_min <= t <= t_max:
                candidates.append(t)

    if not candidates:
        return None

    t_hit = min(candidates)

    point = s3.geodesic_point(origin, direction, t_hit)
    point = s3.project_to_s3(point)

    inward = c - s3.dot(c, point) * point
    if s3.norm(inward) < s3.EPS:
        return None

    outward = -s3.normalize(inward)

    return Hit(
        t=t_hit,
        point=point,
        normal=outward,
        color=sphere.color,
    )


def trace_ray(
    origin: np.ndarray,
    direction: np.ndarray,
    scene: Scene,
    t_min: float = 1e-4,
    t_max: float = 2.0 * math.pi,
) -> Hit | None:
    best: Hit | None = None

    for sphere in scene.spheres:
        hit = sphere_intersection(origin, direction, sphere, t_min, t_max)
        if hit is not None and (best is None or hit.t < best.t):
            best = hit

    return best


def shade(hit: Hit, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
    ray_dir_at_hit = s3.geodesic_tangent(origin, direction, hit.t)
    ray_dir_at_hit = s3.normalize(s3.project_to_tangent(hit.point, ray_dir_at_hit))

    view_factor = max(0.0, s3.dot(hit.normal, -ray_dir_at_hit))
    ambient = 0.20
    diffuse = 0.80 * view_factor

    color = hit.color * (ambient + diffuse)
    return np.clip(color, 0.0, 1.0)


def background(direction: np.ndarray) -> np.ndarray:
    w = 0.5 + 0.5 * direction[3]
    return (
        np.array([0.02, 0.025, 0.04]) * (1.0 - w)
        + np.array([0.08, 0.10, 0.16]) * w
    )


def render_pixel(
    camera: Camera,
    scene: Scene,
    width: int,
    height: int,
    fov_y: float,
    x: int,
    y: int,
) -> np.ndarray:
    aspect = width / height
    half_height = math.tan(0.5 * fov_y)
    half_width = aspect * half_height

    ndc_y = 1.0 - 2.0 * ((y + 0.5) / height)
    ndc_x = 2.0 * ((x + 0.5) / width) - 1.0

    local = np.array([
        1.0,
        ndc_x * half_width,
        ndc_y * half_height,
    ])

    world_dir = s3.local_to_world_tangent(camera.basis, local)
    world_dir = s3.normalize(s3.project_to_tangent(camera.position, world_dir))

    hit = trace_ray(camera.position, world_dir, scene)

    if hit is None:
        color = background(world_dir)
    else:
        color = shade(hit, camera.position, world_dir)

    return np.asarray(np.clip(255.0 * color, 0.0, 255.0), dtype=np.uint8)


def render(
    camera: Camera,
    scene: Scene,
    width: int,
    height: int,
    fov_y: float,
) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            image[y, x] = render_pixel(camera, scene, width, height, fov_y, x, y)

    return image


class ProgressiveRenderer:
    """
    Incremental geodesic ray-caster.

    One complete image is still geometrically coherent, because each frame
    uses a frozen camera snapshot. The user may move the actual camera while
    the current image is being rendered; the next image uses the updated camera.
    """

    def __init__(self, width: int, height: int, fov_y: float):
        self.width = width
        self.height = height
        self.fov_y = fov_y

        self.image = np.zeros((height, width, 3), dtype=np.uint8)

        self.camera_snapshot: Camera | None = None
        self.next_row = 0

        self.completed_frames = 0
        self.last_render_fps = 0.0
        self.frame_start_time = time.perf_counter()

    def start_frame(self, camera: Camera) -> None:
        self.camera_snapshot = Camera(
            position=camera.position.copy(),
            basis=[e.copy() for e in camera.basis],
        )
        self.next_row = 0
        self.frame_start_time = time.perf_counter()

    @property
    def progress(self) -> float:
        return self.next_row / self.height

    def step(self, scene: Scene, time_budget_seconds: float = 0.025) -> bool:
        """
        Renders rows until the time budget is consumed.

        Returns True when a complete image has just been finished.
        """
        if self.camera_snapshot is None:
            raise RuntimeError("ProgressiveRenderer.start_frame(camera) must be called first.")

        start = time.perf_counter()
        rows_done = 0

        while self.next_row < self.height:
            y = self.next_row

            for x in range(self.width):
                self.image[y, x] = render_pixel(
                    self.camera_snapshot,
                    scene,
                    self.width,
                    self.height,
                    self.fov_y,
                    x,
                    y,
                )

            self.next_row += 1
            rows_done += 1

            elapsed = time.perf_counter() - start
            if rows_done >= 1 and elapsed >= time_budget_seconds:
                break

        if self.next_row >= self.height:
            total_time = time.perf_counter() - self.frame_start_time
            if total_time > 0.0:
                self.last_render_fps = 1.0 / total_time

            self.completed_frames += 1
            return True

        return False