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


def make_ray_directions_block(
    camera: Camera,
    width: int,
    height: int,
    fov_y: float,
    y0: int,
    y1: int,
) -> np.ndarray:
    """
    Returns unit tangent ray directions for rows [y0, y1).

    Shape: (rows, width, 4)
    """
    aspect = width / height
    half_height = math.tan(0.5 * fov_y)
    half_width = aspect * half_height

    xs = np.arange(width, dtype=float) + 0.5
    ys = np.arange(y0, y1, dtype=float) + 0.5

    ndc_x = 2.0 * (xs / width) - 1.0
    ndc_y = 1.0 - 2.0 * (ys / height)

    local_right = (ndc_x * half_width)[None, :, None]
    local_up = (ndc_y * half_height)[:, None, None]

    p = camera.position[None, None, :]
    f = camera.forward[None, None, :]
    r = camera.right[None, None, :]
    u = camera.up[None, None, :]

    dirs = f + local_right * r + local_up * u

    tangent_error = np.sum(dirs * p, axis=-1, keepdims=True)
    dirs = dirs - tangent_error * p

    lengths = np.linalg.norm(dirs, axis=-1, keepdims=True)
    return dirs / np.maximum(lengths, s3.EPS)


def render_rows_vectorized(
    camera: Camera,
    scene: Scene,
    width: int,
    height: int,
    fov_y: float,
    y0: int,
    y1: int,
    t_min: float = 1e-4,
    t_max: float = 2.0 * math.pi,
) -> np.ndarray:
    """
    Vectorized geodesic ray-casting for a block of rows.

    The geometry matches render_pixel(), but the per-pixel Python loops are
    replaced with NumPy operations over the whole row block.
    """
    dirs = make_ray_directions_block(camera, width, height, fov_y, y0, y1)

    rows = y1 - y0
    p = camera.position

    best_t = np.full((rows, width), np.inf, dtype=float)
    best_color = np.zeros((rows, width, 3), dtype=float)
    best_center = np.zeros((rows, width, 4), dtype=float)

    for sphere in scene.spheres:
        c = sphere.center
        rhs = math.cos(sphere.radius)

        a = float(np.dot(p, c))
        b = np.tensordot(dirs, c, axes=([-1], [0]))

        amplitude = np.sqrt(a * a + b * b)
        valid = amplitude > s3.EPS

        quotient = np.zeros_like(amplitude)
        quotient[valid] = rhs / amplitude[valid]

        valid = valid & (quotient >= -1.0) & (quotient <= 1.0)
        if not np.any(valid):
            continue

        quotient_clamped = np.clip(quotient, -1.0, 1.0)
        phase = np.arctan2(b, a)
        delta = np.arccos(quotient_clamped)

        for base in (phase + delta, phase - delta):
            for k in range(-2, 4):
                t = base + 2.0 * math.pi * k
                mask = valid & (t >= t_min) & (t <= t_max) & (t < best_t)
                if not np.any(mask):
                    continue

                best_t[mask] = t[mask]
                best_color[mask] = sphere.color
                best_center[mask] = c

    w = 0.5 + 0.5 * dirs[:, :, 3]
    bg0 = np.array([0.02, 0.025, 0.04])
    bg1 = np.array([0.08, 0.10, 0.16])
    image = (
        bg0[None, None, :] * (1.0 - w[:, :, None])
        + bg1[None, None, :] * w[:, :, None]
    )

    hit_mask = np.isfinite(best_t)

    if np.any(hit_mask):
        t = best_t[hit_mask]
        d = dirs[hit_mask]
        c = best_center[hit_mask]
        base_color = best_color[hit_mask]

        cos_t = np.cos(t)[:, None]
        sin_t = np.sin(t)[:, None]

        point = cos_t * p[None, :] + sin_t * d
        point = point / np.maximum(np.linalg.norm(point, axis=1, keepdims=True), s3.EPS)

        inward = c - np.sum(c * point, axis=1, keepdims=True) * point
        inward_len = np.linalg.norm(inward, axis=1, keepdims=True)
        valid_normal = inward_len[:, 0] >= s3.EPS

        shaded = np.zeros_like(base_color)
        if np.any(valid_normal):
            normal = -inward[valid_normal] / inward_len[valid_normal]

            ray_dir_at_hit = (
                -sin_t[valid_normal] * p[None, :]
                + cos_t[valid_normal] * d[valid_normal]
            )
            ray_dir_at_hit = ray_dir_at_hit / np.maximum(
                np.linalg.norm(ray_dir_at_hit, axis=1, keepdims=True),
                s3.EPS,
            )

            view_factor = np.maximum(0.0, np.sum(normal * (-ray_dir_at_hit), axis=1))
            shaded[valid_normal] = (
                base_color[valid_normal] * (0.20 + 0.80 * view_factor)[:, None]
            )

        image[hit_mask] = np.clip(shaded, 0.0, 1.0)

    return np.asarray(np.clip(255.0 * image, 0.0, 255.0), dtype=np.uint8)


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

    def step(
        self,
        scene: Scene,
        time_budget_seconds: float = 0.025,
        block_rows: int = 4,
    ) -> bool:
        """
        Renders row blocks until the time budget is consumed.

        Returns True when a complete image has just been finished.
        """
        if self.camera_snapshot is None:
            raise RuntimeError("ProgressiveRenderer.start_frame(camera) must be called first.")

        start = time.perf_counter()
        block_rows = max(1, block_rows)

        while self.next_row < self.height:
            y0 = self.next_row
            y1 = min(self.height, y0 + block_rows)

            self.image[y0:y1, :, :] = render_rows_vectorized(
                self.camera_snapshot,
                scene,
                self.width,
                self.height,
                self.fov_y,
                y0,
                y1,
            )

            self.next_row = y1

            elapsed = time.perf_counter() - start
            if elapsed >= time_budget_seconds:
                break

        if self.next_row >= self.height:
            total_time = time.perf_counter() - self.frame_start_time
            if total_time > 0.0:
                self.last_render_fps = 1.0 / total_time

            self.completed_frames += 1
            return True

        return False
