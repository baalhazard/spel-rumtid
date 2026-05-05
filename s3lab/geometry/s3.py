from __future__ import annotations

import math
import numpy as np

EPS = 1e-9


def dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def normalize(v: np.ndarray) -> np.ndarray:
    n = norm(v)
    if n < EPS:
        raise ValueError("Cannot normalize near-zero vector.")
    return v / n


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def project_to_s3(p: np.ndarray) -> np.ndarray:
    return normalize(np.asarray(p, dtype=float))


def project_to_tangent(p: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Orthogonal projection of w onto T_p S³.
    """
    return w - dot(w, p) * p


def geodesic_distance(p: np.ndarray, q: np.ndarray) -> float:
    return math.acos(clamp(dot(p, q), -1.0, 1.0))


def geodesic_point(p: np.ndarray, v: np.ndarray, t: float) -> np.ndarray:
    """
    Point on S³ reached by following the geodesic from p
    with initial unit tangent direction v for arclength t.
    """
    return math.cos(t) * p + math.sin(t) * v


def geodesic_tangent(p: np.ndarray, v: np.ndarray, t: float) -> np.ndarray:
    """
    Tangent direction of the same geodesic at parameter t.
    """
    return -math.sin(t) * p + math.cos(t) * v


def orthonormalize_basis(p: np.ndarray, basis: list[np.ndarray]) -> list[np.ndarray]:
    """
    Gram-Schmidt inside T_p S³.
    """
    out: list[np.ndarray] = []

    for w in basis:
        u = project_to_tangent(p, w)

        for e in out:
            u = u - dot(u, e) * e

        if norm(u) < EPS:
            u = fallback_tangent_vector(p, out)

        out.append(normalize(u))

    return out


def fallback_tangent_vector(p: np.ndarray, used: list[np.ndarray]) -> np.ndarray:
    """
    Deterministic fallback vector in T_p S³, used only after numerical degeneracy.
    """
    candidates = [
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 1.0]),
    ]

    best = None
    best_n = -1.0

    for c in candidates:
        u = project_to_tangent(p, c)
        for e in used:
            u = u - dot(u, e) * e

        n = norm(u)
        if n > best_n:
            best = u
            best_n = n

    if best is None or best_n < EPS:
        raise RuntimeError("Could not construct fallback tangent vector.")

    return best


def initial_s3_camera_frame() -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Canonical starting point and tangent basis.

    basis[0] = forward
    basis[1] = right
    basis[2] = up
    """
    p = np.array([1.0, 0.0, 0.0, 0.0])

    forward = np.array([0.0, 1.0, 0.0, 0.0])
    right = np.array([0.0, 0.0, 1.0, 0.0])
    up = np.array([0.0, 0.0, 0.0, 1.0])

    return p, [forward, right, up]


def local_to_world_tangent(basis: list[np.ndarray], local: np.ndarray) -> np.ndarray:
    """
    Convert a local R³ vector into a tangent vector in R⁴.
    local = [forward, right, up].
    """
    return (
        float(local[0]) * basis[0]
        + float(local[1]) * basis[1]
        + float(local[2]) * basis[2]
    )


def parallel_transport_basis_along_geodesic(
    p: np.ndarray,
    basis: list[np.ndarray],
    direction: np.ndarray,
    distance: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Move from p along a geodesic and parallel-transport the camera basis.

    direction must be a unit vector in T_p S³.
    """
    direction = normalize(project_to_tangent(p, direction))

    q = geodesic_point(p, direction, distance)
    q = project_to_s3(q)

    transported_direction = geodesic_tangent(p, direction, distance)
    transported_direction = normalize(project_to_tangent(q, transported_direction))

    transported_basis = []

    for e in basis:
        alpha = dot(e, direction)
        orthogonal_part = e - alpha * direction
        transported = orthogonal_part + alpha * transported_direction
        transported_basis.append(transported)

    transported_basis = orthonormalize_basis(q, transported_basis)
    return q, transported_basis


def rotate_pair(a: np.ndarray, b: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Euclidean rotation inside the 2-plane spanned by a and b.
    Both vectors are tangent at the same S³ point.
    """
    c = math.cos(angle)
    s = math.sin(angle)
    return c * a + s * b, -s * a + c * b