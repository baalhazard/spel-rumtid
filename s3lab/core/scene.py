from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from s3lab.geometry import s3


@dataclass(frozen=True)
class GeodesicSphere:
    center: np.ndarray
    radius: float
    color: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", s3.project_to_s3(self.center))
        object.__setattr__(self, "color", np.asarray(self.color, dtype=float))


@dataclass
class Scene:
    spheres: list[GeodesicSphere]


def default_scene() -> Scene:
    return Scene(
        spheres=[
            GeodesicSphere(
                center=s3.project_to_s3(np.array([0.0, 1.0, 0.0, 0.0])),
                radius=0.22,
                color=np.array([1.0, 0.25, 0.20]),
            ),
            GeodesicSphere(
                center=s3.project_to_s3(np.array([0.0, 0.0, 1.0, 0.0])),
                radius=0.18,
                color=np.array([0.20, 0.75, 1.0]),
            ),
            GeodesicSphere(
                center=s3.project_to_s3(np.array([0.0, 0.0, 0.0, 1.0])),
                radius=0.18,
                color=np.array([0.45, 1.0, 0.35]),
            ),
            GeodesicSphere(
                center=s3.project_to_s3(np.array([-1.0, 0.0, 0.0, 0.0])),
                radius=0.28,
                color=np.array([1.0, 0.9, 0.25]),
            ),
        ]
    )