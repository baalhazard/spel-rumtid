from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from s3lab.geometry import s3


@dataclass
class Camera:
    position: np.ndarray
    basis: list[np.ndarray]

    @classmethod
    def initial(cls) -> "Camera":
        p, basis = s3.initial_s3_camera_frame()
        return cls(position=p, basis=basis)

    @property
    def forward(self) -> np.ndarray:
        return self.basis[0]

    @property
    def right(self) -> np.ndarray:
        return self.basis[1]

    @property
    def up(self) -> np.ndarray:
        return self.basis[2]

    def reorthonormalize(self) -> None:
        self.position = s3.project_to_s3(self.position)
        self.basis = s3.orthonormalize_basis(self.position, self.basis)

    def move_local(self, local_velocity: np.ndarray, dt: float) -> None:
        """
        local_velocity is expressed in the camera frame:
        [forward, right, up], in radians per second on S³.
        """
        speed = s3.norm(local_velocity)
        if speed < s3.EPS:
            return

        distance = speed * dt
        local_direction = local_velocity / speed
        world_direction = s3.local_to_world_tangent(self.basis, local_direction)

        self.position, self.basis = s3.parallel_transport_basis_along_geodesic(
            self.position,
            self.basis,
            world_direction,
            distance,
        )

    def yaw(self, angle: float) -> None:
        f, r = s3.rotate_pair(self.forward, self.right, angle)
        self.basis[0] = f
        self.basis[1] = r
        self.reorthonormalize()

    def pitch(self, angle: float) -> None:
        f, u = s3.rotate_pair(self.forward, self.up, angle)
        self.basis[0] = f
        self.basis[2] = u
        self.reorthonormalize()

    def roll(self, angle: float) -> None:
        r, u = s3.rotate_pair(self.right, self.up, angle)
        self.basis[1] = r
        self.basis[2] = u
        self.reorthonormalize()