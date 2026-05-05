import math

import numpy as np

from s3lab.core.camera import Camera
from s3lab.core.scene import GeodesicSphere, Scene, default_scene
from s3lab.geometry import s3
from s3lab.render.raycast import (
    make_ray_directions_block,
    render,
    render_rows_vectorized,
    sphere_intersection,
)


def assert_camera_frame_is_valid(camera: Camera) -> None:
    assert np.isclose(s3.norm(camera.position), 1.0)

    for axis in camera.basis:
        assert np.isclose(s3.dot(camera.position, axis), 0.0, atol=1e-12)
        assert np.isclose(s3.norm(axis), 1.0)

    for i, axis in enumerate(camera.basis):
        for other in camera.basis[i + 1 :]:
            assert np.isclose(s3.dot(axis, other), 0.0, atol=1e-12)


def test_camera_frame_stays_on_s3_after_motion_and_rotation() -> None:
    camera = Camera.initial()

    camera.move_local(np.array([0.5, -0.25, 0.1]), 0.7)
    camera.yaw(0.4)
    camera.pitch(-0.2)
    camera.roll(0.15)
    camera.move_local(np.array([-0.1, 0.3, 0.2]), 1.2)

    assert_camera_frame_is_valid(camera)


def test_sphere_intersection_hits_expected_geodesic_distance() -> None:
    camera = Camera.initial()
    sphere = GeodesicSphere(
        center=s3.project_to_s3(np.array([0.0, 1.0, 0.0, 0.0])),
        radius=0.22,
        color=np.array([1.0, 0.0, 0.0]),
    )

    hit = sphere_intersection(
        camera.position,
        camera.forward,
        sphere,
        t_min=1e-4,
        t_max=2.0 * math.pi,
    )

    assert hit is not None
    assert np.isclose(s3.geodesic_distance(hit.point, sphere.center), sphere.radius)


def test_vectorized_ray_directions_are_unit_tangent_vectors() -> None:
    camera = Camera.initial()
    dirs = make_ray_directions_block(
        camera,
        width=8,
        height=6,
        fov_y=math.radians(70.0),
        y0=0,
        y1=6,
    )

    assert dirs.shape == (6, 8, 4)
    assert np.allclose(np.linalg.norm(dirs, axis=-1), 1.0)
    assert np.allclose(np.sum(dirs * camera.position[None, None, :], axis=-1), 0.0)


def test_vectorized_renderer_matches_scalar_renderer_on_small_image() -> None:
    camera = Camera.initial()
    scene: Scene = default_scene()

    scalar = render(camera, scene, width=8, height=6, fov_y=math.radians(70.0))
    vectorized = render_rows_vectorized(
        camera,
        scene,
        width=8,
        height=6,
        fov_y=math.radians(70.0),
        y0=0,
        y1=6,
    )

    assert np.array_equal(vectorized, scalar)
