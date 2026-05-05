from __future__ import annotations

import numpy as np
import pygame

from s3lab.core.camera import Camera


def apply_keyboard_input(camera: Camera, dt: float) -> bool:
    """
    Returns False when the application should quit.

    Translational input is expressed in the camera tangent frame.
    Rotational input rotates the tangent basis at the current S³ point.
    """
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False

        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return False

    keys = pygame.key.get_pressed()

    move_speed = 0.9
    turn_speed = 1.4

    local_velocity = np.array([0.0, 0.0, 0.0])

    if keys[pygame.K_w]:
        local_velocity[0] += move_speed
    if keys[pygame.K_s]:
        local_velocity[0] -= move_speed

    if keys[pygame.K_d]:
        local_velocity[1] += move_speed
    if keys[pygame.K_a]:
        local_velocity[1] -= move_speed

    if keys[pygame.K_e]:
        local_velocity[2] += move_speed
    if keys[pygame.K_q]:
        local_velocity[2] -= move_speed

    if keys[pygame.K_LEFT]:
        camera.yaw(-turn_speed * dt)
    if keys[pygame.K_RIGHT]:
        camera.yaw(turn_speed * dt)

    if keys[pygame.K_UP]:
        camera.pitch(turn_speed * dt)
    if keys[pygame.K_DOWN]:
        camera.pitch(-turn_speed * dt)

    if keys[pygame.K_z]:
        camera.roll(-turn_speed * dt)
    if keys[pygame.K_x]:
        camera.roll(turn_speed * dt)

    camera.move_local(local_velocity, dt)

    return True