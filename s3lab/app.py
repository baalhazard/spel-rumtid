from __future__ import annotations

import math
import pygame

from s3lab.core.camera import Camera
from s3lab.core.scene import default_scene
from s3lab.input.pygame_input import apply_keyboard_input
from s3lab.render.raycast import ProgressiveRenderer


def draw_overlay(
    screen: pygame.Surface,
    font: pygame.font.Font,
    renderer: ProgressiveRenderer,
    ui_fps: float,
) -> None:
    padding = 10

    progress = renderer.progress
    bar_w = 260
    bar_h = 12

    x = padding
    y = padding

    pygame.draw.rect(screen, (20, 20, 24), (x, y, bar_w, bar_h))
    pygame.draw.rect(screen, (230, 230, 230), (x, y, int(bar_w * progress), bar_h))
    pygame.draw.rect(screen, (80, 80, 90), (x, y, bar_w, bar_h), 1)

    text = (
        f"UI {ui_fps:5.1f} Hz | "
        f"render {renderer.last_render_fps:5.2f} fps | "
        f"row {renderer.next_row:03d}/{renderer.height} | "
        f"frame {renderer.completed_frames}"
    )

    label = font.render(text, True, (235, 235, 235))
    shadow = font.render(text, True, (0, 0, 0))

    screen.blit(shadow, (x + 1, y + 19))
    screen.blit(label, (x, y + 18))

    marker_y = int((renderer.next_row / renderer.height) * screen.get_height())
    pygame.draw.line(
        screen,
        (255, 255, 255),
        (0, marker_y),
        (screen.get_width(), marker_y),
        1,
    )


def run() -> None:
    pygame.init()
    pygame.font.init()

    render_width = 96
    render_height = 54

    scale = 10
    window_width = render_width * scale
    window_height = render_height * scale

    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("S³ Lab v0.1.1")

    font = pygame.font.SysFont(None, 22)
    clock = pygame.time.Clock()

    camera = Camera.initial()
    scene = default_scene()

    renderer = ProgressiveRenderer(
        width=render_width,
        height=render_height,
        fov_y=math.radians(70.0),
    )
    renderer.start_frame(camera)

    running = True

    while running:
        dt = clock.tick(60) / 1000.0

        running = apply_keyboard_input(camera, dt)

        finished = renderer.step(
            scene,
            time_budget_seconds=0.025,
        )

        if finished:
            renderer.start_frame(camera)

        surface = pygame.surfarray.make_surface(renderer.image.swapaxes(0, 1))
        surface = pygame.transform.scale(surface, (window_width, window_height))

        screen.blit(surface, (0, 0))
        draw_overlay(screen, font, renderer, clock.get_fps())

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    run()