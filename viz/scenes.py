import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from manim import *
from sim.config import BODIES, GRID_X_RANGE, GRID_Y_RANGE
from sim.motion import body_positions_at

COLOR_MAP = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}
BODY_MANIM_COLORS = [RED, GREEN, BLUE]  # correspondencia por color_id


def basin_to_rgb(basin):
    basin = basin.T
    basin = np.flipud(basin)
    h, w = basin.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for value, color in COLOR_MAP.items():
        rgb[basin == value] = color
    return rgb


def load_basin_as_rgb(path="data/basin_map.npy"):
    return basin_to_rgb(np.load(path))


def body_screen_pos(bx, by, img):
    x_min, x_max = GRID_X_RANGE
    y_min, y_max = GRID_Y_RANGE
    u = (bx - x_min) / (x_max - x_min)
    v = (by - y_min) / (y_max - y_min)
    local_x = (u - 0.5) * img.width
    local_y = (v - 0.5) * img.height
    return img.get_center() + np.array([local_x, local_y, 0])


def make_dots(positions, img):
    dots = VGroup()
    for k, (bx, by) in enumerate(positions):
        dot = Dot(
            point=body_screen_pos(bx, by, img),
            radius=0.1,
            fill_color=BODY_MANIM_COLORS[k],
            fill_opacity=1,
            stroke_color=WHITE,
            stroke_width=2,
        )
        dots.add(dot)
    return dots


class BasinMap(Scene):
    def construct(self):
        rgb = load_basin_as_rgb()
        img = ImageMobject(rgb)
        img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
        img.scale_to_fit_height(6)

        self.play(FadeIn(img))
        dots = make_dots([b["pos"] for b in BODIES], img)
        self.play(FadeIn(dots))
        self.wait(2)


class BasinTransition(Scene):
    def construct(self):
        # 1. Foto estática (alta resolución)
        static_rgb = load_basin_as_rgb("data/basin_map.npy")
        static_img = ImageMobject(static_rgb)
        static_img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
        static_img.scale_to_fit_height(6)
        static_dots = make_dots([b["pos"] for b in BODIES], static_img)

        self.add(static_img, static_dots)
        self.wait(4)
        self.remove(static_img, static_dots)

        # 2. Animación de baja resolución con masas en movimiento
        frames = np.load("data/basin_frames.npy")  # (N, res, res)
        n_frames = frames.shape[0]
        fps = 24
        dt_frame = 1.0 / fps

        anim_img = ImageMobject(basin_to_rgb(frames[0]))
        anim_img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
        anim_img.scale_to_fit_height(6)
        self.add(anim_img)

        moving_dots = VGroup(*[
            Dot(radius=0.1, fill_color=BODY_MANIM_COLORS[k], fill_opacity=1,
                stroke_color=WHITE, stroke_width=2)
            for k in range(len(BODIES))
        ])
        self.add(moving_dots)

        for f in range(n_frames):
            t = f / (n_frames - 1)
            pos_t = body_positions_at(t)

            anim_img.pixel_array = basin_to_rgb(frames[f])

            for k in range(len(BODIES)):
                bx, by = pos_t[k]
                moving_dots[k].move_to(body_screen_pos(bx, by, anim_img))

            self.wait(dt_frame)

        self.wait(1)