# viz/scenes.py
import numpy as np
from manim import *
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sim.config import BODIES, GRID_X_RANGE, GRID_Y_RANGE

# Colores según config.py: 0=rojo, 1=verde, 2=azul, -1=no converge
COLOR_MAP = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    -1: (20, 20, 20),  # no converge -> casi negro
}

def load_basin_as_rgb(path="data/basin_map.npy"):
    basin = np.load(path)          # basin[i,j]: i~x, j~y
    basin = basin.T                 # ahora filas=y, columnas=x
    basin = np.flipud(basin)        # fila 0 = y_max, para que "arriba" en imagen = y grande

    h, w = basin.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for value, color in COLOR_MAP.items():
        rgb[basin == value] = color

    return rgb

class BasinMap(Scene):
    def construct(self):
        rgb = load_basin_as_rgb()
        img = ImageMobject(rgb)
        img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
        img.scale_to_fit_height(6)

        self.play(FadeIn(img))

        # convertir posición física (x,y) a posición relativa dentro del frame de la imagen
        x_min, x_max = GRID_X_RANGE
        y_min, y_max = GRID_Y_RANGE
        img_w = img.width
        img_h = img.height

        dots = VGroup()
        for body in BODIES:
            bx, by = body["pos"]
            # normalizar a [0,1] dentro del rango de la grilla
            u = (bx - x_min) / (x_max - x_min)
            v = (by - y_min) / (y_max - y_min)
            # mapear a coordenadas centradas en el centro de la imagen
            local_x = (u - 0.5) * img_w
            local_y = (v - 0.5) * img_h
            dot = Dot(point=img.get_center() + np.array([local_x, local_y, 0]),
                      color=WHITE, radius=0.08)
            dots.add(dot)

        self.play(FadeIn(dots))
        self.wait(2)