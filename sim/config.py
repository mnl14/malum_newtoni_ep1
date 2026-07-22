# config.py

# Constante de gravitación (unidades arbitrarias normalizadas, no SI)
# Usamos G=1 y ajustamos masas/distancias para que la dinámica sea manejable numéricamente
G = 1.0

# Las tres masas fijas: posición (x, y) y magnitud
# Triángulo equilátero centrado en el origen (configuración clásica del video)
BODIES = [
    {"pos": (-1.0, -0.5773502691896258), "mass": 1.0, "color_id": 0},  # rojo
    {"pos": ( 1.0, -0.5773502691896258), "mass": 1.0, "color_id": 1},  # verde
    {"pos": ( 0.0,  1.1547005383792517), "mass": 1.0, "color_id": 2},  # azul
]

# Radio de colisión: si la masa de prueba se acerca más que esto a un cuerpo,
# se considera "caída" en ese cuerpo
COLLISION_RADIUS = 0.05

# Parámetros de integración
DT = 0.01              # paso de tiempo
MAX_TIME = 100.0       # tiempo máximo antes de declarar "no converge"

# Parámetros del barrido (grilla de condiciones iniciales)
GRID_X_RANGE = (-3.0, 3.0)
GRID_Y_RANGE = (-3.0, 3.0)
GRID_RESOLUTION = 500  # 200x200 puntos, ajustable según potencia de cómputo