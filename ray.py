import math
import numpy as np
from numba import njit, prange

EPSILON = 1e-6
NUDGE = 1e-4
FOV = 90.0
RESOLUTION = 640
RAYS_PER_PIXEL = 10000
MAX_BOUNCES = 20
RR_MIN_BOUNCES = 3
PIXEL_HEIGHT = 1


# ---------------------------------------------------------------------------
# Wall layout
# ---------------------------------------------------------------------------
# Numba's nopython mode doesn't like Python classes or dataclasses in the hot
# path, so we pack walls into a single float32 array of shape (N, 10):
#   [x1, y1, x2, y2, r, g, b, reflectivity, light_intensity, _pad]
# Build this once per frame from your existing Wall objects.

WALL_COLS = 10
WALL_X1, WALL_Y1, WALL_X2, WALL_Y2 = 0, 1, 2, 3
WALL_R, WALL_G, WALL_B = 4, 5, 6
WALL_REFL, WALL_LIGHT = 7, 8


def pack_walls(walls):
    """Convert a list of Wall objects into a (N, 10) float32 array."""
    arr = np.zeros((len(walls), WALL_COLS), dtype=np.float32)
    for i, w in enumerate(walls):
        arr[i, WALL_X1] = w.x1
        arr[i, WALL_Y1] = w.y1
        arr[i, WALL_X2] = w.x2
        arr[i, WALL_Y2] = w.y2
        arr[i, WALL_R] = w.color[0]
        arr[i, WALL_G] = w.color[1]
        arr[i, WALL_B] = w.color[2]
        arr[i, WALL_REFL] = w.reflectivity
        arr[i, WALL_LIGHT] = w.light_intensity
    return arr


# ---------------------------------------------------------------------------
# Intersection: returns nearest hit for a single ray.
# Returns (hit_found, x, y, nx, ny, r, g, b, reflectivity, light_intensity)
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def nearest_hit(ox, oy, dx, dy, walls):
    best_t = 1e30
    best_idx = -1
    best_nx = 0.0
    best_ny = 0.0
    best_x = 0.0
    best_y = 0.0

    n_walls = walls.shape[0]
    for i in range(n_walls):
        x1 = walls[i, WALL_X1]
        y1 = walls[i, WALL_Y1]
        x2 = walls[i, WALL_X2]
        y2 = walls[i, WALL_Y2]

        wx = x2 - x1
        wy = y2 - y1

        denom = -dx * wy + dy * wx
        if abs(denom) < EPSILON:
            continue

        rx = x1 - ox
        ry = y1 - oy

        inv = 1.0 / denom
        t = (-wy * rx + wx * ry) * inv
        u = (-dy * rx + dx * ry) * inv

        if t < EPSILON or t >= best_t:
            continue
        if u < -EPSILON or u > 1.0 + EPSILON:
            continue

        # candidate normal, flipped if facing away from ray
        wall_len = math.sqrt(wx * wx + wy * wy)
        nx = -wy / wall_len
        ny = wx / wall_len
        if nx * dx + ny * dy > 0.0:
            nx = -nx
            ny = -ny

        best_t = t
        best_idx = i
        best_nx = nx
        best_ny = ny
        best_x = ox + t * dx
        best_y = oy + t * dy

    return best_idx, best_x, best_y, best_nx, best_ny, best_t


# ---------------------------------------------------------------------------
# Trace a single ray path. Returns RGB contribution to the pixel.
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def trace_path(ox, oy, dx, dy, walls):
    cr = 1.0
    cg = 1.0
    cb = 1.0
    out_r = 0.0
    out_g = 0.0
    out_b = 0.0

    for bounce in range(MAX_BOUNCES):
        idx, hx, hy, nx, ny, t = nearest_hit(ox, oy, dx, dy, walls)
        if idx < 0:
            break

        # multiply throughput by surface color
        cr *= walls[idx, WALL_R]
        cg *= walls[idx, WALL_G]
        cb *= walls[idx, WALL_B]

        light = walls[idx, WALL_LIGHT]
        if light > 0.0:
            out_r += cr * light
            out_g += cg * light
            out_b += cb * light

        # pick reflection vs diffuse
        refl = walls[idx, WALL_REFL]
        if np.random.random() < refl:
            # mirror reflection: d' = d - 2(d·n)n
            dot = dx * nx + dy * ny
            dx = dx - 2.0 * dot * nx
            dy = dy - 2.0 * dot * ny
        else:
            # cosine-weighted diffuse around the normal (2D)
            u = np.random.uniform(-1.0, 1.0)
            cos_t = math.sqrt(1.0 - u * u)
            sin_t = u
            new_dx = nx * cos_t - ny * sin_t
            new_dy = nx * sin_t + ny * cos_t
            dx = new_dx
            dy = new_dy

        # nudge off the surface
        ox = hx + dx * NUDGE
        oy = hy + dy * NUDGE

        # Russian roulette
        if bounce >= RR_MIN_BOUNCES:
            p = cr
            if cg > p:
                p = cg
            if cb > p:
                p = cb
            if p < 0.05:
                p = 0.05
            elif p > 0.95:
                p = 0.95
            if np.random.random() > p:
                break
            inv_p = 1.0 / p
            cr *= inv_p
            cg *= inv_p
            cb *= inv_p

        # early exit on full absorption
        if cr < 1e-6 and cg < 1e-6 and cb < 1e-6:
            break

    return out_r, out_g, out_b


# ---------------------------------------------------------------------------
# Render: parallel over screen columns.
# ---------------------------------------------------------------------------
@njit(cache=True, parallel=True, fastmath=True)
def render(player_x, player_y, walls, ray_angles, jitter_scale):
    n = ray_angles.shape[0]
    col_rgb = np.zeros((n, 3), dtype=np.float32)

    for i in prange(n):
        r_acc = 0.0
        g_acc = 0.0
        b_acc = 0.0

        for _ in range(RAYS_PER_PIXEL):
            angle = ray_angles[i] + np.random.uniform(-jitter_scale, jitter_scale)
            rad = math.radians(angle)
            dx = math.cos(rad)
            dy = math.sin(rad)

            r, g, b = trace_path(player_x, player_y, dx, dy, walls)
            r_acc += r
            g_acc += g
            b_acc += b

        inv = 1.0 / RAYS_PER_PIXEL
        col_rgb[i, 0] = r_acc * inv
        col_rgb[i, 1] = g_acc * inv
        col_rgb[i, 2] = b_acc * inv

    return col_rgb


# ---------------------------------------------------------------------------
# Public entry point: same signature as before.
# ---------------------------------------------------------------------------
def cast_rays(map, player):
    walls = pack_walls(map.walls)

    half_width = math.tan(math.radians(FOV / 2))
    screen_xs = np.linspace(half_width, -half_width, RESOLUTION).astype(np.float32)
    relative_angles = np.degrees(np.arctan(screen_xs))
    ray_angles = (player.view_angle + relative_angles).astype(np.float32)
    jitter_scale = np.float32((FOV / RESOLUTION) / 2)

    col_rgb = render(
        np.float32(player.x),
        np.float32(player.y),
        walls,
        ray_angles,
        jitter_scale,
    )

    # broadcast column color to all 50 rows
    pixels = np.broadcast_to(col_rgb[np.newaxis, :, :], (PIXEL_HEIGHT, RESOLUTION, 3))
    return np.ascontiguousarray(pixels)
