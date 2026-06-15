import math
import time

import cv2
import numpy as np
from pynput import keyboard

from ray import cast_rays

EPSILON = 1e-6


class Wall:
    def __init__(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: np.ndarray = np.full((3,), 0.2, dtype=np.float32),
        reflectivity: float = 0.0,
        light_intensity: float = 0.0,
    ):
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.color = color
        self.reflectivity = reflectivity
        self.light_intensity = light_intensity

        self.m = (y2 - y1) / (x2 - x1 + EPSILON)
        self.b = y1 - self.m * x1
        self.angle = math.degrees(math.atan(self.m))


class Map:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.walls = []
        self.image = np.full((height, width, 3), 0, dtype=np.uint8)

        self.add_wall(
            0,
            0,
            0,
            height,
            color=np.array([1, 1, 1]),
            reflectivity=0.0,
            light_intensity=1.0,
        )
        self.add_wall(
            0,
            0,
            width,
            0,
            color=np.array([1, 1, 1]),
            reflectivity=0.0,
            light_intensity=1.0,
        )
        self.add_wall(
            width,
            0,
            width,
            height,
            color=np.array([1, 1, 1]),
            reflectivity=0.0,
            light_intensity=1.0,
        )
        self.add_wall(
            0,
            height,
            width,
            height,
            color=np.array([1, 1, 1]),
            reflectivity=0.0,
            light_intensity=1.0,
        )

    def add_wall(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: np.ndarray = np.full((3,), 0.5, dtype=np.float32),
        reflectivity: float = 0.0,
        light_intensity: float = 0.0,
    ):
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        wall = Wall(x1, y1, x2, y2, color, reflectivity, light_intensity)
        self.walls.append(wall)
        wall_color = (wall.color * 255).astype(np.uint8).tolist()
        cv2.line(self.image, (x1, y1), (x2, y2), wall_color, 4)

    def add_walls(self, walls: list[tuple[float, float, float, float]]):
        for x1, y1, x2, y2 in walls:
            self.add_wall(x1, y1, x2, y2)

    def can_step(self, x1: float, y1: float, x2: float, y2: float):
        for wall in self.walls:
            if intersects((x1, y1, x2, y2), wall):
                return False
        return True

    def get_image(self, players: list, size: int | tuple[int, int] | None = None):
        image = self.image.copy()

        scale = 1
        if size is not None:
            if isinstance(size, tuple):
                image = cv2.resize(image, size)
            elif isinstance(size, int):
                scale = size / max(self.width, self.height)
                image = cv2.resize(
                    image,
                    (int(self.width * scale), int(self.height * scale)),
                    interpolation=cv2.INTER_NEAREST,  # keeps the lines crisp when upscaling
                )
            else:
                raise TypeError("size must be int, tuple, or None")

        for player in players:
            x = int(player.x * scale)
            y = int(player.y * scale)
            cv2.circle(image, (x, y), 10, (0, 0, 255), -1)
            cv2.line(
                image,
                (x, y),
                (
                    int(x + math.cos(math.radians(player.view_angle)) * 12),
                    int(y + math.sin(math.radians(player.view_angle)) * 12),
                ),
                (0, 0, 0),
                1,
            )
        image = image[::-1, :, :]

        return image


class Player:
    def __init__(self, x: float, y: float, view_angle: float = 0):
        self.x = x
        self.y = y
        self.view_angle = view_angle
        self.health = 100

    def turn_left(self, delta_angle: float = 15):
        self.view_angle += delta_angle
        self.view_angle %= 360

    def turn_right(self, delta_angle: float = 15):
        self.view_angle -= delta_angle
        self.view_angle %= 360

    def forward(self, map: Map, distance: float = 1):
        direction = math.radians(self.view_angle)
        self.move(map, distance, direction)

    def backward(self, map: Map, distance: float = 1):
        direction = math.radians(self.view_angle)
        self.move(map, -distance, direction)

    def right(self, map: Map, distance: float = 1):
        direction = math.radians(self.view_angle)
        self.move(map, distance, direction - math.pi / 2)

    def left(self, map: Map, distance: float = 1):
        direction = math.radians(self.view_angle)
        self.move(map, distance, direction + math.pi / 2)

    def move(self, map: Map, distance: float, direction: float):
        delta_x = math.cos(direction) * distance
        delta_y = math.sin(direction) * distance
        new_x = self.x + delta_x
        new_y = self.y + delta_y
        new_x = min(max(new_x, 0), map.width - 1)
        new_y = min(max(new_y, 0), map.height - 1)

        if map.can_step(self.x, self.y, new_x, new_y):
            self.x = new_x
            self.y = new_y


def intersects(line, wall):
    # swap point order if line is in wrong order
    if line[0] > line[2]:
        line = (line[2], line[3], line[0], line[1])

    # x disjoint?
    if line[0] <= wall.x1:
        if line[2] < wall.x1:
            return False
    elif wall.x2 < line[0]:
        return False

    # y disjoint?
    y1 = min(line[1], line[3])
    y2 = max(line[1], line[3])
    y3 = min(wall.y1, wall.y2)
    y4 = max(wall.y1, wall.y2)

    if y1 <= y3:
        if y2 < y3:
            return False
    elif y4 < y1:
        return False

    x1, y1, x2, y2 = line
    if x1 == x2:
        x2 += EPSILON
    m1 = (y2 - y1) / (x2 - x1)
    b1 = y1 - m1 * x1

    if m1 == wall.m:
        return False

    # m1x + b1 = m2x + b2
    # (m1 - m2)x = (b2 - b1)
    # x = (b2 - b1) / (m1 - m2)

    x_intersect = (wall.b - b1) / (m1 - wall.m)

    if (
        x_intersect >= line[0] - EPSILON
        and x_intersect <= line[2] + EPSILON
        and x_intersect >= wall.x1 - EPSILON
        and x_intersect <= wall.x2 + EPSILON
    ):
        return True

    return False


if __name__ == "__main__":
    SPEED = 2.0  # units per action
    ACTION_INTERVAL = 0.001  # seconds between actions
    TURN_DEGREES = 5

    map = Map(1000, 1000)
    player = Player(50, 50, 45)

    # Outer ring (with gap at top-left for entrance)
    map.add_wall(100, 100, 900, 100, color=np.array([1, 0, 0]))  # top
    map.add_wall(900, 100, 900, 900, color=np.array([1, 0, 0]))  # right
    map.add_wall(900, 900, 100, 900, color=np.array([1, 0, 0]))  # bottom
    map.add_wall(100, 900, 100, 200, color=np.array([1, 0, 0]))  # left (gap)

    # Second ring
    map.add_wall(200, 200, 200, 800, color=np.array([0, 1, 0]))  # left
    map.add_wall(
        200, 800, 800, 800, color=np.array([0, 1, 0]), light_intensity=0.5
    )  # bottom
    map.add_wall(800, 800, 800, 200, color=np.array([0, 1, 0]))  # right
    map.add_wall(800, 200, 300, 200, color=np.array([0, 1, 0]))  # top (gap)

    # Third ring
    map.add_wall(300, 300, 700, 300, color=np.array([0, 0, 1]))  # top
    map.add_wall(700, 300, 700, 700, color=np.array([0, 0, 1]))  # right
    map.add_wall(
        700, 700, 300, 700, color=np.array([0, 0, 1]), light_intensity=0.2
    )  # bottom
    map.add_wall(300, 700, 300, 400, color=np.array([0, 0, 1]))  # left (gap)

    # Fourth ring
    map.add_wall(400, 400, 400, 600)  # left
    map.add_wall(400, 600, 600, 600, light_intensity=0.2)  # bottom
    map.add_wall(600, 600, 600, 400)  # right
    map.add_wall(600, 400, 500, 400)  # top (gap)

    held_keys = set()

    def on_press(key):
        try:
            held_keys.add(key.char)
        except AttributeError:
            pass

    def on_release(key):
        try:
            held_keys.discard(key.char)
        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("Controls: WASD to move, J/L to turn, Q or ESC to quit")

    last_action_time = 0
    running = True

    while running:
        pixels = cast_rays(map, player)
        pixels = cv2.resize(pixels, (1000, 20))
        cv2.imshow("player perspective", pixels)
        cv2.imshow("land", map.get_image([player], 1000))
        key = cv2.waitKey(16) & 0xFF  # pumps the OpenCV window event loop

        if key == 27:  # ESC via the OpenCV window
            running = False

        now = time.monotonic()
        if now - last_action_time >= ACTION_INTERVAL:
            if "w" in held_keys:
                player.forward(map, SPEED)
            if "s" in held_keys:
                player.backward(map, SPEED)
            if "a" in held_keys:
                player.left(map, SPEED)
            if "d" in held_keys:
                player.right(map, SPEED)
            if "j" in held_keys:
                player.turn_left(TURN_DEGREES)
            if "l" in held_keys:
                player.turn_right(TURN_DEGREES)
            if "q" in held_keys:
                running = False
            last_action_time = now

    listener.stop()
    cv2.destroyAllWindows()
