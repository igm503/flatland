import math
import time

import cv2
import numpy as np
from pynput import keyboard

EPSILON = 1e-6


class Map:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.walls = []
        self.image = np.full((height, width, 3), 255, dtype=np.uint8)

    def add_wall(self, x1: float, y1: float, x2: float, y2: float):
        if x1 > x2:
            wall = (x2, y2, x1, y1)
        else:
            wall = (x1, y1, x2, y2)
        self.walls.append(wall)
        cv2.line(self.image, (x1, y1), (x2, y2), (0, 0, 0), 2)

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
            x, y = player.position
            x = int(x * scale)
            y = int(y * scale)
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
    def __init__(self, position: tuple[float, float], view_angle: float = 0):
        self.position = position
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
        new_x = self.position[0] + delta_x
        new_y = self.position[1] + delta_y
        new_x = min(max(new_x, 0), map.width - 1)
        new_y = min(max(new_y, 0), map.height - 1)

        if map.can_step(self.position[0], self.position[1], new_x, new_y):
            self.position = (new_x, new_y)


def intersects(line1, line2):
    if line1[0] > line1[2]:
        line1 = (line1[2], line1[3], line1[0], line1[1])
    if line2[0] > line2[2]:
        line2 = (line2[2], line2[3], line2[0], line2[1])

    # x disjoint?
    if line1[0] <= line2[0]:
        if line1[2] < line2[0]:
            return False
    elif line2[2] < line1[0]:
        return False

    # y disjoint?
    y1 = min(line1[1], line1[3])
    y2 = max(line1[1], line1[3])
    y3 = min(line2[1], line2[3])
    y4 = max(line2[1], line2[3])

    if y1 <= y3:
        if y2 < y3:
            return False
    elif y4 < y1:
        return False

    x1, y1, x2, y2 = line1
    if x1 == x2:
        x2 += EPSILON
    m1 = (y2 - y1) / (x2 - x1)
    b1 = y1 - m1 * x1

    x1, y1, x2, y2 = line2
    if x1 == x2:
        x2 += EPSILON
    m2 = (y2 - y1) / (x2 - x1)
    b2 = y1 - m2 * x1

    if m1 == m2:
        return False

    # m1x + b1 = m2x + b2
    # (m1 - m2)x = (b2 - b1)
    # x = (b2 - b1) / (m1 - m2)

    x_intersect = (b2 - b1) / (m1 - m2)

    if (
        x_intersect >= line1[0] - EPSILON
        and x_intersect <= line1[2] + EPSILON
        and x_intersect >= line2[0] - EPSILON
        and x_intersect <= line2[2] + EPSILON
    ):
        return True

    return False


if __name__ == "__main__":
    SPEED = 2.0  # units per action
    ACTION_INTERVAL = 0.01  # seconds between actions
    TURN_DEGREES = 5

    map = Map(1000, 1000)
    player = Player((50, 50), 0)

    # Outer ring (with gap at top-left for entrance)
    map.add_wall(100, 100, 900, 100)  # top
    map.add_wall(900, 100, 900, 900)  # right
    map.add_wall(900, 900, 100, 900)  # bottom
    map.add_wall(100, 900, 100, 200)  # left (gap)

    # Second ring
    map.add_wall(200, 200, 200, 800)  # left
    map.add_wall(200, 800, 800, 800)  # bottom
    map.add_wall(800, 800, 800, 200)  # right
    map.add_wall(800, 200, 300, 200)  # top (gap)

    # Third ring
    map.add_wall(300, 300, 700, 300)  # top
    map.add_wall(700, 300, 700, 700)  # right
    map.add_wall(700, 700, 300, 700)  # bottom
    map.add_wall(300, 700, 300, 400)  # left (gap)

    # Fourth ring
    map.add_wall(400, 400, 400, 600)  # left
    map.add_wall(400, 600, 600, 600)  # bottom
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
