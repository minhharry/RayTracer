import math
from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np


def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


class Ray:
    def __init__(self, origin: np.ndarray, direction: np.ndarray) -> None:
        self.origin = origin
        self.direction = direction
        norm = np.linalg.norm(self.direction)
        if norm > 0:
            self.direction /= norm

    def at(self, t: float):
        return self.origin + self.direction * t


class Hittable:
    @abstractmethod
    def hit(self, ray: Ray, t_min: float, t_max: float):
        pass


class Sphere(Hittable):
    def __init__(
        self, center: np.ndarray, radius: float, color: np.ndarray, is_metal: bool) -> None:
        self.center = center
        self.radius = radius
        self.color = color
        self.is_metal = is_metal

    def hit(
        self, ray: Ray, t_min: float, t_max: float
    ) -> tuple[bool, float, np.ndarray, np.ndarray, np.ndarray]:
        oc = self.center - ray.origin
        a = np.dot(ray.direction, ray.direction)
        b = -2.0 * np.dot(ray.direction, oc)
        c = np.dot(oc, oc) - self.radius * self.radius
        delta = b * b - 4 * a * c
        if delta < 0:
            return (False, None, None, None, None, None)
        t = (-b - math.sqrt(delta)) / (2.0 * a)
        if t < t_min or t_max < t:
            return (False, None, None, None, None, None)
        p = ray.at(t)
        return (True, t, p, normalize_vector(p - self.center), self.color, self.is_metal)


def random_ray(origin: np.ndarray) -> Ray:
    """Generates a random unit vector."""
    return Ray(
        origin,
        np.array([np.random.random()*2-1, np.random.random()*2-1, np.random.random()*2-1]),
    )


def random_on_hemisphere(origin: np.ndarray, normal: np.ndarray) -> Ray:
    """Generates a random unit vector on hemisphere."""
    randRay = random_ray(origin)
    if np.dot(normal, randRay.direction) <= 0:
        randRay.direction = -randRay.direction
    return randRay


def reflect(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Reflects a vector."""
    return v - 2 * np.dot(v, n) * n

def ray_color(objects: list[Hittable], ray: Ray, depth: int = 1) -> np.ndarray:
    """Computes the color of a sphere for a given ray.

    This function uses the intersection point of the ray and the sphere to
    calculate the final color. It returns the color as an RGB tuple.

    Args:
        sphere (Sphere): The sphere object.
        ray (Ray): The ray object.

    Returns:
        tuple: A tuple containing the RGB color values.
    """
    t_min = 0.0
    t_max = 10000.0
    ans = (False, None, None, None)
    for object in objects:
        t = object.hit(ray, t_min, t_max)
        if t[0]:
            t_max = t[1]
            ans = t
    if ans[0] and depth < 5:
        if ans[5]:
            return ans[4] * ray_color(objects, Ray(ans[2], reflect(ray.direction, ans[3])), depth + 1)
        randRay = random_on_hemisphere(ans[2], ans[3])
        randRay.direction += ans[3]
        randRay.direction /= np.linalg.norm(randRay.direction)
        return 0.9 * ans[4] * ray_color(objects, randRay, depth + 1)
    a = 0.5 * (ray.direction[1] + 1.0)
    return (1 - a) * np.array([1.0, 1.0, 1.0]) + a * np.array([0.5, 0.7, 1.0])


ASPECT_RATIO = 16 / 9
HEIGHT = 720
WIDTH = int(HEIGHT * ASPECT_RATIO)

VIEWPORT_HEIGHT = 2.0
VIEWPORT_WIDTH = VIEWPORT_HEIGHT * (WIDTH / HEIGHT)

focal_length = 1
camera_center = np.array([0, 0, 0])

viewport_u = np.array([VIEWPORT_WIDTH, 0, 0])
viewport_v = np.array([0, -VIEWPORT_HEIGHT, 0])

pixel_delta_u = viewport_u / WIDTH
pixel_delta_v = viewport_v / HEIGHT

viewport_upper_left = (
    camera_center - np.array([0, 0, focal_length]) - viewport_u / 2 - viewport_v / 2
)
pixel00_location = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v)


if __name__ == "__main__":
    sphere = Sphere(np.array([-1, 0, -1]), 0.5, np.array([0.9, 0.2, 0.2]), True)
    sphere2 = Sphere(np.array([0, 0, -1]), 0.5, np.array([0.2, 0.9, 0.2]), False)
    sphere3 = Sphere(np.array([1, 0, -1]), 0.5, np.array([0.2, 0.2, 0.9]), True)
    sphere4 = Sphere(
        np.array([0, -100.5, -1]), 100.0, np.array([0.8, 0.8, 0.8]), False
    )
    image = []
    for i in range(HEIGHT):
        print("Processing row:", i+1, "/", HEIGHT)
        image_row = []
        for j in range(WIDTH):
            pixel_center = (
                pixel00_location + (j * pixel_delta_u) + (i * pixel_delta_v)
            )
            ray_direction = pixel_center - camera_center
            ray = Ray(camera_center, ray_direction)
            rayColor = np.array([0.0, 0.0, 0.0])
            rayColor += ray_color([sphere, sphere2, sphere3, sphere4], ray)
            r, g, b = rayColor
            r, g, b = (
                int(math.sqrt(r) * 255.999),
                int(math.sqrt(g) * 255.999),
                int(math.sqrt(b) * 255.999),
            )
            image_row.append((r, g, b))
        image.append(image_row)

    image = np.array(image, dtype=np.uint8)
    plt.imshow(image)
    plt.axis('off')
    plt.imsave('output.png', image)
    plt.show()

