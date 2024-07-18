import matplotlib.pyplot as plt
import numpy as np


def draw_vector(v, color="blue", label=None, arrow_size=0.1):
    """
    Draws a 3D vector in Matplotlib.

    Args:
        v (list): Vectors list [[x1, y1, z1, x2, y2, z2],...]
        color (str, optional): Color of the vector (default: 'blue').
        label (str, optional): Label for the vector (default: None).
        arrow_size (float, optional): Size of the arrowhead (default: 0.1).

    Returns:
        None
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for vector in v:
        # Draw the line representing the vector
        x1, y1, z1, x2, y2, z2 = vector
        ax.quiver(x1, y1, z1, x2, y2, z2, color=color, label=label)

    # Set labels and limits for the axes
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Set the axes range
    ax.set_xlim3d(-2, 2)
    ax.set_ylim3d(-2, 2)
    ax.set_zlim3d(-2, 2)

    # Add legend if label is provided
    if label:
        ax.legend()

    ax.set_aspect("equal")

    plt.show()


if __name__ == "__main__":
    ASPECT_RATIO = 16 / 9
    WIDTH = 400
    HEIGHT = int(WIDTH / ASPECT_RATIO)

    VIEWPORT_HEIGHT = 2.0
    VIEWPORT_WIDTH = VIEWPORT_HEIGHT * (WIDTH / HEIGHT)

    focal_length = 1.0
    camera_center = np.array([0, 0, 0])

    viewport_u = np.array([VIEWPORT_WIDTH, 0, 0])
    viewport_v = np.array([0, -VIEWPORT_HEIGHT, 0])

    pixel_delta_u = viewport_u / WIDTH
    pixel_delta_v = viewport_v / HEIGHT

    viewport_upper_left = (
        camera_center - np.array([0, 0, focal_length]) - viewport_u / 2 - viewport_v / 2
    )
    pixel00_location = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v)

    draw_vector(
        [[0, 0, 0, pixel00_location[0], pixel00_location[1], pixel00_location[2]]]
    )
