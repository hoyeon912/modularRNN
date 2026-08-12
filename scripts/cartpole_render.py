import numpy as np
from PIL import Image, ImageDraw

# CartPole-v1 physics constants (must match gymnasium's CartPoleEnv for the pole to stay in frame).
X_RANGE = 2.4
POLE_LENGTH = 1.0  # half-length used by gymnasium is 0.5, full visual length 1.0 unit


def render_cartpole_frame(state, size: int = 64) -> np.ndarray:
    """Renders a CartPole-v1 state (x, x_dot, theta, theta_dot) to a size x size x 1
    grayscale uint8 image: a horizontal track, a cart rectangle, and a pole line -- a
    from-scratch renderer (no pygame dependency, which has no prebuilt wheel for this
    Python version and fails to build from source without system SDL headers)."""
    x, _, theta, _ = state
    img = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(img)

    ground_y = int(size * 0.7)
    draw.line([(0, ground_y), (size, ground_y)], fill=255, width=1)

    cart_x = size / 2 + (x / X_RANGE) * (size / 2 - 6)
    cart_w, cart_h = size * 0.18, size * 0.1
    draw.rectangle(
        [cart_x - cart_w / 2, ground_y - cart_h, cart_x + cart_w / 2, ground_y],
        outline=255,
        fill=180,
    )

    pole_pixel_len = (size * 0.35) * (POLE_LENGTH / 1.0)
    tip_x = cart_x + pole_pixel_len * np.sin(theta)
    tip_y = ground_y - cart_h - pole_pixel_len * np.cos(theta)
    draw.line([(cart_x, ground_y - cart_h), (tip_x, tip_y)], fill=255, width=2)

    return np.asarray(img, dtype=np.uint8)[:, :, None]
