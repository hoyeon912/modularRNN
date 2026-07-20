def heuristic_action(state) -> int:
    """Bang-bang pole-balancing controller: push toward the direction the pole is falling."""
    _, _, pole_angle, pole_angular_velocity = state
    return 1 if pole_angle + 0.5 * pole_angular_velocity > 0 else 0
