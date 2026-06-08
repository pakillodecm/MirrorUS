import numpy as np


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calcula el ángulo interno en B formado por los vectores BA y BC.

    Args:
        a: Coordenadas del punto extremo A (ej. cadera).
        b: Coordenadas del vértice B (ej. rodilla).
        c: Coordenadas del punto extremo C (ej. tobillo).

    Returns:
        Ángulo en grados en el rango [0.0, 180.0].
    """
    vector_ba = a - b
    vector_bc = c - b

    dot_product = np.dot(vector_ba, vector_bc)
    norm_ba = np.linalg.norm(vector_ba)
    norm_bc = np.linalg.norm(vector_bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cos_theta = dot_product / (norm_ba * norm_bc)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_rad = np.arccos(cos_theta)
    theta_deg = np.degrees(theta_rad)

    return theta_deg
