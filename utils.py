import math
import cv2
from cv2 import dnn_superres

def increase_img_quality(image):
    sr = dnn_superres.DnnSuperResImpl_create()
    
    path = "../models/EDSR_x4.pb"
    sr.readModel(path)
    sr.setModel("edsr", 4)
    
    upscaled = sr.upsample(image)
    return upscaled
    

def extend_line(point1, point2, length=50):
    """
    Extiende una línea definida por dos puntos en ambas direcciones.

    Args:
        point1 (tuple): Coordenadas (x, y) del primer punto.
        point2 (tuple): Coordenadas (x, y) del segundo punto.
        length (int): Longitud en píxeles para extender la línea en cada dirección.

    Returns:
        tuple: Un par de tuplas con las coordenadas de los puntos extendidos (p1_extended, p2_extended).
    """
    x1, y1 = point1
    x2, y2 = point2
    dx = x2 - x1
    dy = y2 - y1
    norm = math.sqrt(dx*dx + dy*dy)
    if norm == 0:
        return point1, point2  # Evitar división por cero si los puntos son iguales

    # Vector unitario de la dirección
    unit_dx = dx / norm
    unit_dy = dy / norm

    # Calcular los puntos extendidos
    extended_x1 = int(x1 - unit_dx * length)
    extended_y1 = int(y1 - unit_dy * length)
    extended_x2 = int(x2 + unit_dx * length)
    extended_y2 = int(y2 + unit_dy * length)

    return (extended_x1, extended_y1), (extended_x2, extended_y2)