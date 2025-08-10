# ======================================================================================================
# Script: analisis_caninos.py
# Descripción: Medición de la proyección de los ángulos de los caninos sobre la línea media central.
# Entrada: Ruta a una imagen radiográfica.
# Salida: Imagen anotada con líneas, ángulos e intersecciones, guardada en la carpeta 'resultados/'.
# Uso: python analisis_caninos.py ruta/a/radiografia.jpg
# ======================================================================================================

import os
import sys
import math
import cv2
import torch # type: ignore
import numpy as np
from PIL import Image # type: ignore
from torchvision import transforms # type: ignore

# Verificación de argumentos desde consola
if len(sys.argv) != 2:
    print("Uso: python analisis_caninos.py ruta/a/imagen.jpg")
    sys.exit(1)

img_path = sys.argv[1]
if not os.path.exists(img_path):
    print(f"ERROR: No se encontró la imagen en {img_path}")
    sys.exit(1)

# Parámetros de entrada
IMG_HEIGHT = 256
IMG_WIDTH = 256
NUM_CLASSES = 3  # fondo, canino izquierdo, canino derecho
CONF_THRESHOLD = 0.85

# Cargar modelos
sys.path.append('../src')
from tooth_shape_model_unet import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

shape_unet = UNet(num_classes=NUM_CLASSES).to(device)
#shape_unet.load_state_dict(torch.load('models/unet_final.pth', map_location=device))

shape_unet.load_state_dict(torch.load('models/tooth_shape_unet.pth', map_location=device))
shape_unet.eval()

infer_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
])

model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/tooth_detection.pt', force_reload=False).to(device)
model.conf = CONF_THRESHOLD

# Leer imagen
img = cv2.imread(img_path)
if img is None:
    print(f"ERROR: No se pudo leer la imagen desde {img_path}")
    sys.exit(1)

orig = img.copy()
height, width = img.shape[:2]

# Inferencia YOLOv5
df = model(img).pandas().xyxy[0]
print("\n=== RESULTADOS DE DETECCIÓN ===")
print(df)

# Funciones #########################

def calculate_line_intersection(line1_p1, line1_p2, line2_p1, line2_p2):
    """
    Calcula el punto de intersección entre dos líneas definidas por dos puntos cada una.
    Si las líneas son paralelas o no se intersectan dentro de los límites de la imagen, devuelve None.
    
    Args:
        line1_p1, line1_p2: Puntos que definen la primera línea
        line2_p1, line2_p2: Puntos que definen la segunda línea
        
    Returns:
        tuple: Coordenadas (x, y) del punto de intersección o None si no hay intersección
    """
    # Línea 1: (x1, y1) a (x2, y2)
    x1, y1 = line1_p1
    x2, y2 = line1_p2
    
    # Línea 2: (x3, y3) a (x4, y4)
    x3, y3 = line2_p1
    x4, y4 = line2_p2
    
    # Calcular denominador para verificar si las líneas son paralelas
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    
    if denom == 0:  # Líneas paralelas
        return None
    
    # Calcular el punto de intersección
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    
    # Punto de intersección
    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)
    
    return (int(x), int(y))
def extend_line_to_boundaries(point1, point2, img_width, img_height, midline_x=None):
    """
    Extiende una línea definida por dos puntos hasta los límites de la imagen o hasta intersectar con la línea media.
    
    Args:
        point1 (tuple): Coordenadas (x, y) del primer punto.
        point2 (tuple): Coordenadas (x, y) del segundo punto.
        img_width (int): Ancho de la imagen.
        img_height (int): Alto de la imagen.
        midline_x (int, opcional): Coordenada x de la línea media. Si se proporciona, la línea se extenderá hasta esta línea.
        
    Returns:
        tuple: Un par de tuplas con las coordenadas de los puntos extendidos (p1_extended, p2_extended).
    """
    x1, y1 = point1
    x2, y2 = point2
    
    # Si los puntos son iguales, no se puede definir una dirección
    if x1 == x2 and y1 == y2:
        return point1, point2
    
    # Calcular la dirección de la línea
    dx = x2 - x1
    dy = y2 - y1
    
    # Si la línea es vertical
    if dx == 0:
        # Extender hasta los bordes superior e inferior
        return (x1, 0), (x1, img_height)
    
    # Calcular la pendiente y el intercepto
    m = dy / dx
    b = y1 - m * x1
    
    # Puntos extendidos
    extended_points = []
    
    # Si hay una línea media definida, calcular la intersección con ella
    if midline_x is not None:
        # Calcular el punto de intersección con la línea media
        y_intersect = m * midline_x + b
        
        # Verificar si la intersección está dentro de los límites de la imagen
        if 0 <= y_intersect <= img_height:
            # Determinar en qué lado de la línea media está el punto original
            if (x1 < midline_x and x2 < midline_x) or (x1 > midline_x and x2 > midline_x):
                # Ambos puntos están en el mismo lado de la línea media
                # Extender hasta la línea media en una dirección
                if x1 < midline_x:
                    extended_points.append((midline_x, int(y_intersect)))
                else:
                    extended_points.append((midline_x, int(y_intersect)))
            elif (x1 < midline_x and x2 > midline_x) or (x1 > midline_x and x2 < midline_x):
                # No es necesario extender hasta la línea media
                pass
    
    # Intersecciones con los bordes de la imagen
    
    # Intersección con y=0 (borde superior)
    if abs(m) > 0.0001:  # No es una línea horizontal
        x_top = (0 - b) / m
        if 0 <= x_top <= img_width:
            extended_points.append((int(x_top), 0))
    
    # Intersección con y=img_height (borde inferior)
    if abs(m) > 0.0001:  # No es una línea horizontal
        x_bottom = (img_height - b) / m
        if 0 <= x_bottom <= img_width:
            extended_points.append((int(x_bottom), img_height))
    
    # Intersección con x=0 (borde izquierdo)
    y_left = b
    if 0 <= y_left <= img_height:
        extended_points.append((0, int(y_left)))
    
    # Intersección con x=img_width (borde derecho)
    y_right = m * img_width + b
    if 0 <= y_right <= img_height:
        extended_points.append((img_width, int(y_right)))
    
    # Si no hay suficientes puntos de intersección, usar los puntos originales
    if len(extended_points) < 2:
        return point1, point2
    
    # Ordenar los puntos extendidos según su distancia desde el punto medio entre p1 y p2
    midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)
    
    # Si estamos del lado izquierdo de la línea media y queremos extender hacia la línea media
    if midline_x is not None and (x1 < midline_x and x2 < midline_x):
        # Encontrar el punto más cercano al borde y el punto más cercano a la línea media
        points_sorted = sorted(extended_points, key=lambda p: p[0])  # Ordenar por coordenada x
        return points_sorted[0], points_sorted[-1]  # El primero es el más a la izquierda, el último es el más a la derecha
    
    # Si estamos del lado derecho de la línea media y queremos extender hacia la línea media
    elif midline_x is not None and (x1 > midline_x and x2 > midline_x):
        # Encontrar el punto más cercano al borde y el punto más cercano a la línea media
        points_sorted = sorted(extended_points, key=lambda p: p[0], reverse=True)  # Ordenar por coordenada x (reverso)
        return points_sorted[0], points_sorted[-1]  # El primero es el más a la derecha, el último es el más a la izquierda
    
    # En otros casos, simplemente usar las dos intersecciones más alejadas entre sí
    else:
        # Calcular todas las combinaciones de distancias entre puntos
        max_dist = 0
        p1_ext, p2_ext = extended_points[0], extended_points[1]
        
        for i in range(len(extended_points)):
            for j in range(i + 1, len(extended_points)):
                dist = math.sqrt((extended_points[i][0] - extended_points[j][0])**2 + 
                                (extended_points[i][1] - extended_points[j][1])**2)
                if dist > max_dist:
                    max_dist = dist
                    p1_ext, p2_ext = extended_points[i], extended_points[j]
        
        return p1_ext, p2_ext
def get_center(detection):
    xmin, ymin, xmax, ymax = map(int, [detection['xmin'], detection['ymin'], 
                                      detection['xmax'], detection['ymax']])
    cx = (xmin + xmax) // 2
    cy = (ymin + ymax) // 2
    return (cx, cy)
def get_corners(detection, side):
    if side == 'izq':
        return (int(detection['xmax']), int(detection['ymax'])), (int(detection['xmin']), int(detection['ymin']))
    elif side == 'der':
        return (int(detection['xmin']), int(detection['ymax'])), (int(detection['xmax']), int(detection['ymin']))
    else:   
        raise ValueError("Lado no válido. Debe ser 'izq' o 'der'.")
def segment_full_and_crop(orig_img, roi_coords):
    # Segmenta toda la imagen y recorta ROI
    img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    x = infer_transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = shape_unet(x)
        probs  = torch.softmax(logits, dim=1)[0]  # [3,H,W]

    # si side=='left'
    ch = probs[1] if side=='left' else probs[2]
    heatmap = (ch.cpu().numpy() * 255).astype(np.uint8)
    h, w = orig_img.shape[:2]
    heatmap_full = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(f"debug/heat_full_{side}.png", heatmap_full)
    chan = probs[1] if side=='left' else probs[2]
    mask = (chan.cpu().numpy() > 0.03).astype(np.uint8) * 255
    # redimensionar a full-res
    h,w = orig_img.shape[:2]
    mask_full = cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)
    # recortar
    # x1,y1,x2,y2 = roi_coords
    return mask_full
def calculate_angle(line_p1, line_p2, vertical_line_x):
    """
    Calcula el ángulo entre una línea definida por dos puntos y una línea vertical.
    
    Args:
        line_p1 (tuple): Primer punto de la línea.
        line_p2 (tuple): Segundo punto de la línea.
        vertical_line_x (int): Coordenada x de la línea vertical.
        
    Returns:
        float: Ángulo en grados entre las líneas.
    """
    # Verificar que los puntos no sean iguales
    if line_p1[0] == line_p2[0] and line_p1[1] == line_p2[1]:
        return 0  # No se puede calcular el ángulo si los puntos son iguales
    
    # Vector de la línea
    vector_line = (line_p2[0] - line_p1[0], line_p2[1] - line_p1[1])
    
    # Vector de la línea vertical (0, 1) normalizado
    vector_vertical = (0, 1)
    
    # Calcular el ángulo entre los vectores usando el producto punto
    # Normalizar los vectores
    magnitude_line = math.sqrt(vector_line[0]**2 + vector_line[1]**2)
    
    if magnitude_line == 0:
        return 0
    
    unit_vector_line = (vector_line[0] / magnitude_line, vector_line[1] / magnitude_line)
    
    # Producto punto de los vectores unitarios
    dot_product = unit_vector_line[0] * vector_vertical[0] + unit_vector_line[1] * vector_vertical[1]
    
    # Asegurarse de que el producto punto esté en el rango [-1, 1]
    dot_product = max(-1.0, min(1.0, dot_product))
    
    # Calcular el ángulo en radianes y convertirlo a grados
    angle_rad = math.acos(dot_product)
    angle_deg = math.degrees(angle_rad)
    
    # Determinar la dirección del ángulo (positivo o negativo)
    # Si el punto p2 está a la derecha de la línea vertical, el ángulo es positivo
    # Si está a la izquierda, el ángulo es negativo
    direction = 1 if (line_p1[0] < vertical_line_x and line_p2[0] > vertical_line_x) or \
                    (line_p1[0] > vertical_line_x and line_p2[0] < vertical_line_x and line_p1[1] > line_p2[1]) else -1
    
    # Ajustar el ángulo según el cuadrante
    if unit_vector_line[0] < 0:
        angle_deg = 180 - angle_deg
    
    # Asegurarse de que el ángulo esté entre 0 y 180 grados
    if angle_deg > 90:
        angle_deg = 180 - angle_deg
        
    return angle_deg * direction
def process_and_draw_canine(det, orig_img, side, inc_center_x=None):
    coords = (int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax']))
    # Extraemos Cordenadas
    x1, y1, x2, y2 = coords
    roi = orig_img[coords[1]:coords[3], coords[0]:coords[2]]
    # 1) Segmentar
    mask_roi = segment_full_and_crop(roi, coords)
    cv2.imwrite(f"debug/mask_roi_{side}.png", mask_roi)
    # 2) Extraer contorno
    cnts,_ = cv2.findContours(mask_roi,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return orig_img
    contour = max(cnts, key=cv2.contourArea).reshape(-1,2).astype(np.float32)
    # 3) PCA
    mean,vecs,_ = cv2.PCACompute2(contour, mean=None)
    axis = vecs[0]
    # 4) Extremos
    dif = contour - mean
    projs = dif.dot(axis.T)
    p1 = list(contour[np.argmin(projs)].astype(int))
    p2 = list(contour[np.argmax(projs)].astype(int))
    if p1[1] > p2[1]: p1, p2 = p2, p1
    
    # Coordenadas punto 1
    p1x = p1[0]
    p1y = p1[1]
    # Coordenadas punto 2
    p2x = p2[0]
    p2y = p2[1]
    
    if (side == "izq") and (p1x > p2x):
        p1[1] = p2y
        p2[1] = p1y
    if (side == "der") and (p1x < p2x):
        p1[1] = p2y
        p2[1] = p1y
        
    # Ajustar a coords global
    p1g = (p1[0]+coords[0], p1[1]+coords[1])
    p2g = (p2[0]+coords[0], p2[1]+coords[1])
    # 5) Dibujar
    cv2.circle(orig_img, p1g, 4,(0,255,0),-1)
    cv2.circle(orig_img, p2g, 4,(0,255,0),-1)

    # Agregar punto (sacar despues)
    cv2.putText(orig_img, 'p1', (p1g[0] + 10, p1g[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(orig_img, 'p2', (p2g[0] + 10, p2g[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Extender la línea hasta los límites de la imagen o hasta la línea media
    h, w = orig_img.shape[:2]
    midline_x = inc_center_x if inc_center_x is not None else None
    
    exp1, exp2 = extend_line_to_boundaries(p1g, p2g, w, h, midline_x)
    
    cv2.line(orig_img, exp1, exp2, (0,0,255), 2)
    
    # Para los angulos
    # Coordenada central en X
    center_x = (x1 + x2) // 2
    # Coordenada para poner el texto justo debajo de la caja
    text_y = y2 + 30

    if midline_x is not None:
        midline_top = (midline_x, 0)
        midline_bottom = (midline_x, h)
        intersection = calculate_line_intersection(exp1, exp2, midline_top, midline_bottom)
        
        if intersection:
            cv2.circle(orig_img, intersection, 6, (255, 0, 255), -1)
            angle = calculate_angle(exp1, exp2, midline_x)
            angle_text = f"Angulo: {abs(round(angle,2))} Deg"
            cv2.putText(orig_img, angle_text, 
            (center_x - 50, text_y),  # desplazamos 50 px a la izquierda para centrar mejor el texto
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            radius = 40
            start_angle = 90
            
            if angle < 0:
                end_angle = 90 - abs(angle)
            else:
                end_angle = 90 + abs(angle)
            
            cv2.ellipse(orig_img, intersection, (radius, radius), 
                        0, min(start_angle, end_angle), max(start_angle, end_angle), 
                        (255, 0, 255), 2)
            
            #cv2.putText(orig_img, "Intersección", 
             #          (intersection[0] + 10, intersection[1]), 
              #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    return angle # Se elimino orig_image ya que no era necesario

#####################################
# Procesar detecciones
detections_inc = df[df['name'] == 'inc']
detections_can = df[df['name'] == 'canine']

# Línea media
inc_center = None
if len(detections_inc) > 0:
    inc = detections_inc.sort_values('confidence', ascending=False).iloc[0]
    inc_center = get_center(inc)
    cv2.line(orig, (inc_center[0], 0), (inc_center[0], height), (0, 255, 0), 2)
    cv2.putText(orig, 'Línea media', (inc_center[0] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
else:
    print("No se detectaron incisivos")

riesgo = False
# Procesar caninos
for _, det in detections_can.iterrows():
    center = get_center(det)
    side = 'izq' if inc_center and center[0] < inc_center[0] else 'der'
    x1, y1, x2, y2 = map(int, [det['xmin'], det['ymin'], det['xmax'], det['ymax']])
    color = (255, 0, 0) if side == 'izq' else (0, 0, 255)
    cv2.rectangle(orig, (x1, y1), (x2, y2), color, 2)
    cv2.circle(orig, center, 5, color, -1)
    angle = process_and_draw_canine(det, orig, side, inc_center[0] if inc_center else None)
    if side == 'izq':
        print(f"Ángulo canino izquierdo: {abs(round(angle,2))}°")
    else:
        print(f"Ángulo canino derecho: {abs(round(angle,2))}°")
    if abs(round(angle,2)) > 15:
        riesgo = True

if riesgo == True:
    print(f"EXISTE RIESGO")
else:
    print(f"NO EXISTE RIESGO")
# Guardar salida
nombre_salida = os.path.splitext(os.path.basename(img_path))[0] + '_analizada_mod4.jpg'
os.makedirs('pruebas_modelo', exist_ok=True)
cv2.imwrite(os.path.join('pruebas_modelo', nombre_salida), orig)
print(f"Resultado guardado en: pruebas_modelo/{nombre_salida}")