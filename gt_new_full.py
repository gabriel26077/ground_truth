import cv2
import numpy as np
import requests
import math
import json
import paho.mqtt.client as mqtt

# URL do stream MJPEG
MJPEG_URL = "http://10.7.220.108:8000/stream.mjpg"

# MQTT
MQTT_BROKER = "10.7.220.187"  # IP do broker
MQTT_PORT = 1883
MQTT_TOPIC = "ground_truth/data"

client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)

# Função para ler MJPEG
def mjpeg_stream(url):
    stream = requests.get(url, stream=True)
    bytes_data = b''
    for chunk in stream.iter_content(chunk_size=1024):
        bytes_data += chunk
        a = bytes_data.find(b'\xff\xd8')
        b = bytes_data.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = bytes_data[a:b+2]
            bytes_data = bytes_data[b+2:]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                yield frame

# Pontos de referência (pixel → mundo)
pts_img = np.array([
    [128, 685],  # Pixel
    [147, 179],
    [647, 191]
], dtype=np.float32)

pts_world = np.array([
    [1.6, 0.0],  # Mundo
    [0.0, 0.0],
    [0.0, 1.6]
], dtype=np.float32)

# Calcula matriz de transformação afim (3 pontos)
affine_matrix = cv2.getAffineTransform(pts_img, pts_world)

# Configuração ArUco
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

for frame in mjpeg_stream(MJPEG_URL):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for corner in corners:
            c = corner[0]
            center_x = int(c[:, 0].mean())
            center_y = int(c[:, 1].mean())

            # Converte pixel para coordenadas reais
            pixel = np.array([[[center_x, center_y]]], dtype=np.float32)
            world = cv2.transform(pixel, affine_matrix)[0][0]
            wx, wy = world

            # Calcula orientação (theta) usando aresta c[0] -> c[1]
            dx = c[1, 0] - c[0, 0]
            dy = c[1, 1] - c[0, 1]
            theta_rad = -math.atan2(dy, dx)

            # Normaliza [-pi, pi]
            theta_rad = (theta_rad + math.pi + math.pi/2) % (2 * math.pi) - math.pi 
            theta_deg = math.degrees(theta_rad)


            # Monta JSON
            # data = {
            #     "x": f'{float(wx):.5f}',
            #     "y": f'{float(wy):.5f}',
            #     "theta_rad": f'{float(theta_rad):.5f}',
            #     "theta_deg": f'{float(theta_deg):.5f}'
            # }

            payload = {
                "ground_truth":f"{float(wx):.5f}, {float(wy):.5f}, {float(theta_rad):.5f}"
            }

            # Publica no MQTT
            # client.publish(MQTT_TOPIC, json.dumps(data_alt))
            client.publish(MQTT_TOPIC, json.dumps(payload))
            print(f"Publicado: {payload}")

            # Marca o centro
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

    cv2.imshow("ArUco Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
client.disconnect()
