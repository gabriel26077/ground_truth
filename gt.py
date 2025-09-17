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

# Dicionário ArUco
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

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

# Matriz de transformação
matriz = np.array([
    [ 0.00319364,  0.00001260, -0.46487],
    [-0.00000630, -0.00315584,  2.04904],
    [ 0.0,         0.0,         1.0]
])

for frame in mjpeg_stream(MJPEG_URL):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i, corner in enumerate(corners):
            c = corner[0]
            center_x = int(c[:, 0].mean())
            center_y = int(c[:, 1].mean())

            # Ponto transformado
            ponto = np.array([center_x, center_y, 1])
            ponto_transformado = matriz @ ponto
            x_transf, y_transf = ponto_transformado[0], ponto_transformado[1]

            # Calcula a orientação usando a aresta c[0] -> c[1]
            dx = c[1, 0] - c[0, 0]
            dy = c[1, 1] - c[0, 1]
            angulo_rad = -math.atan2(dy, dx)

            # Normaliza para [-pi, pi]
            angulo_rad = (angulo_rad + math.pi) % (2 * math.pi) - math.pi

            # Converte para graus [-180, 180]
            angulo_deg = math.degrees(angulo_rad)

            # Monta JSON
            data = {
                "x": float(x_transf),
                "y": float(y_transf),
                "theta_rad": float(angulo_rad),
                "theta_deg": float(angulo_deg)
            }

            # Publica no MQTT
            client.publish(MQTT_TOPIC, json.dumps(data))
            print(f"Publicado: {data}")

            # Desenha centro
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

    cv2.imshow("ArUco Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
client.disconnect()

