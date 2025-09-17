import cv2
import numpy as np
import requests

MJPEG_URL = "http://10.7.220.108:8000/stream.mjpg"

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
    [638, 690],  # Pixel
    [134, 633],
    [150, 180]
], dtype=np.float32)

pts_world = np.array([
    [1.6, 0.0],  # Mundo
    [0.0, 0.0],
    [0.0, 1.6]
], dtype=np.float32)

# Calcula matriz de transformação afim
affine_matrix = cv2.getAffineTransform(pts_img, pts_world)

# Configuração ArUco
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

for frame in mjpeg_stream(MJPEG_URL):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        for corner in corners:
            c = corner[0]
            center_x = int(c[:, 0].mean())
            center_y = int(c[:, 1].mean())

            # Aplica transformação afim
            pixel = np.array([[center_x, center_y]], dtype=np.float32)
            pixel = np.array([pixel])
            world = cv2.transform(pixel, affine_matrix)[0][0]
            wx, wy = world

            print(f"Pixel: ({center_x}, {center_y}) -> Mundo: ({wx:.2f}, {wy:.2f})")

            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

    cv2.imshow("ArUco Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
