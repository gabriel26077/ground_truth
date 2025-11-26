import cv2
import numpy as np
import threading
import time
from flask import Flask, Response

# === CONFIGURAÇÕES ===
MJPEG_URL = "http://192.168.0.121:81"
MARKER_SIZE = 5.8  # cm
PORTA_SERVIDOR = 5000  # Porta onde seu servidor vai rodar

app = Flask(__name__)

class CameraStream:
    def __init__(self):
        self.video = cv2.VideoCapture(MJPEG_URL)
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        
        # Configuração ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        
        # Matriz da Câmera (Lazy Loading)
        self.cam_matrix = None
        self.dist_coef = None
        
        # Pontos 3D
        self.marker_points = np.array([
            [-MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
            [MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
            [MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
            [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0]
        ], dtype=np.float32)

        # Inicia a thread de captura
        if self.video.isOpened():
            self.thread = threading.Thread(target=self.update, args=())
            self.thread.daemon = True
            self.thread.start()
        else:
            print(f"ERRO: Não foi possível conectar a {MJPEG_URL}")

    def __del__(self):
        self.running = False
        if self.video.isOpened():
            self.video.release()

    def update(self):
        """Lê frames continuamente em background"""
        while self.running:
            success, frame = self.video.read()
            if success:
                with self.lock:
                    self.frame = frame
            else:
                # Pequena pausa para não fritar a CPU se cair a conexão
                time.sleep(0.1)

    def get_current_frame(self, process=False):
        """Retorna o frame codificado em JPEG (Raw ou Processado)"""
        with self.lock:
            if self.frame is None:
                return None
            
            # Trabalhar numa cópia para não alterar o frame original da outra thread
            output_frame = self.frame.copy()

        if process:
            output_frame = self.process_aruco(output_frame)

        # Codifica para JPEG
        ret, jpeg = cv2.imencode('.jpg', output_frame)
        return jpeg.tobytes()

    def process_aruco(self, frame):
        """Aplica a lógica de detecção e desenho"""
        # Inicializa matriz se necessário
        if self.cam_matrix is None:
            h, w = frame.shape[:2]
            focal_length = w
            center_x = w / 2
            center_y = h / 2
            self.cam_matrix = np.array([[focal_length, 0, center_x],
                                        [0, focal_length, center_y],
                                        [0, 0, 1]], dtype=np.float32)
            self.dist_coef = np.zeros((4, 1))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for i, corner in enumerate(corners):
                success, rvec, tvec = cv2.solvePnP(self.marker_points, corner, self.cam_matrix, self.dist_coef)
                if success:
                    cv2.drawFrameAxes(frame, self.cam_matrix, self.dist_coef, rvec, tvec, MARKER_SIZE/2)
                    distance = np.linalg.norm(tvec)
                    
                    rvec_matrix = cv2.Rodrigues(rvec)[0]
                    proj_matrix = np.hstack((rvec_matrix, tvec))
                    euler = cv2.decomposeProjectionMatrix(proj_matrix)[6]
                    yaw = euler[1][0] # Pega o Yaw

                    # Desenha Texto
                    c = corner[0]
                    cx, cy = int(c[:, 0].mean()), int(c[:, 1].mean())
                    cv2.putText(frame, f"Dist: {distance:.1f}cm", (cx, cy - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"Yaw: {yaw:.0f}dg", (cx, cy + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return frame

# Instância global da câmera
cam_stream = CameraStream()

def generate(mode='raw'):
    """Gerador de frames para o Flask"""
    while True:
        # Define se processa ou não
        should_process = (mode == 'processed')
        
        frame_bytes = cam_stream.get_current_frame(process=should_process)
        
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.01)

@app.route('/')
def index():
    return """
    <h1>Servidor ArUco Python</h1>
    <p><a href="/raw">Ver Imagem Crua (Raw)</a></p>
    <p><a href="/processed">Ver Imagem Processada (ArUco)</a></p>
    """

@app.route('/raw')
def video_raw():
    return Response(generate(mode='raw'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/processed')
def video_processed():
    return Response(generate(mode='processed'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # host='0.0.0.0' permite que outros PCs na rede acessem
    app.run(host='0.0.0.0', port=PORTA_SERVIDOR, debug=False, threaded=True)