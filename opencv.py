import cv2
import winsound
import threading
import queue

# URL da câmera RTSP
url = "rtsp://admin:joao1234@192.168.1.106:554/onvif1"
webCam = cv2.VideoCapture(url)

# Configuração do buffer para reduzir latência
webCam.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimiza o buffer de frames

# Carrega o classificador de faces
classificador = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# Coordenadas do retângulo delimitador
rect_x, rect_y, rect_w, rect_h = 100, 100, 400, 300

# Fila para frames (thread-safe)
frame_queue = queue.Queue(maxsize=1)

# Função para capturar frames em uma thread separada
def capture_frames():
    while True:
        ret, frame = webCam.read()
        if not ret:
            print("Erro ao capturar frame")
            break
        # Adiciona o frame na fila, descartando o anterior se cheia
        if not frame_queue.full():
            frame_queue.put(frame)
        else:
            try:
                frame_queue.get_nowait()  # Remove frame antigo
                frame_queue.put(frame)    # Adiciona o novo
            except queue.Empty:
                pass

# Inicia a thread de captura
capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

while True:
    # Pega o frame mais recente da fila
    try:
        frame = frame_queue.get(timeout=1)  # Timeout para não travar
    except queue.Empty:
        continue

    # Reduz a resolução para aliviar o processamento
    frame = cv2.resize(frame, (1280, 720 ))  # Ajuste conforme necessário

    # Converte para cinza apenas a ROI (Region of Interest)
    roi_cinza = cv2.cvtColor(frame[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w], cv2.COLOR_BGR2GRAY)
    upperbodies = classificador.detectMultiScale(roi_cinza, scaleFactor=1.2, minNeighbors=4)

    # Desenha o retângulo delimitador
    cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (255, 0, 0), 2)

    # Processa as detecções
    for (x, y, l, a) in upperbodies:
        # Ajusta as coordenadas para o frame original
        x += rect_x
        y += rect_y
        cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)

        # Verifica se a face está dentro do retângulo
        if (rect_x < x < rect_x + rect_w and rect_y < y < rect_y + rect_h):
            # Toca o som em uma thread separada para não bloquear
            threading.Thread(target=winsound.Beep, args=(1000, 500), daemon=True).start()

    # Exibe o frame
    cv2.imshow('Video', frame)

    # Sai com 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera recursos
webCam.release()
cv2.destroyAllWindows()