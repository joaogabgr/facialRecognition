import cv2
import winsound
import threading
import queue
import numpy as np
from ultralytics import YOLO
import time
import pygame
import os

# Inicializa o pygame para tocar sons
pygame.mixer.init()

# Carrega o som de tiro
som_tiro = pygame.mixer.Sound("sounds/ak47.wav")

# Variável para controlar o tempo entre sons
ultimo_som = 0
TEMPO_ENTRE_SONS = 1  # segundos

# Função para tocar o som de tiro
def tocar_som_tiro():
    global ultimo_som
    tempo_atual = time.time()
    
    # Só emite som se passou tempo suficiente desde o último
    if tempo_atual - ultimo_som >= TEMPO_ENTRE_SONS:
        som_tiro.play()
        ultimo_som = tempo_atual

# Carrega o modelo YOLOv8
model = YOLO('yolov8n.pt')

url = "rtsp://admin:joao1234@192.168.1.106:554/onvif1"
webCam = cv2.VideoCapture(url)

# Configuração do buffer para reduzir latência
webCam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Configurações da interface
WINDOW_NAME = "Sistema Avançado de Detecção de Pessoas"
cv2.namedWindow(WINDOW_NAME)

# Coordenadas do retângulo delimitador
rect_x, rect_y, rect_w, rect_h = 100, 100, 400, 300

# Fila para frames (thread-safe)
frame_queue = queue.Queue(maxsize=1)

# Configurações de detecção
CONFIDENCE_THRESHOLD = 0.5  # Limiar de confiança para detecção
PERSON_CLASS_ID = 0  # ID da classe 'person' no COCO dataset

# Função para adicionar texto com fundo
def draw_text_with_background(img, text, pos, font_scale=0.7, color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # Adiciona fundo semi-transparente
    overlay = img.copy()
    cv2.rectangle(overlay, (pos[0], pos[1] - text_size[1] - 5),
                 (pos[0] + text_size[0], pos[1] + 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    # Adiciona texto
    cv2.putText(img, text, pos, font, font_scale, color, thickness)

# Função para capturar frames em uma thread separada
def capture_frames():
    while True:
        ret, frame = webCam.read()
        if not ret:
            print("Erro ao capturar frame")
            break
        if not frame_queue.full():
            frame_queue.put(frame)
        else:
            try:
                frame_queue.get_nowait()
                frame_queue.put(frame)
            except queue.Empty:
                pass

# Inicia a thread de captura
capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

pessoas_detectadas = 0
frame_count = 0
fps = 0
last_time = cv2.getTickCount()

while True:
    try:
        frame = frame_queue.get(timeout=1)
    except queue.Empty:
        continue

    frame = cv2.resize(frame, (1280, 720))
    frame_display = frame.copy()

    # Calcula FPS
    current_time = cv2.getTickCount()
    if frame_count % 30 == 0:  # Atualiza FPS a cada 30 frames
        fps = 30 * cv2.getTickFrequency() / (current_time - last_time)
        last_time = current_time
    frame_count += 1

    # Detecção usando YOLOv8
    results = model(frame, verbose=False)[0]
    pessoas_detectadas = 0
    pessoas_na_area = 0

    # Processa detecções
    for result in results.boxes.data:
        x1, y1, x2, y2, conf, class_id = result
        if int(class_id) == PERSON_CLASS_ID and conf > CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            pessoas_detectadas += 1
            
            # Verifica se qualquer parte da pessoa está na área de monitoramento
            pessoa_na_area = False
            
            # Verifica se há interseção entre o retângulo da pessoa e a área de monitoramento
            if not (x2 < rect_x or  # Pessoa está totalmente à esquerda
                   x1 > rect_x + rect_w or  # Pessoa está totalmente à direita
                   y2 < rect_y or  # Pessoa está totalmente acima
                   y1 > rect_y + rect_h):  # Pessoa está totalmente abaixo
                pessoa_na_area = True
                pessoas_na_area += 1
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                draw_text_with_background(frame_display, f"ALERTA! Pessoa {conf:.2f}", (x1, y1-10), color=(0, 255, 255))
                # Inicia thread para som
                threading.Thread(target=tocar_som_tiro, daemon=True).start()
            else:
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 165, 255), 2)
                draw_text_with_background(frame_display, f"Pessoa {conf:.2f}", (x1, y1-10))

    # Desenha área de monitoramento
    cv2.rectangle(frame_display, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 255, 0), 2)
    
    # Interface gráfica
    # Título e informações
    draw_text_with_background(frame_display, "Sistema Avançado de Detecção de Pessoas - YOLOv8", (10, 30))
    draw_text_with_background(frame_display, f"Pessoas detectadas: {pessoas_detectadas}", (10, 60))
    draw_text_with_background(frame_display, f"Pessoas na área: {pessoas_na_area}", (10, 90))
    draw_text_with_background(frame_display, f"FPS: {fps:.1f}", (10, 120))
    draw_text_with_background(frame_display, "Pressione 'Q' para sair", (10, 150))
    
    # Status do sistema
    status = "MONITORANDO" if pessoas_detectadas > 0 else "AGUARDANDO"
    status_color = (0, 255, 0) if status == "MONITORANDO" else (0, 0, 255)
    draw_text_with_background(frame_display, f"Status: {status}", 
                            (frame_display.shape[1] - 300, 30), 
                            color=status_color)

    # Exibe o frame
    cv2.imshow(WINDOW_NAME, frame_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera recursos
webCam.release()
cv2.destroyAllWindows()