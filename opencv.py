import cv2
import winsound

webCam = cv2.VideoCapture(0)
classificador = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# Defina as coordenadas do retângulo delimitador
rect_x, rect_y, rect_w, rect_h = 100, 100, 200, 200

while True:
    camera, frame = webCam.read()
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    upperbodies = classificador.detectMultiScale(cinza, 1.2, 4)
    
    # Desenhe o retângulo delimitador na tela
    cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (255, 0, 0), 2)
    
    for (x, y, l, a) in upperbodies:
        cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)
        
        
        if (rect_x < x < rect_x + rect_w and rect_y < y < rect_y + rect_h):
            winsound.Beep(1000, 100)
    
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webCam.release()
cv2.destroyAllWindows()