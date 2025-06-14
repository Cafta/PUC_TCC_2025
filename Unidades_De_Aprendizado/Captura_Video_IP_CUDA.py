import cv2
import threading
from queue import Queue

class VideoStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        
        # Configurações para melhor performance
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduzir buffer
        self.cap.set(cv2.CAP_PROP_FPS, 30)        # Limitar FPS
        
        # Reduzir resolução se necessário (comente se não quiser)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.q = Queue(maxsize=2)  # Buffer pequeno
        self.running = True
        
    def start(self):
        self.thread = threading.Thread(target=self.update)
        self.thread.start()
        return self
        
    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break
                
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # Remove frame antigo
                except:
                    pass
            self.q.put(frame)
            
    def read(self):
        if not self.q.empty():
            return self.q.get()
        return None
        
    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()
        self.cap.release()

# Uso otimizado
ip = "http://192.168.15.34:8080/video"

# Iniciar stream em thread separada
vs = VideoStream(ip).start()

print("Stream iniciado. Pressione 'q' para sair")

try:
    while True:
        frame = vs.read()
        if frame is None:
            continue
            
        # Redimensionar para acelerar exibição (opcional)
        # frame = cv2.resize(frame, (640, 480))
        
        cv2.imshow("Reconhecimento", frame)
        
        # Diminuir o waitKey para mais responsividade
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
except KeyboardInterrupt:
    print("\nInterrompido pelo usuário")
    
finally:
    vs.stop()
    cv2.destroyAllWindows()
    print("Stream finalizado")