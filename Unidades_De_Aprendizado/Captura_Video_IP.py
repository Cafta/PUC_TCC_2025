import cv2

ip = "http:192.168.15.34:8080/video"  # IP da câmera (mude conforme necessário)

cap = cv2.VideoCapture(ip)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Erro: Não foi possível conectar à câmera")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Stream ended.')
            break
        cv2.imshow("Reconhecimento", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nInterrompido pelo usuário")

finally:
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    print("Recursos liberados")