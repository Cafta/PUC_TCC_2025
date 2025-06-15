import cv2
import numpy as np
import mediapipe as mp
import time
from collections import defaultdict
from ultralytics import YOLO

ip = "http:192.168.15.35:8080/video"  # IP da câmera (mude conforme necessário)

cap = cv2.VideoCapture(ip)

# width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# print(f"Tamanho real dos frames de entrada: {int(width)}x{int(height)}")


class MultiPersonCommandDetector:
    def __init__(self):
        # Estados do sistema
        self.state = "MAPPING"  # MAPPING -> LISTENING -> PROCESSING
        self.active_person_id = None
        self.command_timeout = 5.0  # 5 segundos para dar comando
        self.command_start_time = None
        
        # Mapeamento de pessoas
        self.person_tracker = {}
        self.next_person_id = 0
        
        # YOLO com GPU
        self.yolo_model = YOLO('yolov8n.pt')
        self.yolo_model.to('cuda')  # Força uso da GPU
        
        # MediaPipe otimizado - só inicializa quando necessário
        self.mp_hands = None
        self.mp_pose = None
        self.init_mediapipe()
        
        # ROI da mão ativa
        self.active_hand_roi = None
        self.hand_positions_history = []
        
        # Otimizações de performance
        self.frame_skip = 0  # Pular frames para otimização
        self.skip_interval = 2  # Processar 1 a cada 3 frames
        
    def init_mediapipe(self):
        """Inicializa MediaPipe de forma otimizada"""
        try:
            # Tentar usar GPU para MediaPipe (nem sempre funciona)
            import mediapipe as mp
            
            # Configurações mais leves para melhor performance
            self.mp_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,  # Apenas 1 mão por vez
                min_detection_confidence=0.3,  # Threshold ainda menor
                min_tracking_confidence=0.2,   # Threshold ainda menor
                model_complexity=0  # Modelo mais leve
            )
            self.mp_pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=0,  # Modelo mais leve
                enable_segmentation=False,  # Desabilita segmentação
                smooth_landmarks=False,  # Desabilita suavização
                min_detection_confidence=0.3,
                min_tracking_confidence=0.2
            )
            print("MediaPipe inicializado com configurações otimizadas")
        except Exception as e:
            print(f"Erro ao inicializar MediaPipe: {e}")
    
    def detect_persons(self, frame):
        """Detecta todas as pessoas no frame usando YOLO otimizado"""
        # Redimensionar frame para processamento mais rápido
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640 / width
            new_width = 640
            new_height = int(height * scale)
            resized_frame = cv2.resize(frame, (new_width, new_height))
        else:
            resized_frame = frame
            scale = 1.0
        
        # YOLO com configurações otimizadas
        results = self.yolo_model(resized_frame, 
                                classes=[0],  # Apenas pessoas
                                conf=0.5,     # Confiança mínima
                                iou=0.7,      # IoU threshold
                                verbose=False,  # Sem logs
                                device='cuda')  # Força GPU
        
        persons = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                
                # Reescalar coordenadas para frame original
                x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale
                
                persons.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': confidence
                })
        return persons
    
    def track_persons(self, current_persons):
        """Associa pessoas detectadas com IDs existentes ou cria novos"""
        if not self.person_tracker:
            # Primeira detecção - criar IDs para todas as pessoas
            for person in current_persons:
                person['id'] = self.next_person_id
                self.person_tracker[self.next_person_id] = person
                self.next_person_id += 1
            return current_persons
        
        # Associar pessoas existentes (algoritmo simples por distância)
        tracked_persons = []
        used_ids = set()
        
        for current_person in current_persons:
            best_match_id = None
            min_distance = float('inf')
            
            for person_id, tracked_person in self.person_tracker.items():
                if person_id in used_ids:
                    continue
                    
                # Calcular distância entre centroides
                curr_center = self.get_bbox_center(current_person['bbox'])
                track_center = self.get_bbox_center(tracked_person['bbox'])
                distance = np.sqrt((curr_center[0] - track_center[0])**2 + 
                                 (curr_center[1] - track_center[1])**2)
                
                if distance < min_distance and distance < 100:  # threshold
                    min_distance = distance
                    best_match_id = person_id
            
            if best_match_id is not None:
                current_person['id'] = best_match_id
                self.person_tracker[best_match_id] = current_person
                used_ids.add(best_match_id)
            else:
                # Nova pessoa
                current_person['id'] = self.next_person_id
                self.person_tracker[self.next_person_id] = current_person
                self.next_person_id += 1
            
            tracked_persons.append(current_person)
        
        return tracked_persons
    
    def get_bbox_center(self, bbox):
        """Calcula o centro de uma bounding box"""
        return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
    
    def check_raised_hand(self, frame, person):
        """Verifica se a pessoa levantou a mão usando MediaPipe Pose OTIMIZADO"""
        x1, y1, x2, y2 = person['bbox']
        
        # Expandir bbox um pouco para pegar braços
        margin = 20
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(frame.shape[1], x2 + margin)
        y2 = min(frame.shape[0], y2 + margin)
        
        person_roi = frame[y1:y2, x1:x2]
        
        if person_roi.size == 0:
            return False, None, None
        
        # Redimensionar ROI para processamento mais rápido
        roi_height, roi_width = person_roi.shape[:2]
        if roi_width > 300:
            scale = 300 / roi_width
            new_width = 300
            new_height = int(roi_height * scale)
            person_roi = cv2.resize(person_roi, (new_width, new_height))
        else:
            scale = 1.0
        
        # Processar pose na ROI da pessoa
        rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
        pose_results = self.mp_pose.process(rgb_roi)
        
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            
            # Verificar se pulso está acima do ombro
            left_wrist = landmarks[15]      # Left wrist
            right_wrist = landmarks[16]     # Right wrist
            left_elbow = landmarks[13]      # cotovelo esquerdo // usado para calcular a escala do boundingbox.
            right_elbow = landmarks[14]     # cotovelo direito  // usado para calcular a escala do boundingbox.
            left_shoulder = landmarks[11]   # Left shoulder
            right_shoulder = landmarks[12]  # Right shoulder

            if left_wrist.y < left_shoulder.y - 0.15:
                hand_x = x1 + int(left_wrist.x * (x2 - x1))
                hand_y = y1 + int(left_wrist.y * (y2 - y1))

                # Escala baseada no comprimento do antebraço
                escala = np.sqrt(
                    (left_wrist.x - left_elbow.x) ** 2 +
                    (left_wrist.y - left_elbow.y) ** 2
                )
                return True, 'left', (hand_x, hand_y), escala
        
            if right_wrist.y < right_shoulder.y - 0.15:
                hand_x = x1 + int(right_wrist.x * (x2 - x1))
                hand_y = y1 + int(right_wrist.y * (y2 - y1))

                escala = np.sqrt(
                    (right_wrist.x - right_elbow.x) ** 2 +
                    (right_wrist.y - right_elbow.y) ** 2
                )
                return True, 'right', (hand_x, hand_y), escala

        return False, None, None, None
    
    def create_hand_roi(self, hand_position, frame_shape, escala):
        """Cria ROI ao redor da mão detectada"""
        x, y = hand_position

        if escala is None:
            roi_size = 120  # fallback
        else:
            roi_size = int(max(60, min(escala * frame_shape[0] * 1.2, 180)))    # entre 1.2 e 2.0
        
        x1 = max(0, x - roi_size//2)
        y1 = max(0, y - roi_size//2)
        x2 = min(frame_shape[1], x + roi_size//2)
        y2 = min(frame_shape[0], y + roi_size//1.9)
        
        return [x1, y1, x2, y2]
    
    def detect_hand_gesture(self, frame, roi):
        """Detecta gesto específico na ROI da mão OTIMIZADO"""
        x1, y1, x2, y2 = roi
        hand_roi = frame[y1:y2, x1:x2]
        
        if hand_roi.size == 0:
            return None
        
        # Redimensionar ROI para processamento mais rápido
        # if hand_roi.shape[1] > 128:
        #     hand_roi = cv2.resize(hand_roi, (128, 128))
        hand_roi = cv2.resize(hand_roi, (128, 128))
        
        rgb_roi = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)
        hand_results = self.mp_hands.process(rgb_roi)
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Armazenar posição para detectar rotação
                wrist_pos = hand_landmarks.landmark[0]
                current_time = time.time()
                self.hand_positions_history.append((wrist_pos.x, wrist_pos.y, current_time))
                
                # Manter apenas últimas 10 posições (mais eficiente)
                self.hand_positions_history = self.hand_positions_history[-10:]
                
                # Detectar se mão está aberta ou fechada
                is_open = self.is_hand_open(hand_landmarks)
                
                # Detectar rotação (simplificado)
                is_rotating = self.detect_rotation_simple()
                
                # Determinar comando
                if is_rotating:
                    if is_open:
                        return "LIGAR_VENTILADOR"
                    else:
                        return "DESLIGAR_VENTILADOR"
                else:
                    # Implementação simples para abrir/fechar
                    if is_open and len(self.hand_positions_history) > 5:
                        return "ACENDER_LUZ"
                    elif not is_open and len(self.hand_positions_history) > 5:
                        return "APAGAR_LUZ"
        
        return None
    
    def detect_rotation_simple(self):
        """Detecção de rotação simplificada e mais rápida"""
        if len(self.hand_positions_history) < 8:
            return False
        
        # Calcular movimento total
        total_movement = 0
        for i in range(1, len(self.hand_positions_history)):
            prev_pos = self.hand_positions_history[i-1]
            curr_pos = self.hand_positions_history[i]
            movement = ((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)**0.5
            total_movement += movement
        
        # Se há movimento significativo, assume rotação
        return total_movement > 0.3  # Threshold ajustável
    
    def is_hand_open(self, hand_landmarks):
        """Determina se a mão está aberta baseado na posição dos dedos"""
        # Pontos das pontas dos dedos
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        finger_pips = [3, 6, 10, 14, 18]  # PIP joints
        
        open_fingers = 0
        
        # Polegar (diferente dos outros)
        if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
            open_fingers += 1
        
        # Outros dedos
        for tip, pip in zip(finger_tips[1:], finger_pips[1:]):
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                open_fingers += 1
        
        return open_fingers >= 3  # Pelo menos 3 dedos abertos
    
    def detect_rotation(self):
        """Detecta movimento rotacional da mão"""
        if len(self.hand_positions_history) < 10:
            return False
        
        # Calcular ângulos consecutivos
        angles = []
        center_x = sum(pos[0] for pos in self.hand_positions_history) / len(self.hand_positions_history)
        center_y = sum(pos[1] for pos in self.hand_positions_history) / len(self.hand_positions_history)
        
        for pos in self.hand_positions_history:
            angle = np.arctan2(pos[1] - center_y, pos[0] - center_x)
            angles.append(angle)
        
        # Verificar se houve rotação completa
        total_rotation = sum(np.diff(angles))
        return abs(total_rotation) > np.pi  # Mais de 180 graus
    
    def detect_open_close_gesture(self):
        """Detecta gesto de abrir/fechar mão (implementação simplificada)"""
        # Esta função precisaria de histórico de estados aberto/fechado
        # Por simplicidade, retorna None aqui
        return None
    
    def execute_command(self, command):
        """Executa o comando detectado"""
        print(f"Executando comando: {command}")
        
        # Aqui você pode implementar a lógica específica para cada comando
        if command == "ACENDER_LUZ":
            print("💡 Luz acesa!")
            # Implementar comunicação com dispositivo IoT aqui
            
        elif command == "APAGAR_LUZ":
            print("💡 Luz apagada!")
            # Implementar comunicação com dispositivo IoT aqui
            
        elif command == "LIGAR_VENTILADOR":
            print("🌀 Ventilador ligado!")
            # Implementar comunicação com dispositivo IoT aqui
            
        elif command == "DESLIGAR_VENTILADOR":
            print("🌀 Ventilador desligado!")
            # Implementar comunicação com dispositivo IoT aqui
            
        else:
            print(f"Comando não reconhecido: {command}")

    def process_frame(self, frame):
        """Função principal que processa cada frame COM OTIMIZAÇÕES"""
        current_time = time.time()
        
        # Skip frames para melhor performance
        self.frame_skip += 1
        if self.frame_skip % self.skip_interval != 0 and self.state == "MAPPING":
            return frame, None
        
        if self.state == "MAPPING":
            # Etapa 1: Mapear pessoas (menos frequente)
            persons = self.detect_persons(frame)
            tracked_persons = self.track_persons(persons)
            
            # Etapa 2: Procurar primeira pessoa que levanta a mão
            for person in tracked_persons:
                has_raised_hand, hand_side, hand_pos, escala = self.check_raised_hand(frame, person)
                if has_raised_hand:
                    self.active_person_id = person['id']
                    self.active_hand_roi = self.create_hand_roi(hand_pos, frame.shape, escala)
                    self.state = "PROCESSING"
                    self.command_start_time = current_time
                    self.hand_positions_history = []
                    print(f"Pessoa {self.active_person_id} ativada - mão {hand_side}")
                    # Reduzir skip interval quando ativo
                    self.skip_interval = 1
                    break
        
        elif self.state == "PROCESSING":
            # Etapa 4: Processar apenas a pessoa ativa (todo frame)
            if self.active_hand_roi:
                command = self.detect_hand_gesture(frame, self.active_hand_roi)
                
                if command:
                    print(f"Comando detectado: {command}")
                    self.execute_command(command)
                    self.reset_to_mapping()
                    return frame, command
                
                # Etapa 5: Timeout - voltar para mapeamento
                if current_time - self.command_start_time > self.command_timeout:
                    print("Timeout - voltando ao mapeamento")
                    self.reset_to_mapping()
        
        return frame, None
    
    def reset_to_mapping(self):
        """Reseta o sistema para o estado de mapeamento"""
        self.state = "MAPPING"
        self.active_person_id = None
        self.active_hand_roi = None
        self.command_start_time = None
        self.hand_positions_history = []
        self.skip_interval = 2  # Volta ao skip interval normal
    
    def draw_debug_info(self, frame):
        """Desenha informações de debug no frame"""
        # Estado atual
        cv2.putText(frame, f"Estado: {self.state}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Pessoa ativa
        if self.active_person_id is not None:
            cv2.putText(frame, f"Pessoa Ativa: {self.active_person_id}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # ROI da mão
        if self.active_hand_roi:
            x1, y1, x2, y2 = self.active_hand_roi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "ROI Mao", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Timeout
        if self.command_start_time:
            remaining = self.command_timeout - (time.time() - self.command_start_time)
            cv2.putText(frame, f"Timeout: {remaining:.1f}s", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame

# Exemplo de uso
def main():
    detector = MultiPersonCommandDetector()
    
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 384))
        else:
            print("Erro ao capturar frame da câmera")
            break
        
        # Processar frame
        processed_frame, command = detector.process_frame(frame)
        
        # Adicionar informações de debug
        debug_frame = detector.draw_debug_info(processed_frame)
        
        # Mostrar frame
        cv2.imshow('Sistema de Comandos', debug_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()