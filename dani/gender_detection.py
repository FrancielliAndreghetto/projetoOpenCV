import cv2
import numpy as np
import os

GENDER_MODEL = "dani/models/gender_net.caffemodel"
GENDER_PROTO = "dani/models/deploy_gender.prototxt"

GENDER_LIST = ['HOMEM', 'MULHER']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)

def resizeFrame(frame, target_width=800, target_height=600):
    original_height, original_width = frame.shape[:2]
    scale_w = target_width / original_width
    scale_h = target_height / original_height
    scale = min(scale_w, scale_h)
    
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    output_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    output_frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame
    
    return output_frame

def predict_gender(face_img):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                 (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    i = gender_preds[0].argmax()
    return GENDER_LIST[i]

def analyze_image(image_path):
    if not os.path.exists(image_path):
        print(f"Erro: arquivo {image_path} não encontrado.")
        return

    img = cv2.imread(image_path)

    if img is None:
        print(f"Erro ao carregar a imagem {image_path}.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        print("Nenhum rosto detectado.")
        return

    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    (x, y, w, h) = faces[0]

    face = img[y:y+h, x:x+w]

    if face.shape[0] < 10 or face.shape[1] < 10:
        print("Rosto detectado é muito pequeno.")
        return

    gender = predict_gender(face)

    img_height, img_width = img.shape[:2]
    reference_size = max(img_width, img_height)
    thickness = max(2, reference_size // 300)
    font_scale = reference_size / 1000.0
    font_thickness = max(1, thickness)

    color = (255, 0, 0) if gender == "HOMEM" else (255, 0, 255)
    cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)

    offset = int(h * 0.05)
    text_position = (x, y - offset if (y - offset) > 20 else y + h + offset)

    cv2.putText(img, gender, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

    img_resized = resizeFrame(img)
    cv2.imshow('Resultado', img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

EMOCAO_LIST = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']

def get_user_input():
    num = int(input("Escolha um número de 0 a 18: "))
    
    if num < 0 or num > 18:
        print("Número inválido! Por favor, escolha um número entre 0 e 18.")
        return None, None

    print("Escolha uma emoção:")
    print("0 - Raiva")
    print("1 - Desprezo")
    print("2 - Nojo")
    print("3 - Medo")
    print("4 - Alegria")
    print("5 - Neutro")
    print("6 - Tristeza")
    print("7 - Surpresa")

    emocao_num = int(input("Digite o número da emoção escolhida: "))
    
    if emocao_num < 0 or emocao_num >= len(EMOCAO_LIST):
        print("Emoção inválida! Por favor, escolha um número válido.")
        return None, None

    return num, EMOCAO_LIST[emocao_num]

def main():
    num, emocao = get_user_input()
    
    if num is None or emocao is None:
        return

    image_path = f"imagens/{num}/{emocao}.jpg"
    print(f"Analisando a imagem: {image_path}")

    analyze_image(image_path)

if __name__ == "__main__":
    main()
