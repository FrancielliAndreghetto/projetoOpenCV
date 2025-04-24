import cv2
import numpy as np

# Ler imagem
img = cv2.imread('imagens/7/Disgust.jpg')

# Converter de BGR para HSV (BLUE, GREEN RED para HUE, SATURATION, VALUE)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Definir os limites superior e inferior para, no caso, tons de azul, no espaço
# HSV
lower = np.array([100, 150, 50])
upper = np.array([140, 255, 255])

# Criar uma máscara onde os pixels que estão dentro do intervalo ficam brancos
# e os que não estão ficam pretos
mask = cv2.inRange(hsv, lower, upper)

# Aplica a máscara na imagem, mostrando agora apenas as áreas da imagem que
# estão dentro do intervalo de cor (em azul)
newImage = cv2.bitwise_and(img, img, mask=mask)

# Salvar a imagem na pasta
cv2.imwrite('imagensProcessadas/imagem_hsv_blue.jpg', newImage)

# Cria uma janela redimensionável
cv2.namedWindow('imagem com segmentacao por cor com hsv tons de azul', cv2.WINDOW_NORMAL)

# Mostra a imagem
cv2.imshow('imagem com segmentacao por cor com hsv tons de azul', newImage)

# Espera o usuário pressionar um botão para sair
cv2.waitKey(0)