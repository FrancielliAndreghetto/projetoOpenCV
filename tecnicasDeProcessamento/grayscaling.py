import cv2

# Ler imagem
img = cv2.imread('imagens/17/Anger.jpg')

# Realizar a técnica de grayscaling
newImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Salvar a imagem na pasta
cv2.imwrite('imagensProcessadas/imagem_grayscale.jpg', newImage)

# Cria uma janela redimensionável
cv2.namedWindow('imagem em grayscale', cv2.WINDOW_NORMAL)

# Mostra a imagem
cv2.imshow('imagem em grayscale', newImage)

# Espera o usuário pressionar um botão para sair
cv2.waitKey(0)
