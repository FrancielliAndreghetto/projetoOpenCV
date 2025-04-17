import cv2

# Ler imagem
img = cv2.imread('imagens/17/Anger.jpg')

# Realizar a técnica de grayscaling
newImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Salvar a imagem na pasta
cv2.imwrite('imagensFormatadas/imagem_grayscale.jpg', newImage)
