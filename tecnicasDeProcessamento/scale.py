import cv2

# Ler imagem
img = cv2.imread('imagens/0/Happy.jpg')

# Realizar a técnica de resize
newImage = cv2.resize(img, (1000, 500))

# Salvar a imagem na pasta
cv2.imwrite('imagensProcessadas/imagem_resize.jpg', newImage)

# Neste caso não será criada uma janela redimensionável, para demonstrar o resize

# Mostra a imagem
cv2.imshow('imagem com resize', newImage)

# Espera o usuário pressionar um botão para sair
cv2.waitKey(0)
