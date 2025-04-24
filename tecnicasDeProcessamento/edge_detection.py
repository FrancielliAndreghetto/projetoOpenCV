import cv2
import os

# Carrega a imagem
imagem = cv2.imread('imagens/17/Sad.jpg')

# Converte para escala de cinza
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Aplica detecção de bordas Canny
bordas = cv2.Canny(imagemCinza, 100, 200)

# Cria pasta de destino, se necessário
pastaDestino = 'imagensProcessadas'
os.makedirs(pastaDestino, exist_ok=True)

# Caminho do arquivo de saída
caminhoSaida = os.path.join(pastaDestino, 'imagem_com_bordas.jpg')

# Salva a imagem com as bordas detectadas
cv2.imwrite(caminhoSaida, bordas)
print(f'Imagem com bordas salva em: {caminhoSaida}')

# Exibe a imagem na tela
cv2.imshow('Imagem com Bordas', bordas)

# Aguarda até que qualquer tecla seja pressionada
cv2.waitKey(0)

# Fecha a janela
cv2.destroyAllWindows()
