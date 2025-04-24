import cv2
import os
import numpy as np

# Carrega a imagem
imagem = cv2.imread('imagens/17/Sad.jpg')

# Converte para escala de cinza
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Calcula o brilho médio da imagem
brilho_medio = np.mean(imagemCinza)
print(f'Brilho médio: {brilho_medio:.2f}')

# Ajuste de brilho baseado no brilho médio
if brilho_medio < 100:
    ajuste_brilho = 50  # Aumenta o brilho
else:
    ajuste_brilho = 50  # Aumenta o brilho (sem diminuir)

# Aplica o ajuste de brilho
imagemTransformada = cv2.convertScaleAbs(imagem, alpha=1, beta=ajuste_brilho)

# Cria a pasta de destino, se necessário
pastaDestino = 'imagensProcessadas'
os.makedirs(pastaDestino, exist_ok=True)

# Caminho do arquivo de saída
pastaTransformada = os.path.join(pastaDestino, 'imagem_com_brilho.jpg')

# Salva a imagem com o brilho ajustado
cv2.imwrite(pastaTransformada, imagemTransformada)
print(f'Imagem salva em: {pastaTransformada}')

# Cria uma janela redimensionável
cv2.namedWindow('Imagem com Brilho Ajustado', cv2.WINDOW_NORMAL)

# Exibe a imagem com brilho ajustado
cv2.imshow('Imagem com Brilho Ajustado', imagemTransformada)

# Aguarda até que qualquer tecla seja pressionada
cv2.waitKey(0)

# Fecha a janela
cv2.destroyAllWindows()
