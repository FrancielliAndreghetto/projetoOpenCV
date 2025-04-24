import cv2
import os
import numpy as np

imagem = cv2.imread('imagens/0/Anger.jpg')
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

brilho_medio = np.mean(imagemCinza)
print(f'Brilho médio: {brilho_medio:.2f}')

if brilho_medio < 100:
    ajuste_brilho = 50 
elif brilho_medio > 180:
    ajuste_brilho = -50 
else:
    ajuste_brilho = 0 

imagemTransformada = cv2.convertScaleAbs(imagem, alpha=1, beta=ajuste_brilho)
pastaDestino = 'imagensProcessadas'
os.makedirs(pastaDestino, exist_ok=True)
pastaTransformada = os.path.join(pastaDestino, 'imagem_com_brilho.jpg')
cv2.imwrite(pastaTransformada, imagemTransformada)
print(f'Imagem salva em: {pastaTransformada}')
