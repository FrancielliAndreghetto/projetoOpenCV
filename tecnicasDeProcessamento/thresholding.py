import cv2
import os

# Carrega a imagem
imagem = cv2.imread('imagens/0/Sad.jpg')

# Verifica se a imagem foi carregada corretamente
if imagem is None:
    print("Erro ao carregar a imagem.")
    exit()

# Converte para escala de cinza
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Aplica thresholding binário
limiar = 127
valor_maximo = 255
_, imagem_threshold = cv2.threshold(imagemCinza, limiar, valor_maximo, cv2.THRESH_BINARY)

# Cria a pasta de destino se não existir
pastaDestino = 'imagensProcessadas'
os.makedirs(pastaDestino, exist_ok=True)

# Caminho do arquivo de saída
caminhoSaida = os.path.join(pastaDestino, 'imagem_com_threshold.jpg')

# Salva a imagem com threshold
cv2.imwrite(caminhoSaida, imagem_threshold)
print(f'Imagem com threshold salva em: {caminhoSaida}')

# Exibe a imagem com threshold
cv2.imshow('Imagem com Threshold', imagem_threshold)

# Aguarda até que qualquer tecla seja pressionada
cv2.waitKey(0)

# Fecha a janela
cv2.destroyAllWindows()
