import cv2
import os

# Carrega a imagem
imagem = cv2.imread('imagens/17/Anger.jpg')

if imagem is None:
    print("Erro ao carregar a imagem.")
    exit()

# Converte para escala de cinza
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Carrega o classificador Haar Cascade para rostos
caminhoCascade = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
detectorRosto = cv2.CascadeClassifier(caminhoCascade)

# Detecta rostos (parâmetros ajustáveis)
rostos = detectorRosto.detectMultiScale(imagemCinza, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Desenha as bounding boxes nos rostos detectados
for (x, y, w, h) in rostos:
    cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(imagem, 'Rosto', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Cria a pasta de destino
pastaDestino = 'imagensProcessadas'
os.makedirs(pastaDestino, exist_ok=True)

# Caminho de saída
caminhoSaida = os.path.join(pastaDestino, 'imagem_com_rostos.jpg')

# Salva a imagem anotada
cv2.imwrite(caminhoSaida, imagem)
print(f'Imagem com rostos anotados salva em: {caminhoSaida}')

# Exibe a imagem com as anotações
cv2.imshow('Imagem com Rostos Anotados', imagem)

# Aguarda até que qualquer tecla seja pressionada
cv2.waitKey(0)

# Fecha a janela
cv2.destroyAllWindows()
