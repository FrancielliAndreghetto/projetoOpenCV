# Computação gráfica

Danielle Parmigiani, Francielli Andreghetto, Matthäus Campanher

Link do dataset: https://www.kaggle.com/datasets/tapakah68/facial-emotion-recognition

Situação problema: Identificar o gênero de cada pessoa do dataset.

## Funções dos membros do grupo

No geral, todos os membros do grupo auxiliaram com a solução do problema principal, realizando a pesquisa
das funções da biblioteca OpenCV, sendo a integrante Danielle a responsável pela maior parte do código de
identificação de gênero de cada pessoa do dataset. Além disso, os membros Francielli e Matthäus ficaram
responsáveis pela maior parte da aplicação das técnicas de processamento de imagens. Por fim, todos os colegas
trabalharam juntos estudando a biblioteca, assim como as técnicas de processamento e as imagens do dataset.

## Solução da situação problema

Para resolver a situação problema de identificar o gênero de cada pessoas do dataset, foi utilizado o
OpenCV para detectar rostos com Haar Cascade. Após detectar o rosto, ele é recortado e redimensionado
para ser analisado por uma rede neural treinada em Caffe (Convolutional Architecture for Fast Feature Embedding),
cujo é um framework de deep learning projetado, entre outras coisas, para classificação de imagens.

Ou seja, o fluxo de funcionamento da solução para o problema é:

1. Ler a imagem.
2. Utilizar o Haar Cascade para detectar a presença e a localização do rosto na imagem.
3. O rosto recortado e redimensionado é transformado em um blob de entrada para o modelo Caffe.
4. O OpenCV, através do módulo ```cv2.dnn``` carrega esse modelo Caffe(```.prototxt```, ```.caffemodel```) e
realiza a predição de gênero no rosto detectado.

O código da solução pode ser encontrado na pasta 📁 [solucaoProblema/](./solucaoProblema).

## Técnicas utilizadas

Para aplicar técnicas de processamento de imagens, foram utilizadas diversas funções da biblioteca OpenCV.
Além disso, após aplicar as técnicas as imagens são mostradas na tela e são salvas na pasta 

As técnicas utilizadas para realizar o processamento de imagens foram:

- Ajuste de brilho/contraste.
- Anotação com bounding boxes ou textos
- Conversão para escala de cinza.
- Detecção de bordas (Canny)
- Redimensionamento
- Segmentação por cor (HSV)
- Thresholding

Os códigos de processamentos de imagens podem ser encontrados na pasta 📁 [tecnicasDeProcessamento/](./tecnicasDeProcessamento).

As imagens processadas podem ser encontradas na pasta 📁 [imagensProcessadas/](./imagensProcessadas).
