import cv2
import numpy as np

#image = imagem de entrada , alpha = grau de nitidez, tamanho_mascara = tamanho da mascara (3x3 nesse exemplo)
def high_boost_filter(image, alpha=1.9, tamanho_mascara=3):

    # Converte a imagem de entrada para o tipo float
    image = image.astype(float)

    # Aplica o filtro de suavização passa-baixa (filtro Gaussiano) na imagem de entrada
    borrado = cv2.GaussianBlur(image, (tamanho_mascara, tamanho_mascara), 0)

    # Calculando a imagem de alta frequência
    freq_alta = image - borrado

    # Multiplicar a imagem de alta frequência pelo fator alpha e adicioná-la de volta à imagem original
    nitidez = image + (alpha * freq_alta)

    # Garante que os valores estejam no intervalo [0, 255], assim garante que a imagem seja de 8 bits
    nitidez = np.clip(nitidez, 0, 255).astype(np.uint8)

    return nitidez


# atribuindo a imagem de entrada a uma variável
imagem_entrada = cv2.imread('imagens/lena.tif')


# Aplica o filtro high-boost com filtragem passa-baixa no domínio espacial na imagem de entrada
imagem_nitidez = high_boost_filter(imagem_entrada)


# mostra as duas imagens
cv2.imshow('Imagem de Entrada', imagem_entrada)
cv2.imshow('Imagem Nitidez Agucada', imagem_nitidez)
cv2.waitKey(0)  
cv2.destroyAllWindows()  

cv2.imwrite('imagem_de_entrada.jpg', imagem_entrada)
cv2.imwrite('imagem_nitidez_agucada.jpg', imagem_nitidez)
