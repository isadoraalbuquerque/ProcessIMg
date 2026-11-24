#  prepara a imagem para a segmentação — separa os canais RGB,
# converte para cinza e HSV e calcula o histograma (do zero, com NumPy).
"""Pré-processamento: canais, cinza e HSV (Fase 2 — Passos 5 a 7)."""
import cv2
import numpy as np


def extrair_canais(img: np.ndarray) -> tuple:
    """Separa uma imagem RGB em três planos de cor.

    Recebe: img (altura, largura, 3) em RGB — convenção do projeto.
    Retorna: tupla (r, g, b), cada plano (altura, largura) em tons de cinza.
    """
    # fatiamento = vista (custo zero); índice 0=R, 1=G, 2=B (convenção RGB do projeto)
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    return r, g, b


def para_cinza(img: np.ndarray) -> np.ndarray:
    """Converte RGB (alt, larg, 3) para um único plano de intensidade (alt, larg).

    Usa os pesos BT.601 do OpenCV: 0.299·R + 0.587·G + 0.114·B (calibrados
    para a sensibilidade do olho humano ao verde).
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def para_cinza_manual(img: np.ndarray) -> np.ndarray:
    """Mesma conversão com os pesos explícitos, em NumPy puro.

    Serve de validação didática: deve diferir da versão do OpenCV
    em no máximo 1 nível (arredondamento).
    """
    return (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)


def para_hsv(img: np.ndarray) -> np.ndarray:
    """Converte RGB para HSV (mesmo shape, 3 canais reinterpretados).

    Convenção do OpenCV (uint8): H em [0, 179] (ângulo 360°/2 p/ caber em 8 bits),
    S e V em [0, 255].
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


def histograma(img_cinza: np.ndarray, normalizar: bool = False) -> np.ndarray:
    """Histograma manual: nº de pixels em cada nível de intensidade (0–255).

    Implementação própria com NumPy (np.bincount conta ocorrências por nível,
    vetorizado). Com normalizar=True devolve H(i) = n_i / (M·N), uma
    distribuição de probabilidade cuja soma é 1.
    """
    hist = np.bincount(img_cinza.ravel(), minlength=256)
    return hist / img_cinza.size if normalizar else hist
