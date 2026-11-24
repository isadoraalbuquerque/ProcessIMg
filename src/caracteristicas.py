# mede a lesão encontrada — contorno, área, perímetro,
# circularidade e retângulo envolvente.
"""Extração de características (Fase 4 — Passos 20 a 22)."""
import cv2
import numpy as np


def maior_contorno(msk: np.ndarray) -> np.ndarray:
    """Extrai o contorno do maior objeto da máscara (slides 9_1: 21–30).

    RETR_EXTERNAL: só o contorno externo (buracos internos não interessam ao
    perímetro da lesão). CHAIN_APPROX_SIMPLE: comprime segmentos colineares,
    guardando apenas os vértices.
    """
    contornos, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        raise ValueError("máscara sem objetos — nada a contornar")
    return max(contornos, key=cv2.contourArea)


def desenhar_contorno(img_rgb: np.ndarray, contorno: np.ndarray,
                      cor=(255, 0, 0), espessura: int = 3) -> np.ndarray:
    """Desenha o contorno sobre uma CÓPIA da imagem original.

    cor em RGB (a imagem do projeto é RGB — (255,0,0) desenha vermelho).
    """
    saida = img_rgb.copy()
    cv2.drawContours(saida, [contorno], -1, cor, espessura)
    return saida


def area_pixels(msk: np.ndarray) -> int:
    """Área = contagem direta de pixels do objeto (definição, slides 9_2: 56–58).

    Para máscara binária, somar os valores equivale a contar os pixels do objeto.
    """
    return int(np.sum(msk > 0))


def area_poligono(contorno: np.ndarray) -> float:
    """Área do polígono do contorno (fórmula de Green, via OpenCV).

    Difere levemente da contagem de pixels: o polígono passa pelos CENTROS dos
    pixels da borda e 'corta os cantos' (~0,1–0,9% nos nossos testes).
    """
    return float(cv2.contourArea(contorno))


def perimetro(contorno: np.ndarray) -> float:
    """Perímetro do contorno fechado (arcLength: soma dos lados do polígono)."""
    return float(cv2.arcLength(contorno, True))


def circularidade(area: float, perimetro: float) -> float:
    """Circularidade = 4πA/P² (círculo perfeito = 1) — slides 9_2: 61–63.

    Em imagens digitais, um círculo rasterizado mede ~0,9 (a borda em escada
    infla o perímetro); o que valida a métrica é a ordenação entre formas.
    """
    if perimetro <= 0:
        raise ValueError("perímetro deve ser positivo")
    return float(4 * np.pi * area / perimetro ** 2)


def bounding_box(msk: np.ndarray) -> tuple:
    """Retângulo envolvente do objeto: (x, y, largura, altura) — slides 9_2: 51–54.

    boundingRect opera direto sobre a máscara (pixels não nulos).
    """
    if not msk.any():
        raise ValueError("máscara vazia — sem retângulo envolvente")
    x, y, w, h = cv2.boundingRect(msk)
    return int(x), int(y), int(w), int(h)