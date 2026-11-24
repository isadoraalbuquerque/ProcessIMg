# separa a lesão do fundo — limiarização por degrau e o método de
# Otsu (implementado do zero, validado contra o OpenCV) + sobreposição visual.
"""Segmentação (Fase 3 — Passos 11 a 14)."""
import numpy as np

from src.preprocessamento import histograma


def binarizar(img_cinza: np.ndarray, limiar: int, invertida: bool = False) -> np.ndarray:
    """Limiarização por degrau: máscara binária {0, 255}.

    Padrão: pixels >= limiar -> 255 (claro vira objeto), < limiar -> 0.
    Com invertida=True a polaridade troca: pixels < limiar -> 255
    (útil quando o objeto é MAIS ESCURO que o fundo, como nossa lesão no cinza).
    """
    if invertida:
        return np.where(img_cinza < limiar, 255, 0).astype(np.uint8)
    return np.where(img_cinza >= limiar, 255, 0).astype(np.uint8)


def otsu(img_cinza: np.ndarray) -> int:
    """Limiar de Otsu: T* = argmax da variância entre classes sigma_B(t).

    Varredura vetorizada sobre o histograma próprio (preprocessamento.histograma):
    np.cumsum acumula peso (omega) e soma de níveis (mu) — cada posição do array
    é o cálculo da classe escura para um t; a classe clara vem por diferença.
    Guarda 1e-12 evita divisão por zero nos extremos (classes vazias).

    Exige uint8 (0–255): o histograma de 256 níveis pressupõe essa faixa.
    Em imagem constante sigma_B é nulo para todo t e a função devolve 0 —
    sem duas populações no histograma não há corte com significado (Passo 13).
    """
    if img_cinza.dtype != np.uint8:
        raise ValueError(f"otsu exige imagem uint8 0–255 (recebido: {img_cinza.dtype})")
    hist = histograma(img_cinza, normalizar=True)      # p(i), soma 1
    niveis = np.arange(256)
    omega = np.cumsum(hist)                            # fração da classe escura (≤ t)
    mu_acum = np.cumsum(niveis * hist)                 # Σ i·p(i) acumulado
    mu_global = niveis @ hist                          # média global
    mu_escura = mu_acum / np.maximum(omega, 1e-12)
    mu_clara = (mu_global - mu_acum) / np.maximum(1 - omega, 1e-12)
    sigma_b = omega * (1 - omega) * (mu_escura - mu_clara) ** 2
    return int(niveis[np.argmax(sigma_b)])


def segmentar_otsu(img_cinza: np.ndarray, invertida: bool = True) -> np.ndarray:
    """Segmentação por Otsu: t* do otsu() + binarizar com a polaridade dada.

    Padrão invertida=True: no cinza a lesão é MAIS ESCURA que a pele, então a
    máscara com lesão=branca exige pixels < t* -> 255.
    """
    t = otsu(img_cinza)
    return binarizar(img_cinza, t, invertida=invertida)


def sobrepor(img_rgb: np.ndarray, msk: np.ndarray, cor=(255, 0, 0), alpha: float = 0.45) -> np.ndarray:
    """Auxiliar de visualização: pinta a máscara com transparência sobre a original.

    cor em RGB; alpha = fração da cor sobre o pixel original (0 = nada, 1 = sólido).
    """
    if msk.shape != img_rgb.shape[:2]:
        raise ValueError(f"máscara {msk.shape} não confere com a imagem {img_rgb.shape[:2]}")
    overlay = img_rgb.copy()
    regiao = msk > 0
    for c in range(3):
        overlay[:, :, c][regiao] = (
            (1 - alpha) * overlay[:, :, c][regiao] + alpha * cor[c]
        ).astype(np.uint8)
    return overlay
