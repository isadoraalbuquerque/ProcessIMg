# remove pelos finos e ruído da imagem com o filtro da mediana,
# preservando a borda da lesão.
"""Filtragem espacial (Fase 2 — Passo 9)."""
import cv2
import numpy as np


def filtrar_mediana(img: np.ndarray, k: int = 5) -> np.ndarray:
    """Aplica o filtro da mediana com janela k×k (k ímpar ≥ 3).

    Filtro de ordem: substitui cada pixel pela mediana da vizinhança.
    Elimina estruturas mais finas que ~metade da janela (pelos, ruído
    sal-e-pimenta) preservando bordas — slides 4_3: 58–63.
    k maior = mais limpeza, mas também mais borramento de detalhes finos.
    """
    if k < 3 or k % 2 == 0:
        raise ValueError(f"k deve ser ímpar e >= 3 (recebido: {k})")
    return cv2.medianBlur(img, k)