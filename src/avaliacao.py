# julga a qualidade da segmentação — IoU e Dice (comparação
# pixel a pixel com a máscara do dermatologista) e cronômetro do pipeline.
"""Avaliação da segmentação (Fase 4 — Passos 23 a 25)."""
import time

import numpy as np


def iou(msk_predita: np.ndarray, msk_referencia: np.ndarray) -> float:
    """Intersection over Union: |A∩B| / |A∪B| — implementação própria com NumPy.

    Operações lógicas pixel a pixel entre as máscaras binárias; devolve 0.0
    quando a união é vazia (duas máscaras sem objeto) para evitar divisão por zero.
    """
    a, b = msk_predita > 0, msk_referencia > 0
    uniao = np.logical_or(a, b).sum()
    if uniao == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / uniao)


def dice(msk_predita: np.ndarray, msk_referencia: np.ndarray) -> float:
    """Dice (F1 da segmentação): 2|A∩B| / (|A|+|B|).

    Sempre Dice >= IoU (mesma interseção, denominador menor ou igual);
    devolve 0.0 se a soma das áreas for vazia.
    """
    a, b = msk_predita > 0, msk_referencia > 0
    soma = a.sum() + b.sum()
    if soma == 0:
        return 0.0
    return float(2 * np.logical_and(a, b).sum() / soma)


def medir(funcao, *args, **kwargs):
    """Executa a função e devolve (resultado, tempo_em_ms).

    perf_counter: monotônico e de alta resolução — imune a ajustes do relógio
    do sistema (que podem fazer time.time() 'voltar no tempo' durante uma medida).
    """
    inicio = time.perf_counter()
    resultado = funcao(*args, **kwargs)
    return resultado, (time.perf_counter() - inicio) * 1000