"""Pré-processamento em lote: gera as imagens intermediárias das 40 imagens.

Para cada imagem de dataset/imagens/, salva em resultados/pre_processamento/:
  <stem>_canais.png      — painel original + R, G, B (escala fixa 0–255)
  <stem>_hsv.png         — painel original + H (colormap circular), S, V
  <stem>_cinza.png       — escala de cinza (BT.601)
  <stem>_histograma.png  — histograma do cinza (contagem própria do Passo 8)
  <stem>_mediana.png     — cinza após mediana k=5 (remoção de pelos finos)

Uso (venv ativado, qualquer pasta):
    python scripts/preprocessar_tudo.py
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")            # lote: sem janela, só salva arquivos
import matplotlib.pyplot as plt

from src.leitura import carregar_imagem, PASTA_IMAGENS
from src.preprocessamento import para_cinza, para_hsv, extrair_canais, histograma
from src.filtragem import filtrar_mediana

SAIDA = RAIZ / "resultados" / "pre_processamento"
SAIDA.mkdir(parents=True, exist_ok=True)


def salvar_fig(fig, nome):
    fig.savefig(SAIDA / nome, bbox_inches="tight", dpi=120)
    plt.close(fig)


imagens = sorted(p.name for p in PASTA_IMAGENS.iterdir()
                 if p.suffix.lower() in {".jpg", ".bmp"})
print(f"Pré-processando {len(imagens)} imagens -> {SAIDA}", flush=True)

for i, nome in enumerate(imagens, start=1):
    stem = Path(nome).stem
    img = carregar_imagem(PASTA_IMAGENS / nome)
    cinza = para_cinza(img)
    hsv = para_hsv(img)

    # 1) canais RGB: original + R + G + B com escala fixa (comparação honesta)
    r, g, b = extrair_canais(img)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img); axes[0].set_title(f"{stem} — original")
    for ax, plano, letra in zip(axes[1:], [r, g, b], ["R", "G", "B"]):
        ax.imshow(plano, cmap="gray", vmin=0, vmax=255)
        ax.set_title(letra)
    for ax in axes:
        ax.axis("off")
    salvar_fig(fig, f"{stem}_canais.png")

    # 2) HSV: H com colormap circular e range do OpenCV; S e V em cinza
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img); axes[0].set_title("original")
    for ax, plano, letra in zip(axes[1:], [hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]], ["H", "S", "V"]):
        if letra == "H":
            ax.imshow(plano, cmap="hsv", vmin=0, vmax=179)
        else:
            ax.imshow(plano, cmap="gray", vmin=0, vmax=255)
        ax.set_title(letra)
    for ax in axes:
        ax.axis("off")
    salvar_fig(fig, f"{stem}_hsv.png")

    # 3) cinza isolado
    cv2.imwrite(str(SAIDA / f"{stem}_cinza.png"), cinza)

    # 4) histograma do cinza (contagem própria) + mediana k=5
    hist = histograma(cinza)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(hist, color="dimgray")
    ax.fill_between(range(256), hist, color="lightgray")
    ax.set_title(f"{stem} — histograma do cinza (nível mais frequente: {hist.argmax()})")
    ax.set_xlabel("nível de intensidade (0–255)")
    ax.set_xlim(0, 255)
    salvar_fig(fig, f"{stem}_histograma.png")

    cv2.imwrite(str(SAIDA / f"{stem}_mediana.png"), filtrar_mediana(cinza, 5))

    print(f"[{i:>2}/{len(imagens)}] {stem} ✓ (5 arquivos)", flush=True)

print("\nConcluído — imagens intermediárias das 40 imagens em resultados/pre_processamento/")
