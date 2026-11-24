#!/usr/bin/env python3
"""Coleta metricas de area de lesao e contraste (proxy) para curadoria.

ISIC 2017: amostra de 300 pares (seed 42) dos 2000 disponiveis.
PH2: todas as 200 pastas IMDxxx.

Metricas (somente PIL):
  area_lesao_frac = pixels de lesao (>127) / total, via histograma da mascara
  contraste = |cinza medio dentro da lesao - media global| / 255
Saida: curadoria_scripts/metricas_isic.csv e curadoria_scripts/metricas_ph2.csv
"""
import csv
import os
import random
import sys
import time

from PIL import Image, ImageStat

BASE = "/home/isadora/Documentos/ProcessIMg"
ISIC_IMG = os.path.join(BASE, "dataset/isic2017/images")
ISIC_MSK = os.path.join(BASE, "dataset/isic2017/masks")
PH2_ROOT = os.path.join(BASE, "dataset/ph2/PH2Dataset/PH2 Dataset images")
OUT_DIR = os.path.join(BASE, "curadoria_scripts")


def metricas(img_path: str, msk_path: str) -> dict:
    img = Image.open(img_path)
    msk = Image.open(msk_path).convert("L")
    if img.size != msk.size:
        raise ValueError(f"resolucao divergente: {img_path} {img.size} vs {msk_path} {msk.size}")
    w, h = msk.size
    hist = msk.histogram()
    lesion_px = sum(n for v, n in enumerate(hist) if v > 127)
    frac = lesion_px / float(w * h)
    cinza = img.convert("L")
    media_global = ImageStat.Stat(cinza).mean[0]
    stat_lesao = ImageStat.Stat(cinza, msk)
    media_lesao = stat_lesao.mean[0]
    contraste = abs(media_lesao - media_global) / 255.0
    mais_escura = media_lesao < media_global
    return {
        "nome": os.path.splitext(os.path.basename(img_path))[0],
        "resolucao": f"{w}x{h}",
        "area_lesao_frac": round(frac, 5),
        "contraste": round(contraste, 5),
        "lesao_mais_escura": int(mais_escura),
    }


def coletar_isic():
    nomes = sorted(
        f[:-4]
        for f in os.listdir(ISIC_IMG)
        if f.endswith(".jpg") and os.path.exists(os.path.join(ISIC_MSK, f[:-4] + "_segmentation.png"))
    )
    rng = random.Random(42)
    amostra = rng.sample(nomes, 300)
    linhas = []
    t0 = time.time()
    for i, nome in enumerate(amostra, 1):
        img = os.path.join(ISIC_IMG, nome + ".jpg")
        msk = os.path.join(ISIC_MSK, nome + "_segmentation.png")
        linhas.append(metricas(img, msk))
        if i % 50 == 0:
            print(f"ISIC {i}/300 ({time.time()-t0:.0f}s)", flush=True)
    linhas.sort(key=lambda r: r["nome"])
    return linhas


def coletar_ph2():
    pastas = sorted(d for d in os.listdir(PH2_ROOT) if d.startswith("IMD"))
    linhas = []
    t0 = time.time()
    for i, imd in enumerate(pastas, 1):
        img = os.path.join(PH2_ROOT, imd, f"{imd}_Dermoscopic_Image", f"{imd}.bmp")
        msk = os.path.join(PH2_ROOT, imd, f"{imd}_lesion", f"{imd}_lesion.bmp")
        linhas.append(metricas(img, msk))
        if i % 50 == 0:
            print(f"PH2 {i}/{len(pastas)} ({time.time()-t0:.0f}s)", flush=True)
    return linhas


def salvar(linhas: list, caminho: str):
    with open(caminho, "w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=["nome", "resolucao", "area_lesao_frac", "contraste", "lesao_mais_escura"])
        wtr.writeheader()
        wtr.writerows(linhas)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Coletando ISIC (amostra 300, seed 42)...", flush=True)
    salvar(coletar_isic(), os.path.join(OUT_DIR, "metricas_isic.csv"))
    print("Coletando PH2 (200)...", flush=True)
    salvar(coletar_ph2(), os.path.join(OUT_DIR, "metricas_ph2.csv"))
    print("OK")


if __name__ == "__main__":
    sys.exit(main())
