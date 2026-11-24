#!/usr/bin/env python3
"""Seleciona 40 pares (30 ISIC + 10 PH2) e monta a secao 8 do dataset.

Alocacao ISIC (30) por dificuldade de contraste:
  dificil (baixo <0.08): 9  |  medio (0.08-0.18): 12  |  facil (alto >0.18): 9
  (~30% facil / 40% medio / 30% dificil), cruzado com area:
    baixo: 1 pequena + 5 media + 3 grande
    medio: 4 pequena + 6 media + 2 grande
    alto : 2 pequena + 5 media + 2 grande
PH2 (10): 3 baixo + 4 medio + 3 alto, evenly-spaced na area dentro da faixa.
Exclusoes: contraste < 0.04 ou area < 1% (extremos).
"""
import csv
import os
import shutil
import sys

from PIL import Image

BASE = "/home/isadora/Documentos/ProcessIMg"
SCRIPTS = os.path.join(BASE, "curadoria_scripts")
ISIC_IMG = os.path.join(BASE, "dataset/isic2017/images")
ISIC_MSK = os.path.join(BASE, "dataset/isic2017/masks")
PH2_ROOT = os.path.join(BASE, "dataset/ph2/PH2Dataset/PH2 Dataset images")
DIR_IMG = os.path.join(BASE, "dataset/imagens")
DIR_MSK = os.path.join(BASE, "dataset/mascaras_referencia")
META = os.path.join(BASE, "dataset/metadata.csv")

# limites
C_BAIXO, C_ALTO = 0.08, 0.18
A_PEQ, A_GRA = 0.05, 0.30
C_MIN, A_MIN = 0.04, 0.01

COTAS_ISIC = {  # (contraste, area) -> quantidade
    ("baixo", "pequena"): 1, ("baixo", "media"): 5, ("baixo", "grande"): 3,
    ("medio", "pequena"): 4, ("medio", "media"): 6, ("medio", "grande"): 2,
    ("alto", "pequena"): 2, ("alto", "media"): 5, ("alto", "grande"): 2,
}
COTAS_PH2 = {"baixo": 3, "medio": 4, "alto": 3}


def faixa_contraste(c: float) -> str:
    return "baixo" if c < C_BAIXO else ("alto" if c > C_ALTO else "medio")


def faixa_area(a: float) -> str:
    return "pequena" if a < A_PEQ else ("grande" if a > A_GRA else "media")


def evenly_spaced(rows, k):
    """k itens espalhados pelo espectro de area (rows ordenadas por area asc)."""
    rows = sorted(rows, key=lambda r: r["area"])
    n = len(rows)
    if k <= 0:
        return []
    if n <= k:
        return list(rows)
    if k == 1:
        return [rows[n // 2]]
    return [rows[round(i * (n - 1) / (k - 1))] for i in range(k)]


def carregar(caminho):
    with open(caminho, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["area"] = float(r["area_lesao_frac"])
        r["contraste"] = float(r["contraste"])
    return rows


def selecionar_isic(rows):
    el = [r for r in rows if r["contraste"] >= C_MIN and r["area"] >= A_MIN]
    sel = []
    for (fc, fa), k in COTAS_ISIC.items():
        celula = [r for r in el if faixa_contraste(r["contraste"]) == fc and faixa_area(r["area"]) == fa]
        faltam = k - len(celula)
        sel.extend(evenly_spaced(celula, k))
        print(f"  ISIC {fc}/{fa}: pedidos={k} disponiveis={len(celula)}")
        if faltam > 0:
            print(f"    AVISO: celula com apenas {len(celula)} candidatos; completando fora da celula")
            usados = {r["nome"] for r in sel}
            extra_pool = [r for r in el if r["nome"] not in usados and faixa_contraste(r["contraste"]) == fc]
            sel.extend(evenly_spaced(extra_pool, faltam))
    return sel


def selecionar_ph2(rows):
    el = [r for r in rows if r["contraste"] >= C_MIN and r["area"] >= A_MIN]
    sel = []
    for fc, k in COTAS_PH2.items():
        faixa = [r for r in el if faixa_contraste(r["contraste"]) == fc]
        print(f"  PH2 {fc}: pedidos={k} disponiveis={len(faixa)}")
        sel.extend(evenly_spaced(faixa, k))
    return sel


def observacoes(r):
    obs = []
    c, a = r["contraste"], r["area"]
    if c < C_BAIXO:
        obs.append("baixo contraste")
    elif c > C_ALTO:
        obs.append("alto contraste")
    if a < A_PEQ:
        obs.append("lesao pequena")
    elif a > A_GRA:
        obs.append("lesao grande")
    w, h = map(int, r["resolucao"].split("x"))
    if max(w, h) > 4000:
        obs.append("resolucao muito alta")
    elif max(w, h) < 1000:
        obs.append("resolucao baixa")
    if r.get("lesao_mais_escura") == "0":
        obs.append("lesao mais clara que a pele")
    return "; ".join(obs) if obs else "caso tipico"


def binariza(msk_path, destino):
    msk = Image.open(msk_path).convert("L")
    msk = msk.point(lambda p: 255 if p > 127 else 0)  # garante 0/255 estrito
    msk.save(destino, format="PNG")
    return msk.size


def main():
    os.makedirs(DIR_IMG, exist_ok=True)
    os.makedirs(DIR_MSK, exist_ok=True)

    print("Selecao ISIC:")
    sel_isic = selecionar_isic(carregar(os.path.join(SCRIPTS, "metricas_isic.csv")))
    print("Selecao PH2:")
    sel_ph2 = selecionar_ph2(carregar(os.path.join(SCRIPTS, "metricas_ph2.csv")))
    assert len(sel_isic) == 30, f"ISIC={len(sel_isic)}"
    assert len(sel_ph2) == 10, f"PH2={len(sel_ph2)}"

    metadados = []
    for r in sel_isic:
        nome_img = r["nome"] + ".jpg"
        nome_msk = r["nome"] + "_mask.png"
        shutil.copy2(os.path.join(ISIC_IMG, nome_img), os.path.join(DIR_IMG, nome_img))
        tam = binariza(os.path.join(ISIC_MSK, r["nome"] + "_segmentation.png"), os.path.join(DIR_MSK, nome_msk))
        assert tam == tuple(map(int, r["resolucao"].split("x"))), f"resolucao divergente {r['nome']}"
        metadados.append({
            "nome": nome_img, "base": "ISIC 2017", "formato": "JPG",
            "resolucao": r["resolucao"], "area_lesao_frac": f"{r['area']:.4f}",
            "contraste": f"{r['contraste']:.4f}", "observacoes": observacoes(r),
        })

    for r in sel_ph2:
        imd = r["nome"]  # IMDxxx
        nome_img = f"PH2_{imd}.bmp"
        nome_msk = f"PH2_{imd}_mask.png"
        shutil.copy2(
            os.path.join(PH2_ROOT, imd, f"{imd}_Dermoscopic_Image", f"{imd}.bmp"),
            os.path.join(DIR_IMG, nome_img),
        )
        tam = binariza(
            os.path.join(PH2_ROOT, imd, f"{imd}_lesion", f"{imd}_lesion.bmp"),
            os.path.join(DIR_MSK, nome_msk),
        )
        assert tam == tuple(map(int, r["resolucao"].split("x"))), f"resolucao divergente {imd}"
        metadados.append({
            "nome": nome_img, "base": "PH2", "formato": "BMP",
            "resolucao": r["resolucao"], "area_lesao_frac": f"{r['area']:.4f}",
            "contraste": f"{r['contraste']:.4f}", "observacoes": observacoes(r),
        })

    with open(META, "w", newline="", encoding="utf-8") as fh:
        campos = ["nome", "base", "formato", "resolucao", "area_lesao_frac", "contraste", "observacoes"]
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(metadados)
    print(f"\nMontado: {len(metadados)} pares; metadata em {META}")


if __name__ == "__main__":
    sys.exit(main())
