# lê as imagens do disco (já em RGB), casa cada imagem com sua máscara
# de referência, cria as pastas de resultados e audita a base (40/40 pares OK).
"""Leitura, auditoria e organização da base (Passos 2 e 4).

Convenção do projeto: toda imagem retornada por este módulo já está em RGB.
"""
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent          # .../ProcessIMg
PASTA_IMAGENS = RAIZ / "dataset" / "imagens"
PASTA_MASCARAS = RAIZ / "dataset" / "mascaras_referencia"
PASTA_RESULTADOS = RAIZ / "resultados"

# Árvore da seção 8 do plano — o pipeline salvará uma etapa em cada pasta
SUBPASTAS_RESULTADOS = ["originais", "canais", "cinza", "filtradas", "histogramas",
                        "segmentadas", "morfologia", "contornos", "hsv"]


def carregar_imagem(caminho) -> np.ndarray:
    """Carrega uma imagem do disco e retorna um ndarray RGB (altura, largura, 3).

    Levanta FileNotFoundError com mensagem útil se a leitura falhar.
    """
    img = cv2.imread(str(caminho))
    if img is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem: {caminho}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def nome_da_mascara(caminho_imagem: Path) -> Path:
    """Deriva o caminho da máscara: ISIC_0012940.jpg -> ISIC_0012940_mask.png."""
    return PASTA_MASCARAS / f"{caminho_imagem.stem}_mask.png"


def criar_pastas_resultados() -> None:
    """Cria a árvore resultados/ da seção 8 do plano (idempotente: roda quantas vezes quiser)."""
    for sub in SUBPASTAS_RESULTADOS:
        (PASTA_RESULTADOS / sub).mkdir(parents=True, exist_ok=True)
    print(f"Árvore de resultados pronta em: {PASTA_RESULTADOS}")


def auditar_base() -> list:
    """Confere os pares imagem<->máscara da base de teste.

    Verifica, para cada imagem em dataset/imagens/:
      1. se a máscara correspondente existe;
      2. se a leitura de ambos funciona (imread devolve None quando falha);
      3. se as resoluções (altura x largura) coincidem;
      4. se a máscara é estritamente binária (0/255).
    Retorna a lista de problemas (vazia = base OK).
    """
    problemas = []

    # glob nas DUAS extensões da base (ISIC vem .jpg, PH2 vem .bmp)
    imagens = sorted(p for p in PASTA_IMAGENS.iterdir()
                     if p.is_file() and p.suffix.lower() in {".jpg", ".bmp"})
    n_isic = sum(1 for p in imagens if p.name.startswith("ISIC"))

    for img_path in imagens:
        mask_path = nome_da_mascara(img_path)

        # 1) existência — sem máscara, sem IoU/Dice no futuro
        if not mask_path.exists():
            problemas.append(f"{img_path.name}: máscara NÃO encontrada ({mask_path.name})")
            continue

        # 2) leitura — imread cru: auditoria não exibe nada, conversão é desnecessária
        img = cv2.imread(str(img_path))
        msk = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)   # máscara interessa em 1 canal
        if img is None or msk is None:
            problemas.append(f"{img_path.name}: falha na leitura (img ok={img is not None}, mask ok={msk is not None})")
            continue

        # 3) as grades de pixels coincidem? shape[:2] ignora canais de propósito
        if img.shape[:2] != msk.shape[:2]:
            problemas.append(f"{img_path.name}: resolução diferente — imagem {img.shape[1]}x{img.shape[0]}, máscara {msk.shape[1]}x{msk.shape[0]}")

        # 4) binariedade ESTRITA: TODOS os valores devem pertencer a {0, 255}.
        #    O teste por extremos (min/max) deixaria passar uma máscara {0,128,255}.
        if not np.isin(msk, (0, 255)).all():
            n_valores = len(np.unique(msk))
            problemas.append(f"{img_path.name}: máscara não é binária ({n_valores} valores distintos além de 0/255)")

    # relatório final (padrão acumulador: falhas viram lista, não prints soltos)
    print("=== AUDITORIA DA BASE ===")
    print(f"Imagens encontradas: {len(imagens)}  (ISIC: {n_isic} | PH2: {len(imagens) - n_isic})")
    ok = len(imagens) - len(problemas)
    if problemas:
        for p in problemas:
            print("  [ERRO]", p)
        print(f"Pares OK: {ok}/{len(imagens)} — corrija antes de avançar")
    else:
        print(f"Pares OK: {ok}/{len(imagens)} — base íntegra, pode avançar")
    return problemas


if __name__ == "__main__":
    criar_pastas_resultados()
    problemas = auditar_base()
