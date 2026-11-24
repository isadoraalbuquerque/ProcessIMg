#  encadeia todas as etapas — roda UMA imagem com passo a passo
# (--ver abre as figuras) ou TODAS as 40 gerando resultados/tabela_final.csv.
"""Pipeline completo de segmentação de lesões de pele (Passo 29).

Uso (com o venv ativado, a partir da raiz do projeto):
    python src/main.py                     -> modo lote: processa TODAS as imagens
                                              de dataset/imagens/ e escreve
                                              resultados/tabela_final.csv
    python src/main.py ISIC_0000145.jpg    -> modo demonstração: processa UMA imagem,
                                              salva contorno/máscara e imprime o passo a passo
    python src/main.py ISIC_0000145.jpg --ver
                                           -> o mesmo, e ABRE a janela com original,
                                              máscara e contorno (Matplotlib)

Pipeline oficial (decisões documentadas): canal S (vencedor do Passo 26, IoU médio
0,724) -> mediana k=5 -> Otsu próprio -> binarização com polaridade direta (a lesão
é a parte CLARA no S) -> pós-processamento -> contorno e métricas vs referência.

NOTA metodológica sobre tempo: o tempo_ms da tabela_final.csv inclui leitura, contorno
e cálculo das métricas — escopo DIFERENTE do tabela_metricas.csv do Passo 26 (só
canal->pós-processamento). Não cruzar os dois sem citar a diferença de escopo.
"""
import statistics
import sys
import time
from pathlib import Path

# Bootstrap: rodando 'python src/main.py' diretamente, o path começa em src/ —
# localizamos a raiz do projeto (mesmo padrão da célula 1 dos cadernos).
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import cv2

from src.leitura import carregar_imagem, nome_da_mascara, PASTA_IMAGENS, PASTA_RESULTADOS
from src.preprocessamento import para_hsv
from src.filtragem import filtrar_mediana
from src.segmentacao import binarizar, otsu
from src.morfologia import pos_processar
from src.caracteristicas import (maior_contorno, area_pixels, perimetro,
                                 circularidade, desenhar_contorno)
from src.avaliacao import iou, dice, medir

CANAL = "S"          # vencedor do comparativo do Passo 26 (IoU médio 0,724)
K_MEDIANA = 5        # janela provisória do Passo 9


def pipeline_canal(img_rgb):
    """Segmenta uma imagem RGB: canal S -> mediana k=5 -> Otsu -> máscara.

    Devolve (máscara bruta {0,255}, t* do Otsu). A polaridade é DIRETA: no canal
    S a lesão é a parte CLARA (mais saturada), então pixels >= t* -> 255.
    """
    canal = para_hsv(img_rgb)[:, :, 1].copy()  # cópia contígua, padrão do driver do lote
    canal = filtrar_mediana(canal, K_MEDIANA)  # remove pelos finos antes do Otsu
    t = otsu(canal)
    return binarizar(canal, t, invertida=False), t


def processar_imagem(nome: str) -> dict:
    """Processa UMA imagem: pipeline completo + métricas vs referência.

    Salva contorno e máscara em resultados/pipeline/. Devolve a linha de métricas.
    """
    caminho = PASTA_IMAGENS / nome
    tempos = {}

    img, tempos["leitura"] = medir(carregar_imagem, caminho)
    # pipeline_canal devolve (máscara, t*) e medir devolve (resultado, tempo):
    # o desempacotamento precisa casar as duas tuplas — senão 'bruta' vira tupla
    (bruta, t_otsu), tempos["segmentacao"] = medir(pipeline_canal, img)
    final, tempos["pos_processar"] = medir(pos_processar, bruta)
    contorno, tempos["contorno"] = medir(maior_contorno, final)
    ref, _ = medir(cv2.imread, str(nome_da_mascara(caminho)), cv2.IMREAD_GRAYSCALE)
    if ref is None:
        raise FileNotFoundError(f"máscara de referência não encontrada: {nome_da_mascara(caminho)}")

    v_iou, tempos["iou"] = medir(iou, final, ref)
    v_dice, tempos["dice"] = medir(dice, final, ref)
    area = area_pixels(final)
    per = perimetro(contorno)
    circ = circularidade(area, per)
    tempo_total = sum(tempos.values())

    # saídas visuais (cv2.imwrite espera BGR — passe-e-volta do Passo 2)
    pasta = PASTA_RESULTADOS / "pipeline"
    pasta.mkdir(parents=True, exist_ok=True)
    stem = Path(nome).stem
    cv2.imwrite(str(pasta / f"{stem}_mascara.png"), final)
    destacado = desenhar_contorno(img, contorno)
    cv2.imwrite(str(pasta / f"{stem}_contorno.png"),
                cv2.cvtColor(destacado, cv2.COLOR_RGB2BGR))

    return {"nome": nome, "t_otsu": t_otsu, "iou": v_iou, "dice": v_dice,
            "area_px": area, "perimetro_px": per, "circularidade": circ,
            "tempo_ms": tempo_total, "tempos": tempos,
            "final": final, "img": img, "contorno": contorno}


def modo_demonstracao(nome: str, ver: bool = False) -> None:
    """Processa UMA imagem imprimindo o passo a passo detalhado.

    Com ver=True, abre a janela do Matplotlib com original, máscara e contorno
    (mesma exibição dos cadernos) — além de salvar os arquivos.
    """
    r = processar_imagem(nome)

    print(f"=== {nome} | canal {CANAL} | mediana k={K_MEDIANA} ===")
    print(f"T* do Otsu:            {r['t_otsu']}")
    print(f"Área da lesão:         {r['area_px']:,} px")
    print(f"Perímetro:             {r['perimetro_px']:,.0f} px")
    print(f"Circularidade:         {r['circularidade']:.3f}")
    print(f"IoU vs referência:     {r['iou']:.3f}")
    print(f"Dice vs referência:    {r['dice']:.3f}")
    print(f"Tempo total:           {r['tempo_ms']:.0f} ms")
    for etapa, ms in sorted(r["tempos"].items(), key=lambda x: -x[1]):
        print(f"   {etapa:<14} {ms:>8.1f} ms")
    print(f"\nSaídas salvas em resultados/pipeline/ ({Path(nome).stem}_mascara.png e _contorno.png)")

    if ver:
        import matplotlib.pyplot as plt   # import tardio: o lote não precisa de interface gráfica

        painel = desenhar_contorno(r["img"], r["contorno"])
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        axes[0].imshow(r["img"])
        axes[0].set_title(f"{nome} — original")
        axes[1].imshow(r["final"], cmap="gray", vmin=0, vmax=255)
        axes[1].set_title(f"máscara final ({r['iou']:.3f} IoU)")
        axes[2].imshow(painel)
        axes[2].set_title("contorno destacado")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        plt.show()


def modo_lote() -> None:
    """O FOR: processa TODAS as imagens e escreve resultados/tabela_final.csv."""
    imagens = sorted(p.name for p in PASTA_IMAGENS.iterdir()
                     if p.suffix.lower() in {".jpg", ".bmp"})
    linhas = []
    inicio = time.perf_counter()

    for i, nome in enumerate(imagens, start=1):          # <<< O FOR do lote
        try:
            r = processar_imagem(nome)
        except (ValueError, FileNotFoundError) as e:
            # falha prevista por design (ex.: objeto ocupando a borda) — registra e segue
            linhas.append({"nome": nome, "iou": 0.0, "dice": 0.0,
                           "area_px": 0, "perimetro_px": 0.0, "circularidade": 0.0,
                           "tempo_ms": 0.0, "status": f"FALHA:{e}"})
            print(f"[{i:>2}/{len(imagens)}] {nome:<22} FALHA: {e}", flush=True)
            continue
        r["status"] = "OK"
        r.pop("final"); r.pop("img"); r.pop("contorno")  # lote não retém arrays (~89 MB/imagem)
        linhas.append(r)
        print(f"[{i:>2}/{len(imagens)}] {nome:<22} IoU={r['iou']:.3f} "
              f"Dice={r['dice']:.3f} ({r['tempo_ms']:.0f} ms)", flush=True)

    duracao = time.perf_counter() - inicio

    # tabela final
    pasta = PASTA_RESULTADOS
    pasta.mkdir(parents=True, exist_ok=True)
    with open(pasta / "tabela_final.csv", "w", encoding="utf-8") as f:
        f.write("nome,t_otsu,iou,dice,area_px,perimetro_px,circularidade,tempo_ms,status\n")
        for r in linhas:
            f.write(f"{r['nome']},{r.get('t_otsu', '')},{r.get('iou', 0.0):.4f},{r.get('dice', 0.0):.4f},"
                    f"{r.get('area_px', 0)},{r.get('perimetro_px', 0.0):.2f},"
                    f"{r.get('circularidade', 0.0):.4f},{r.get('tempo_ms', 0.0):.1f},"
                    f"{r.get('status', 'OK')}\n")

    # resumo (só sobre as execuções OK)
    oks = [r for r in linhas if r.get("status") == "OK"]
    ious = sorted(r["iou"] for r in oks)
    media = sum(ious) / len(ious)
    mediana = statistics.median(ious)
    print(f"\n=== RESUMO DO LOTE ({len(imagens)} imagens, canal {CANAL}) ===")
    print(f"IoU médio:   {media:.4f}")
    print(f"IoU mediano: {mediana:.4f}")
    print(f"Melhor:      {max(linhas, key=lambda r: r['iou'])['nome']} "
          f"(IoU {max(r['iou'] for r in linhas):.3f})")
    print(f"Pior:        {min(linhas, key=lambda r: r['iou'])['nome']} "
          f"(IoU {min(r['iou'] for r in linhas):.3f})")
    print(f"Tempo total: {duracao:.1f} s")
    print("Tabela salva em resultados/tabela_final.csv")


if __name__ == "__main__":
    nome = next((a for a in sys.argv[1:] if not a.startswith("--")), None)
    if nome is not None:
        try:
            modo_demonstracao(nome, ver="--ver" in sys.argv)
        except FileNotFoundError as e:
            print(f"Erro: {e}\nUso: python src/main.py <imagem.jpg> [--ver]  |  python src/main.py --todas")
    else:
        if "--ver" in sys.argv:
            print("Uso: python src/main.py <imagem.jpg> --ver   |   python src/main.py --todas")
        else:
            modo_lote()
