"""Passo 26 — Lote completo: 40 imagens x 3 canais (cinza, B, S) -> métricas.

Driver de execução: NÃO altera nada em src/, apenas importa e roda o pipeline
oficial do projeto (mediana k=5 -> Otsu -> binarização -> morfologia) e consolida
as métricas calculadas pelas funções do próprio usuário (iou, dice, area_pixels,
perimetro, circularidade).

Cada linha da tabela = imagem x canal:
  canal (cinza/B/S) -> filtrar_mediana(k=5) -> otsu(t*) -> binarizar(polaridade)
  -> pos_processar -> iou/dice vs. máscara de referência + características.

Saídas (todas em resultados/):
  tabela_metricas.csv     — 120 linhas (40 imagens x 3 canais)
  resumo_canais.csv       — média/mediana de IoU/Dice por canal + veredicto
  melhores_piores.csv     — 3 melhores e 3 piores casos do canal vencedor
  fig_iou_por_canal.png   — boxplot + barras de IoU por canal
  fig_grade_melhores_piores.png — sobreposição dos 6 casos extremos
"""
import csv
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent   # portável: raiz = pai de scripts/
sys.path.insert(0, str(RAIZ))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")            # backend sem tela: roda em lote, só salva arquivo
import matplotlib.pyplot as plt

# --- pipeline oficial do projeto (somente importação; nada é alterado) ---
from src.leitura import carregar_imagem, nome_da_mascara, PASTA_IMAGENS
from src.preprocessamento import para_cinza, para_hsv
from src.filtragem import filtrar_mediana
from src.segmentacao import binarizar, otsu, sobrepor
from src.morfologia import pos_processar
from src.caracteristicas import (maior_contorno, area_pixels, perimetro,
                                 circularidade, bounding_box)
from src.avaliacao import iou, dice, medir

PASTA_RESULTADOS = RAIZ / "resultados"
PASTA_MASCARAS = RAIZ / "dataset" / "mascaras_referencia"
ARQ_METADATA = RAIZ / "dataset" / "metadata.csv"

# Canais candidatos e a polaridade da binarização (invertida=True -> pixels < t*
# viram objeto; usada quando a lesão é MAIS ESCURA que o fundo).
CANAIS = [
    ("cinza", True),    # luminância BT.601 — lesão escura
    ("B", True),        # canal azul do RGB — lesão escura
    ("S", False),       # saturação do HSV — lesão mais CLARA (saturada) que o fundo
]

COLUNAS_TABELA = ["nome", "base", "canal", "t_otsu", "iou", "dice", "area_px",
                  "perimetro_px", "circularidade", "n_objetos_brutos",
                  "tempo_ms", "status"]


def ler_metadata():
    """Lê dataset/metadata.csv (curadoria) para enriquecer as hipóteses de causa."""
    meta = {}
    with open(ARQ_METADATA, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            meta[row["nome"]] = row
    return meta


def extrator(canal_nome, img):
    """Devolve uma função sem argumentos que extrai o canal, já contíguo em uint8.

    .copy() é necessário porque fatias de NumPy (img[:,:,2], hsv[:,:,1]) não são
    contíguas e o cv2.medianBlur exige memória contígua — isso é tratamento de
    dados do driver, não alteração do pipeline.
    """
    if canal_nome == "cinza":
        return lambda: para_cinza(img)
    if canal_nome == "B":
        return lambda: img[:, :, 2].copy()
    return lambda: para_hsv(img)[:, :, 1].copy()


def rodar_canal(img, msk_ref, canal_nome, invertida):
    """Executa canal -> mediana -> Otsu -> binarizar -> pos_processar para UM canal.

    Retorna (linha_de_resultado, mascara_final_ou_None). Erros previstos do
    pipeline (ValueError em pos_processar/maior_contorno) viram status FALHA.
    """
    tempo_ms = 0.0
    t_otsu = ""
    n_brutos = ""

    try:
        # etapa 1: extração do canal (cinza = BT.601; B = fatia; S = HSV)
        canal, ms = medir(extrator(canal_nome, img))
        tempo_ms += ms
        # etapa 2: mediana k=5 (remove pelos finos/ruído preservando bordas)
        canal, ms = medir(filtrar_mediana, canal, 5)
        tempo_ms += ms
        # etapa 3: limiar de Otsu (maximiza variância entre classes)
        t_otsu, ms = medir(otsu, canal)
        tempo_ms += ms
        # etapa 4: binarização com a polaridade do canal
        bruta, ms = medir(binarizar, canal, t_otsu, invertida=invertida)
        tempo_ms += ms
        # diagnóstico: quantas 'ilhas' a limiarização crua produziu
        n_brutos = cv2.connectedComponentsWithStats(bruta, 8)[0] - 1
        # etapa 5: morfologia (abertura 5 + fechamento 7 + maior componente +
        # preencher buracos). Pode levantar ValueError se o objeto ocupar a borda.
        final, ms = medir(pos_processar, bruta)
        tempo_ms += ms
    except ValueError as e:
        return {**dict.fromkeys(COLUNAS_TABELA, ""),
                "status": f"FALHA:{e}",
                "t_otsu": t_otsu,
                "tempo_ms": f"{tempo_ms:.1f}",
                "n_objetos_brutos": n_brutos}, None, tempo_ms, f"FALHA:{e}", t_otsu, n_brutos

    linha = {
        "t_otsu": t_otsu,
        "iou": f"{iou(final, msk_ref):.4f}",
        "dice": f"{dice(final, msk_ref):.4f}",
        "n_objetos_brutos": n_brutos,
        "tempo_ms": f"{tempo_ms:.1f}",
        "status": "OK",
    }
    try:
        contorno = maior_contorno(final)            # erro se máscara vazia
        area = area_pixels(final)
        per = perimetro(contorno)
        linha["area_px"] = str(area)
        linha["perimetro_px"] = f"{per:.2f}"
        linha["circularidade"] = f"{circularidade(area, per):.4f}"
    except ValueError as e:
        linha["status"] = f"FALHA:{e}"
        return linha, None, tempo_ms, linha["status"], t_otsu, n_brutos
    return linha, final, tempo_ms, "OK", t_otsu, n_brutos


def main():
    inicio_lote = time.perf_counter()
    PASTA_RESULTADOS.mkdir(exist_ok=True)

    imagens = sorted(p for p in PASTA_IMAGENS.iterdir()
                     if p.is_file() and p.suffix.lower() in {".jpg", ".bmp"})
    print(f"Lote: {len(imagens)} imagens x {len(CANAIS)} canais = "
          f"{len(imagens) * len(CANAIS)} execuções\n", flush=True)

    meta = ler_metadata()
    linhas = []            # cada item: dict com nome, base, canal + resultado
    mascaras_finais = {}   # (nome, canal) -> máscara final (para a figura)

    for idx, caminho in enumerate(imagens, 1):
        base = "ISIC" if caminho.name.startswith("ISIC") else "PH2"
        try:
            img = carregar_imagem(caminho)
            msk_ref = cv2.imread(str(nome_da_mascara(caminho)), cv2.IMREAD_GRAYSCALE)
            if msk_ref is None:
                raise ValueError("máscara de referência não encontrada ou ilegível")
        except (FileNotFoundError, ValueError) as e:
            for nome_canal, _ in CANAIS:
                linhas.append({"nome": caminho.name, "base": base,
                               "canal": nome_canal, "status": f"FALHA:{e}",
                               "n_objetos_brutos": "", "tempo_ms": "0.0"})
            print(f"[{idx:2d}/40] {caminho.name}: FALHA na carga ({e})", flush=True)
            continue

        print(f"[{idx:2d}/40] {caminho.name} ({base}, {img.shape[1]}x{img.shape[0]})", flush=True)
        for canal_nome, invertida in CANAIS:
            linha, final, tempo_ms, status, t_otsu, n_brutos = rodar_canal(
                img, msk_ref, canal_nome, invertida)
            linha.update({"nome": caminho.name, "base": base, "canal": canal_nome})
            # preencher chaves ausentes (caso de falha precoce)
            linha = {c: linha.get(c, "") for c in COLUNAS_TABELA}
            linhas.append(linha)
            if status == "OK":
                mascaras_finais[(caminho.name, canal_nome)] = final
                print(f"    {canal_nome:>5}: t*={t_otsu:>3}  IoU={linha['iou']}  "
                      f"Dice={linha['dice']}  {tempo_ms:.0f} ms  ({n_brutos} obj. brutos)",
                      flush=True)
            else:
                print(f"    {canal_nome:>5}: {status}", flush=True)
        del img, msk_ref

    tempo_lote_ms = (time.perf_counter() - inicio_lote) * 1000

    # ---------- 1) tabela_metricas.csv ----------
    with open(PASTA_RESULTADOS / "tabela_metricas.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS_TABELA)
        w.writeheader()
        w.writerows(linhas)

    # ---------- 2) resumo_canais.csv ----------
    resumo = []
    for canal_nome, _ in CANAIS:
        ok = [l for l in linhas if l["canal"] == canal_nome and l["status"] == "OK"]
        falhas = [l for l in linhas if l["canal"] == canal_nome and l["status"] != "OK"]
        ious = np.array([float(l["iou"]) for l in ok], dtype=float)
        dices = np.array([float(l["dice"]) for l in ok], dtype=float)
        tempos = np.array([float(l["tempo_ms"]) for l in ok], dtype=float)
        resumo.append({
            "canal": canal_nome,
            "n_ok": len(ok),
            "n_falhas": len(falhas),
            "media_iou": f"{ious.mean():.4f}",
            "mediana_iou": f"{np.median(ious):.4f}",
            "media_dice": f"{dices.mean():.4f}",
            "mediana_dice": f"{np.median(dices):.4f}",
            "tempo_total_ms": f"{tempos.sum():.0f}",
        })
    vencedor = max(resumo, key=lambda r: float(r["media_iou"]))["canal"]
    for r in resumo:
        r["veredicto"] = ("VENCEDOR (maior IoU médio)" if r["canal"] == vencedor else "")
    with open(PASTA_RESULTADOS / "resumo_canais.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["canal", "n_ok", "n_falhas", "media_iou",
                                          "mediana_iou", "media_dice", "mediana_dice",
                                          "tempo_total_ms", "veredicto"])
        w.writeheader()
        w.writerows(resumo)

    # ---------- 3) melhores_piores.csv (canal vencedor) ----------
    do_canal = [l for l in linhas if l["canal"] == vencedor and l["status"] == "OK"]
    ordenado = sorted(do_canal, key=lambda l: float(l["iou"]), reverse=True)
    extremos = [("melhor", l) for l in ordenado[:3]] + \
               [("pior", l) for l in ordenado[-3:]]

    def observacao(linha):
        """Hipótese curta de causa: curadoria + área/contraste + lesão na borda."""
        m = meta.get(linha["nome"], {})
        partes = []
        if m.get("observacoes"):
            partes.append(m["observacoes"])
        if m.get("area_lesao_frac"):
            partes.append(f"área={float(m['area_lesao_frac']):.1%}")
        # bounding_box (código do usuário) na máscara de referência detecta
        # lesão encostada na borda da imagem
        ref = cv2.imread(str(PASTA_MASCARAS / f"{Path(linha['nome']).stem}_mask.png"),
                         cv2.IMREAD_GRAYSCALE)
        if ref is not None:
            x, y, bw, bh = bounding_box(ref)
            h, wid = ref.shape
            if x == 0 or y == 0 or x + bw >= wid or y + bh >= h:
                partes.append("lesão toca a borda")
        return "; ".join(partes)

    with open(PASTA_RESULTADOS / "melhores_piores.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["classificacao", "nome", "iou", "dice", "area_px", "observacao"])
        for rotulo, l in extremos:
            w.writerow([rotulo, l["nome"], l["iou"], l["dice"], l["area_px"],
                        observacao(l)])

    # ---------- 4a) figura IoU por canal ----------
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.5))
    dados = [np.array([float(l["iou"]) for l in linhas
                       if l["canal"] == c and l["status"] == "OK"]) for c, _ in CANAIS]
    nomes = [c for c, _ in CANAIS]
    bp = eixos[0].boxplot(dados, tick_labels=nomes, patch_artist=True)
    for caixa, cor in zip(bp["boxes"], ["#8ecae6", "#ffb703", "#fb8500"]):
        caixa.set_facecolor(cor)
    eixos[0].set_title("IoU por canal (casos OK)")
    eixos[0].set_ylabel("IoU")
    eixos[0].grid(axis="y", alpha=0.3)
    medias = [d.mean() for d in dados]
    barras = eixos[1].bar(nomes, medias, color=["#8ecae6", "#ffb703", "#fb8500"])
    for b, m in zip(barras, medias):
        eixos[1].text(b.get_x() + b.get_width() / 2, m + 0.01, f"{m:.3f}",
                      ha="center", fontsize=10)
    eixos[1].set_title(f"IoU médio por canal — vencedor: {vencedor}")
    eixos[1].set_ylim(0, 1)
    eixos[1].grid(axis="y", alpha=0.3)
    fig.suptitle("Passo 26 — comparação de canais (mediana k=5 + Otsu + morfologia)")
    fig.tight_layout()
    fig.savefig(PASTA_RESULTADOS / "fig_iou_por_canal.png", dpi=150)
    plt.close(fig)

    # ---------- 4b) grade de sobreposição dos extremos ----------
    fig, eixos = plt.subplots(2, 3, figsize=(13, 8))
    for col, (_, l) in enumerate(extremos[:3]):
        eixos[0][col].set_title(f"MELHOR {col+1}: {l['nome']} — IoU={l['iou']}",
                                fontsize=9)
    for col, (_, l) in enumerate(extremos[3:]):
        eixos[1][col].set_title(f"PIOR {col+1}: {l['nome']} — IoU={l['iou']}",
                                fontsize=9)
    for ax in eixos.ravel():
        ax.axis("off")
    for pos, (_, l) in enumerate(extremos):
        ax = eixos[pos // 3][pos % 3]
        caminho = PASTA_IMAGENS / l["nome"]
        img = carregar_imagem(caminho)
        final = mascaras_finais.get((l["nome"], vencedor))
        if final is None:   # reexecuta o canal vencedor se a máscara não está em memória
            _, final, _, _, _, _ = rodar_canal(
                img, cv2.imread(str(nome_da_mascara(caminho)), cv2.IMREAD_GRAYSCALE),
                vencedor, dict(CANAIS)[vencedor])
        sobre = sobrepor(img, final)
        # reduz para exibição (imagens ISIC chegam a ~29 MP)
        escala = 500 / sobre.shape[1]
        sobre = cv2.resize(sobre, (500, int(sobre.shape[0] * escala)),
                           interpolation=cv2.INTER_AREA)
        ax.imshow(sobre)
        del img, final, sobre
    fig.suptitle(f"Passo 26 — melhores (linha 1) e piores (linha 2) casos — canal {vencedor}")
    fig.tight_layout()
    fig.savefig(PASTA_RESULTADOS / "fig_grade_melhores_piores.png", dpi=150)
    plt.close(fig)

    # ---------- relatório no stdout ----------
    n_ok = sum(1 for l in linhas if l["status"] == "OK")
    n_falha = len(linhas) - n_ok
    print("\n=== RESUMO DO PASSO 26 ===")
    print(f"Linhas: {len(linhas)} | OK: {n_ok} | FALHA: {n_falha}")
    for r in resumo:
        print(f"  {r['canal']:>5}: IoU médio={r['media_iou']} (mediana {r['mediana_iou']}) | "
              f"Dice médio={r['media_dice']} (mediana {r['mediana_dice']}) | "
              f"falhas={r['n_falhas']} | tempo={float(r['tempo_total_ms'])/1000:.1f}s")
    print(f"VEREDICTO: canal vencedor por IoU médio = {vencedor}")
    print(f"Tempo total do lote: {tempo_lote_ms/1000:.1f} s | "
          f"média por imagem: {tempo_lote_ms/1000/len(imagens):.1f} s")
    falhas = [l for l in linhas if l["status"] != "OK"]
    if falhas:
        print("Falhas registradas:")
        for l in falhas:
            print(f"  {l['nome']} [{l['canal']}]: {l['status']}")


if __name__ == "__main__":
    main()
