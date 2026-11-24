#limpa a máscara binária — erosão, dilatação, abertura, fechamento,
# mantém só o maior objeto (a lesão) e preenche buracos internos.
"""Morfologia matemática (Fase 3 — Passos 15 a 17)."""
import cv2
import numpy as np


def elemento_estruturante(k: int, forma: str = "ellipse") -> np.ndarray:
    """Cria o 'pincel' da morfologia: ellipse, rect ou cross, k×k.

    ellipse preserva cantos arredondados (ideal para formas orgânicas como
    lesões); rect produz efeitos com cantos quadrados.
    """
    formas = {"ellipse": cv2.MORPH_ELLIPSE, "rect": cv2.MORPH_RECT, "cross": cv2.MORPH_CROSS}
    if forma not in formas:
        raise ValueError(f"forma deve ser ellipse/rect/cross (recebido: {forma})")
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"k deve ser inteiro >= 1 (recebido: {k})")
    return cv2.getStructuringElement(formas[forma], (k, k))


def erosao(msk: np.ndarray, k: int = 5, forma: str = "ellipse") -> np.ndarray:
    """Encolhe o objeto: pixel sobrevive só se o pincel inteiro couber dentro dele.

    Apaga estruturas mais finas que o pincel — slides 6_1: 23–28.
    """
    return cv2.erode(msk, elemento_estruturante(k, forma))


def dilatacao(msk: np.ndarray, k: int = 5, forma: str = "ellipse") -> np.ndarray:
    """Expande o objeto: basta o pincel tocar o objeto para o pixel vizinho crescer.

    Engorda bordas e une pontas próximas — slides 6_1: 29–34.
    """
    return cv2.dilate(msk, elemento_estruturante(k, forma))


def abertura(msk: np.ndarray, k: int = 5, forma: str = "ellipse") -> np.ndarray:
    """Erosão -> dilatação: remove o que é mais fino que o pincel e devolve o tamanho.

    Equivalente a cv2.morphologyEx(msk, cv2.MORPH_OPEN, ...) — aqui composta
    explicitamente com as nossas erosao/dilatacao, seguindo os slides 6_1: 37–40.
    """
    return dilatacao(erosao(msk, k, forma), k, forma)


def fechamento(msk: np.ndarray, k: int = 5, forma: str = "ellipse") -> np.ndarray:
    """Dilatação -> erosão: une falhas estreitas e fecha buracos menores que o pincel.

    Equivalente a cv2.morphologyEx(msk, cv2.MORPH_CLOSE, ...) — slides 6_1: 41–49.
    """
    return erosao(dilatacao(msk, k, forma), k, forma)


def limpar_mascara(msk: np.ndarray, k_abertura: int = 5, k_fechamento: int = 7) -> np.ndarray:
    """Limpeza padrão do pipeline: abertura mata salpicados/pelos; fechamento fecha buraquinhos.

    k_fechamento deve superar o diâmetro dos buracos a fechar.
    """
    return fechamento(abertura(msk, k_abertura), k_fechamento)


def maior_componente(msk: np.ndarray, conectividade: int = 8) -> np.ndarray:
    """Mantém apenas o maior componente conexo (a lesão de verdade).

    connectedComponentsWithStats rotula cada 'ilha' branca e devolve a área de
    cada uma; mantemos o rótulo de maior área (rótulo 0 é o fundo) — slides 6_1: 68–74.
    """
    if conectividade not in (4, 8):
        raise ValueError(f"conectividade deve ser 4 ou 8 (recebido: {conectividade})")
    n, rotulos, stats, _ = cv2.connectedComponentsWithStats(msk, conectividade)
    if n <= 1:                       # só fundo: máscara vazia, nada a fazer
        return msk
    areas = stats[1:, cv2.CC_STAT_AREA]
    maior = 1 + int(np.argmax(areas))
    return (rotulos == maior).astype(np.uint8) * 255


def preencher_buracos(msk: np.ndarray) -> np.ndarray:
    """Fecha buracos internos: floodFill inunda o fundo pela borda (slides 6_1: 62–67).

    A semente é um pixel de FUNDO na borda da imagem — nunca fixa em (0,0), que
    pode pertencer ao objeto (nesse caso a inundação não parte e a máscara
    inteira viraria branca, silenciosamente). Tudo que sobra escuro após a
    inundação está encerrado pelo objeto — são os buracos; a união com a
    máscara os preenche. Se o objeto ocupar a borda inteira, não há fundo
    acessível e um ValueError é levantado (premissa da análise do Passo 17).
    """
    h, w = msk.shape
    borda = [(x, 0) for x in range(w)] + [(x, h - 1) for x in range(w)] \
          + [(0, y) for y in range(h)] + [(w - 1, y) for y in range(h)]
    semente = next(((x, y) for x, y in borda if msk[y, x] == 0), None)
    if semente is None:
        raise ValueError("preencher_buracos: o objeto ocupa toda a borda — não há fundo acessível pela borda")

    inundada = msk.copy()
    molde = np.zeros((h + 2, w + 2), dtype=np.uint8)   # moldura exigida pelo floodFill
    cv2.floodFill(inundada, molde, semente, 255)
    buracos = cv2.bitwise_not(inundada)
    return cv2.bitwise_or(msk, buracos)


def pos_processar(msk: np.ndarray, k_abertura: int = 5, k_fechamento: int = 7) -> np.ndarray:
    """Pós-processamento completo do pipeline: limpeza morfológica -> maior
    componente -> buracos preenchidos. É esta função que o lote da Fase 4 usará.
    """
    um_objeto = maior_componente(limpar_mascara(msk, k_abertura, k_fechamento))
    return preencher_buracos(um_objeto)