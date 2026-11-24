# Segmentação de Lesões de Pele em Imagens Dermatoscópicas

**Projeto de Processamento Digital de Imagens (PDI) — técnicas clássicas, sem inteligência artificial.**

O câncer de pele tem prognóstico fortemente associado à detecção precoce, e a delimitação da lesão em relação à pele ao redor é a etapa central da análise de uma imagem dermatoscópica — é dela que saem as medidas de área, perímetro e forma usadas em critérios clínicos de triagem (como o ABCD). Este projeto desenvolve, em Python, um pipeline completo que recebe uma imagem dermatoscópica e produz a **máscara binária da lesão e seu contorno destacado**, com medidas de forma (área, perímetro, circularidade) e métricas de sobreposição contra máscaras de referência (IoU e Dice), usando filtragem por mediana, **limiarização de Otsu implementada do zero com NumPy**, morfologia matemática e descrição de formas — sem redes neurais nem aprendizado de máquina.

## Requisitos

- Python 3.14 ou superior
- Bibliotecas: `numpy`, `opencv-python-headless` (OpenCV), `matplotlib`
- Para os cadernos de exploração: `jupyterlab` e `ipykernel`

Todas as dependências de execução estão fixadas em `requirements.txt`; a fotografia completa do ambiente (transitivas incluídas) está em `requirements-freeze.txt`.

## Instalação

A partir da raiz do projeto:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Estrutura de pastas

```text
ProcessIMg/
├── dataset/
│   ├── imagens/                  # 40 imagens de teste (30 ISIC 2017 .jpg + 10 PH2 .bmp)
│   ├── mascaras_referencia/      # 40 máscaras PNG binárias 0/255, pareadas por nome
│   └── metadata.csv              # nome, base, formato, resolução, área da lesão, contraste
├── src/                          # motor modular (um módulo por funcionalidade)
│   ├── leitura.py                # aquisição BGR→RGB, auditoria da base (40/40 pares)
│   ├── preprocessamento.py       # canais RGB, cinza BT.601, HSV, histograma próprio
│   ├── filtragem.py              # filtro de mediana (pelos)
│   ├── segmentacao.py            # limiarização por degrau, Otsu próprio, máscara, sobreposição
│   ├── morfologia.py             # erosão/dilatação, abertura/fechamento, pós-processamento
│   ├── caracteristicas.py        # contorno, área, perímetro, circularidade, bounding box
│   ├── avaliacao.py              # IoU e Dice próprios, medição de tempo
│   └── main.py                   # integração ponta a ponta (3 modos de execução)
├── scripts/
│   ├── passo26_lote.py           # driver do lote: 40 imagens × 3 canais (cinza, B, S)
│   └── preprocessar_tudo.py      # painéis de pré-processamento das 40 imagens
├── notebooks/                    # cadernos Jupyter numerados (05_canais … 25_tempo + exploratoria_dataset)
├── curadoria_img_scripts/        # scripts reproduzíveis de montagem da base
├── resultados/
│   ├── originais/ canais/ cinza/ hsv/ histogramas/ filtradas/ exploratoria/   # Fases 1–2
│   ├── pre_processamento/        # painéis de etapas das 40 imagens (preprocessar_tudo.py)
│   ├── segmentadas/ morfologia/                                               # Fase 3
│   ├── contornos/                                                             # Fase 4
│   ├── pipeline/                 # máscara + contorno das 40 imagens (main.py)
│   ├── tabela_metricas.csv       # 120 linhas (40 imagens × 3 canais)
│   ├── tabela_final.csv          # 40 linhas do pipeline oficial (canal S)
│   ├── resumo_canais.csv         # veredicto por canal
│   ├── melhores_piores.csv       # 3 melhores e 3 piores casos
│   └── fig_iou_por_canal.png / fig_grade_melhores_piores.png
├── requirements.txt              # dependências diretas de execução
└── requirements-freeze.txt       # fotografia completa do ambiente (pip freeze)
```

## Como executar

Com o venv ativado, sempre **a partir da raiz do projeto**.

### Pipeline ponta a ponta (`src/main.py`) — 3 modos

```bash
# 1) Modo lote: processa TODAS as 40 imagens no canal S (pipeline oficial)
#    e escreve resultados/tabela_final.csv + máscaras/contornos em resultados/pipeline/
python src/main.py

# 2) Modo demonstração: processa UMA imagem, imprime o passo a passo
#    (T* do Otsu, área, perímetro, circularidade, IoU, Dice, tempo por etapa)
python src/main.py ISIC_0000145.jpg

# 3) Modo --ver: o mesmo, e abre a janela com original, máscara e contorno
python src/main.py ISIC_0000145.jpg --ver
```

Pipeline oficial: canal S do HSV (vencedor do comparativo por IoU) → mediana k=5 → Otsu (implementação própria) → binarização com polaridade direta (a lesão é a parte clara no S) → pós-processamento morfológico → contorno e métricas contra a máscara de referência.

### Auditoria da base

```bash
# confere os 40 pares imagem↔máscara (existência, resolução, binariedade estrita 0/255)
python src/leitura.py
```

### Lote com os três canais (reproduz o comparativo cinza × B × S)

```bash
# 40 imagens × 3 canais = 120 execuções; gera tabela_metricas.csv,
# resumo_canais.csv, melhores_piores.csv e as 2 figuras comparativas
python scripts/passo26_lote.py
```

### Cadernos Jupyter (exploração passo a passo)

```bash
source .venv/bin/activate && jupyter lab
```

Os cadernos em `notebooks/` são numerados por passo do plano (`05_canais.ipynb` a `25_tempo.ipynb`) e importam as funções de `src/` — nenhum código é duplicado neles.

## Destaques dos resultados

- **Canal S vence o comparativo** (IoU médio nas 40 imagens): **0,7242** (S) contra 0,6432 (B) e 0,5288 (cinza); IoU mediano 0,8422 no pipeline oficial.
- Melhor caso: ISIC_0000145 (IoU 0,987; Dice 0,994). Pior caso: ISIC_0014346 (IoU 0,011) — lesão de 1,1% da imagem; os três piores casos compartilham o padrão de falha "lesão pequena → inundação".
- Números completos em `resultados/` (`tabela_final.csv`, `tabela_metricas.csv`, `resumo_canais.csv`, `melhores_piores.csv`).

## Implementações próprias (diferencial)

As seguintes operações foram implementadas do zero com NumPy e **validadas contra as equivalentes do OpenCV**, seguindo a preferência do professor por soluções desenvolvidas do zero:

| Operação | Validação |
|---|---|
| Histograma (`np.bincount`) | idêntico ao `cv2.calcHist` |
| Conversão para cinza BT.601 (manual) | diferença máxima de 1 nível vs `cv2.cvtColor` |
| **Otsu** (varredura vetorizada de σ²_B) | t\* idêntico ao `cv2.THRESH_OTSU` em 5/5 imagens de teste |
| **Circularidade** (4πA/P²) | ordenação correta de formas sintéticas (círculo > quadrado > retângulo) |
| **IoU** | testes de sanidade (idênticas 1,0; invertidas 0,0; deslocada 0,143; vazias sem divisão por zero) |
| **Dice** | propriedade Dice ≥ IoU verificada em 500 pares aleatórios |

