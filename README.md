# Análise Exploratória e Preparação de Dados para Modelos de Aprendizado de Máquina

## Identificação

**Aluna:** Isabelle Cavalcanti da Silva
**Matrícula:** 20250031684 
**Disciplina:** DCT1401 — Inteligência Artificial  
**Curso:** Sistemas de Informação — UFRN  
**Período:** 2026.2

## Sobre o projeto

Este projeto apresenta uma análise exploratória e a preparação de dados para problemas de aprendizado de máquina utilizando um conjunto de dados relacionado à degradação e falha de baterias de veículos elétricos.

A partir do mesmo conjunto de dados foram definidos dois problemas:

- **Regressão:** previsão da vida útil restante da bateria (`predicted_remaining_life_cycles`);
- **Classificação:** previsão da ocorrência de falha da bateria (`battery_failure`).

O projeto contempla análise exploratória, identificação de valores ausentes e outliers, análise de correlações, tratamento e transformação dos dados e investigação das características mais relevantes para os dois problemas.

## Dataset

Foi utilizado o **EV Battery Failure Prediction Dataset (200K)**, disponível publicamente no Kaggle.

**Fonte:**  
https://www.kaggle.com/datasets/sarveshchhetri/ev-battery-failure-prediction-dataset-200k

O conjunto utilizado possui **200.000 registros e 70 variáveis**, incluindo características dos veículos e baterias, padrões de carregamento e condução, condições ambientais, manutenção e indicadores relacionados à saúde e degradação das baterias.

O arquivo CSV não é armazenado diretamente neste repositório devido ao seu tamanho. Para executar os notebooks, faça o download do dataset no Kaggle e disponibilize o arquivo `ev_battery_failure_dataset.csv` no ambiente de execução.

## Estrutura do projeto

```text
projeto-ia/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_regression_analysis.ipynb
│   └── 04_classification_analysis.ipynb
├── src/
├── reports/
│   └── figures/
└── docs/

## Notebooks

### 01 — Análise Exploratória

Realiza a inspeção inicial do conjunto de dados, estatísticas descritivas, análise de variáveis categóricas, valores ausentes, outliers, correlações e análise inicial das variáveis-alvo.

### 02 — Pré-processamento

Realiza a preparação dos dados para aprendizado de máquina, incluindo remoção de identificadores, divisão entre treino e teste, imputação de valores ausentes, One-Hot Encoding das variáveis categóricas e padronização das características numéricas.

### 03 — Análise para Regressão

Analisa `predicted_remaining_life_cycles` como alvo de regressão, investigando sua distribuição e as características numéricas mais relacionadas à vida útil restante prevista da bateria.

### 04 — Análise para Classificação

Analisa `battery_failure` como alvo de classificação, incluindo distribuição das classes, correlações, boxplots, ANOVA e teste qui-quadrado para identificação de características associadas à ocorrência de falha.

## Principais resultados

A análise exploratória identificou valores ausentes em 67 variáveis, porém com percentuais relativamente baixos, aproximadamente entre 3% e 5%. Também foram observados valores extremos e forte multicolinearidade entre algumas características.

No problema de regressão, as características com maior correlação absoluta com `predicted_remaining_life_cycles` foram:

- `cell_voltage_avg`: -0,827;
- `thermal_runaway_risk`: -0,736;
- `thermal_health_score`: 0,698.

No problema de classificação, `battery_failure` apresentou forte desbalanceamento, com aproximadamente **90,04% dos registros sem falha** e **9,96% com falha**.

Entre as características numéricas mais associadas à falha destacaram-se `capacity_loss_percent`, `battery_health_percent`, `state_of_health`, `aging_score` e `cell_voltage_std`.

Entre as características categóricas, foram identificadas associações estatisticamente significativas com `fleet_or_private`, `battery_chemistry`, `vehicle_brand`, `vehicle_model` e `vehicle_type`.

Veículos classificados como `Fleet` apresentaram taxa de falha de aproximadamente **26,29%**, enquanto veículos `Private` apresentaram aproximadamente **3,60%**.

## Como executar

1. Faça o download do dataset no Kaggle.
2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Abra os notebooks disponíveis na pasta `notebooks/`.
4. Disponibilize o arquivo `ev_battery_failure_dataset.csv` no caminho utilizado pelos notebooks.
5. Execute os notebooks na seguinte ordem:

```text
01_eda.ipynb
02_preprocessing.ipynb
03_regression_analysis.ipynb
04_classification_analysis.ipynb
```

Os notebooks foram desenvolvidos utilizando Python e Google Colab.

## Dependências

As principais bibliotecas utilizadas são:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy
- jupyter