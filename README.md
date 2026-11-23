# Projeto de Inteligência Artificial — Predição de Falhas em Baterias de Veículos Elétricos

**DCT1401 — Inteligência Artificial | Sistemas de Informação — UFRN**

## Identificação

- **Aluna:** Isabelle Cavalcanti da Silva
- **Matrícula:** 20250031684
- **Disciplina:** DCT1401 — Inteligência Artificial
- **Curso:** Sistemas de Informação — UFRN
- **Período:** 2026.2

---

## 1. Sobre o projeto

Este projeto utiliza técnicas de aprendizado de máquina para analisar dados relacionados à condição de baterias de veículos elétricos e desenvolver modelos capazes de realizar duas tarefas:

- **Regressão:** estimar a vida útil restante da bateria, representada pela variável `predicted_remaining_life_cycles`.
- **Classificação:** prever a ocorrência de falha na bateria, representada pela variável `battery_failure`.

O trabalho foi desenvolvido em duas etapas principais:

1. **Análise exploratória e preparação dos dados:** investigação das características do conjunto de dados, identificação de padrões, tratamento de dados e preparação das variáveis para modelagem.
2. **Redes neurais artificiais com PyTorch:** implementação de redes neurais do tipo MLP (Multilayer Perceptron), realização de experimentos com hiperparâmetros, comparação de configurações e avaliação dos modelos finais.

## 2. Conjunto de dados

Foi utilizado o conjunto **EV Battery Failure Prediction Dataset (200K)**, disponibilizado no Kaggle:

[EV Battery Failure Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/sarveshchhetri/ev-battery-failure-prediction-dataset-200k)

O conjunto original contém aproximadamente 200 mil registros e 70 variáveis. Para as atividades, foram utilizadas 66 variáveis preditoras, além das variáveis-alvo correspondentes às tarefas de classificação e regressão.

O arquivo original deve estar localizado em:

```text
data/raw/ev_battery_failure_dataset.csv
```

Os conjuntos preparados para cada tarefa estão em:

```text
data/processed/dataset_classification.csv
data/processed/dataset_regression.csv
```

O conjunto de classificação possui 200.000 registros, enquanto o conjunto de regressão possui 191.026 registros.

As variáveis preditoras incluem características numéricas e categóricas relacionadas à bateria, como tensão, temperatura, capacidade, eficiência de carregamento, ciclos de uso e condições operacionais.

## 3. Estrutura do projeto

```text
projeto-ia/
│
├── data/
│   ├── raw/
│   │   └── ev_battery_failure_dataset.csv
│   │
│   └── processed/
│       ├── dataset_classification.csv
│       └── dataset_regression.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_regression_analysis.ipynb
│   └── 04_classification_analysis.ipynb
│
├── src/
│   ├── data_prep.py
│   ├── profiling.py
│   ├── metrics.py
│   ├── models.py
│   ├── inference.py
│   │
│   └── pytorch/
│       ├── preprocessing.py
│       ├── dataset.py
│       ├── models.py
│       ├── train.py
│       ├── train_regression.py
│       ├── evaluate.py
│       ├── evaluate_regression.py
│       ├── run_experiments.py
│       ├── run_interaction.py
│       ├── run_final_search.py
│       ├── plot_experiments.py
│       └── evaluate_final.py
│
├── experiments/
│   ├── classification/
│   └── regression/
│
├── models/
│   ├── classification/
│   └── regression/
│
reports/
├── figures/
│   ├── classification/
│   └── regression/
├── predictions/
├── tables/
└── final_test_metrics.json
│
├── docs/
│   ├── Relatorio_Projeto_IA_II_.pdf
│   └── relatorio_final.pdf
│
├── requirements.txt
└── README.md
```

Os diretórios `experiments/`, `models/` e `reports/` armazenam os resultados dos experimentos, os modelos treinados e os materiais utilizados na análise.

## 4. Etapa 1 — Análise exploratória e preparação dos dados

A primeira etapa teve como objetivo compreender o conjunto de dados e prepará-lo para a construção dos modelos de aprendizado de máquina.

Foram realizadas as seguintes atividades:

- Análise da estrutura e das dimensões do conjunto de dados.
- Identificação de valores ausentes e investigação de possíveis inconsistências.
- Análise estatística das variáveis numéricas.
- Investigação de distribuições, correlações e possíveis valores discrepantes.
- Análise da distribuição da variável-alvo de classificação.
- Comparação de características entre grupos de veículos.
- Codificação de variáveis categóricas por One-Hot Encoding.
- Padronização das variáveis numéricas.

### Principais observações da análise exploratória

Na tarefa de classificação, observou-se um desequilíbrio entre as classes: aproximadamente 90,04% dos registros correspondiam à ausência de falha e 9,96% à ocorrência de falha.

Também foram identificadas diferenças nas taxas de falha entre categorias de veículos, com aproximadamente 26,29% na categoria Fleet e 3,60% na categoria Private.

Na análise de correlação, algumas variáveis apresentaram associação relevante com os alvos. Para regressão, destacaram-se `cell_voltage_avg`, `thermal_runaway_risk` e `thermal_health_score`. Para classificação, destacaram-se `capacity_loss_percent`, `battery_health_percent`, `state_of_health` e `aging_score`.

As correlações foram utilizadas como instrumento de investigação exploratória, não sendo interpretadas como evidência de causalidade ou como substitutas da importância de atributos calculada pelos modelos.

## 5. Etapa 2 — Redes neurais artificiais com PyTorch

Na segunda etapa, foram implementadas duas redes neurais artificiais do tipo **Multilayer Perceptron (MLP)**, uma para classificação e outra para regressão.

A implementação foi realizada utilizando PyTorch, com camadas totalmente conectadas e possibilidade de configuração da arquitetura e dos principais hiperparâmetros de treinamento.

### 5.1. Pré-processamento

O pré-processamento foi implementado separadamente para cada tarefa.

A divisão dos dados preservou a separação treino/teste utilizada na atividade anterior, com `random_state=42` e proporção de 80% para treinamento e 20% para teste.

Na classificação, foi utilizada estratificação (`stratify=y`) para preservar a proporção entre as classes.

O conjunto de treinamento foi posteriormente dividido para formar os conjuntos de treinamento e validação. A proporção final foi:

| Conjunto | Proporção |
|---|---:|
| Treinamento | 60% |
| Validação | 20% |
| Teste | 20% |

O pré-processador e o escalonador do alvo de regressão foram ajustados somente com os dados de treinamento, evitando vazamento de informações dos conjuntos de validação e teste.

As variáveis categóricas foram transformadas por One-Hot Encoding, enquanto as numéricas passaram por padronização. Após o pré-processamento, foram obtidas **163 características de entrada** para as redes neurais.

### 5.2. Arquiteturas

Foram implementadas duas arquiteturas:

- **MLPClassifier:** rede neural para classificação binária, com saída utilizada para estimar a probabilidade de ocorrência de falha.
- **MLPRegressor:** rede neural para regressão, com saída contínua para estimar a vida útil restante da bateria.

Foram consideradas diferentes funções de ativação, incluindo ReLU, Leaky ReLU, Tanh, Sigmoid, ELU e GELU.

Os experimentos também avaliaram diferentes quantidades de neurônios e camadas ocultas, além de configurações de otimizadores e regularização.

### 5.3. Hiperparâmetros avaliados

Os principais hiperparâmetros investigados foram:

| Hiperparâmetro | Descrição |
|---|---|
| Neurônios | Quantidade de neurônios nas camadas ocultas |
| Camadas ocultas | Profundidade da rede neural |
| Função de ativação | Função aplicada às camadas ocultas |
| Otimizador | Algoritmo utilizado para atualizar os pesos |
| Learning rate | Taxa de aprendizado |
| Batch size | Quantidade de amostras por lote |
| Dropout | Regularização por desativação aleatória de neurônios |
| Weight decay | Regularização L2 aplicada aos pesos |

O otimizador Adam foi utilizado nas configurações finais selecionadas, com taxa de aprendizado de 0,001 e batch size de 256.

## 6. Experimentos e seleção de hiperparâmetros

Os experimentos foram organizados em três etapas.

### 6.1. Experimentos exploratórios

Foram avaliadas configurações de hiperparâmetros para as tarefas de classificação e regressão.

A etapa exploratória contemplou 17 configurações por tarefa, executadas com três sementes aleatórias:

- 42
- 123
- 2026

Foram realizados, portanto, 102 treinamentos exploratórios no total.

Os resultados foram armazenados em arquivos CSV, permitindo comparar o desempenho médio e o desvio-padrão das configurações avaliadas.

### 6.2. Experimento de interação

Foi realizado um experimento adicional para investigar a interação entre a quantidade de neurônios e o parâmetro dropout na tarefa de regressão.

Foram comparadas quatro combinações:

- 32 neurônios e dropout 0,0;
- 32 neurônios e dropout 0,3;
- 64 neurônios e dropout 0,0;
- 64 neurônios e dropout 0,3.

Cada combinação foi executada com três sementes aleatórias.

Os resultados indicaram que a aplicação de dropout 0,3 não trouxe benefícios para as configurações avaliadas. A configuração com 32 neurônios e dropout 0,0 apresentou o melhor desempenho médio entre as combinações testadas.

### 6.3. Busca final

Na busca final foram avaliadas três configurações de classificação e quatro de regressão.

A configuração selecionada para classificação foi a C3:

| Hiperparâmetro | Valor |
|---|---|
| Neurônios | 32 |
| Camadas ocultas | 2 |
| Ativação | GELU |
| Dropout | 0,0 |
| Weight decay | 0,001 |
| Otimizador | Adam |
| Learning rate | 0,001 |
| Batch size | 256 |
| Seed do checkpoint | 123 |

A configuração selecionada para regressão foi a R1:

| Hiperparâmetro | Valor |
|---|---|
| Neurônios | 32 |
| Camadas ocultas | 2 |
| Ativação | ReLU |
| Dropout | 0,0 |
| Weight decay | 0,0 |
| Otimizador | Adam |
| Learning rate | 0,001 |
| Batch size | 256 |
| Seed do checkpoint | 42 |

A seleção foi realizada considerando os resultados obtidos nas configurações avaliadas, priorizando F1-score na classificação e MAE na regressão.

## 7. Resultados da avaliação final

Após a seleção das configurações, os modelos foram avaliados no conjunto de teste, que permaneceu separado durante o treinamento e a seleção dos hiperparâmetros.

### 7.1. Classificação

A tarefa de classificação apresentou os seguintes resultados:

| Métrica | Resultado |
|---|---:|
| Acurácia | 96,575% |
| Precisão | 81,816% |
| Recall | 84,362% |
| F1-score | 83,070% |
| ROC-AUC | 98,918% |
| PR-AUC | 92,206% |

A matriz de confusão foi:

| | Predito: sem falha | Predito: com falha |
|---|---:|---:|
| Real: sem falha | 35.269 | 747 |
| Real: com falha | 623 | 3.361 |

O limiar de decisão utilizado foi aproximadamente 0,439272.

O resultado demonstra que o modelo conseguiu identificar uma parcela relevante dos registros com falha, apresentando recall de aproximadamente 84,36%. A precisão de aproximadamente 81,82% indica que ainda ocorreram falsos positivos, aspecto importante em uma aplicação de predição de falhas.

### 7.2. Regressão

A tarefa de regressão apresentou os seguintes resultados:

| Métrica | Resultado |
|---|---:|
| MAE | 45,1288 ciclos |
| RMSE | 95,1296 ciclos |
| R² | 0,997247 |

O MAE indica que, em média, as previsões apresentaram erro absoluto de aproximadamente 45 ciclos em relação aos valores reais.

O coeficiente R² de aproximadamente 0,9972 indica que o modelo explicou uma parcela elevada da variabilidade observada no conjunto de teste.

A métrica MAPE não foi utilizada, pois a variável-alvo contém valores iguais a zero, situação que torna seu cálculo indefinido ou inadequado para esses registros.

### 7.3. Considerações sobre os resultados

Os resultados indicam desempenho elevado nas duas tarefas. Entretanto, métricas altas não garantem, isoladamente, que os modelos terão o mesmo desempenho em dados de produção ou em conjuntos de dados com distribuições diferentes.

Também é importante considerar o desequilíbrio entre as classes na classificação e a possibilidade de diferenças entre os dados utilizados no treinamento e aqueles encontrados em aplicações reais.

## 8. Persistência dos modelos e inferência

Os modelos finais foram salvos em arquivos `.pt`, permitindo sua reutilização sem a necessidade de executar novamente o treinamento.

Os checkpoints selecionados são:

```text
models/classification/final_C3_seed_123.pt
models/regression/final_R1_seed_42.pt
```

Os arquivos armazenam informações como:

- Estado dos parâmetros da rede (`model_state_dict`);
- Estado do otimizador (`optimizer_state_dict`);
- Configuração da arquitetura;
- Configurações de treinamento;
- Semente utilizada;
- Época selecionada;
- Melhor perda de validação.

Os objetos utilizados no pré-processamento também foram persistidos:

```text
models/classification/preprocessing.joblib
models/regression/preprocessing.joblib
```

Esses arquivos permitem aplicar aos novos dados as mesmas transformações utilizadas durante o treinamento.

### 8.1. Executando a inferência

Foi implementado o script `src/inference.py`, responsável por carregar o modelo treinado, recuperar o pré-processador correspondente e realizar previsões sobre novos dados.

O arquivo de entrada deve conter as variáveis preditoras originais utilizadas pelo modelo, sem necessidade de incluir a variável-alvo.

Para classificação:

```bash
python src/inference.py \
    --task classification \
    --input data/novos_dados.csv
```

Para regressão:

```bash
python src/inference.py \
    --task regression \
    --input data/novos_dados.csv
```

O script também permite informar o caminho do arquivo de saída por meio do argumento `--output`.

Os resultados são exportados em CSV, preservando as colunas originais e acrescentando as previsões:

- **Classificação:** `probabilidade_falha` e `previsao_battery_failure`.
- **Regressão:** `previsao_remaining_life_cycles`.

A implementação foi testada com os dois checkpoints finais, incluindo o carregamento independente dos modelos e a geração de arquivos de previsão para novos registros.

## 9. Tecnologias utilizadas

- Python 3.12
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Matplotlib
- Seaborn
- Joblib
- Jupyter Notebook

O treinamento das redes neurais foi realizado em CPU, sem utilização de GPU.

### Ambiente de execução

| Componente | Configuração |
|---|---|
| Sistema operacional | Ubuntu |
| Python | 3.12.3 |
| PyTorch | 2.15.0+cpu |
| Scikit-learn | 1.9.1 |
| Pandas | 3.1.0 |
| NumPy | 2.5.4 |
| Processador | Intel Core i5-1135G7 |
| Processadores lógicos | 8 |
| Memória RAM disponível | Aproximadamente 7,5 GiB |

## 10. Instalação e execução

### 10.1. Clonar o repositório

```bash
git clone https://github.com/Isabellecavalcant/projeto-ia-ev-battery.git projeto-ia
cd projeto-ia
```

### 10.2. Criar e ativar o ambiente virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 10.3. Instalar as dependências

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Para a implementação com PyTorch em CPU:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Também é necessário ter o Joblib disponível:

```bash
python -m pip install joblib
```

### 10.4. Executar os experimentos

Os scripts principais podem ser executados diretamente a partir da raiz do projeto.

**Experimentos exploratórios:**

```bash
python src/pytorch/run_experiments.py
```

**Experimento de interação entre dropout e neurônios:**

```bash
python src/pytorch/run_interaction.py
```

**Geração dos gráficos:**

```bash
python src/pytorch/plot_experiments.py
```

**Busca final de hiperparâmetros:**

```bash
python src/pytorch/run_final_search.py
```

**Avaliação final no conjunto de teste:**

```bash
python src/pytorch/evaluate_final.py
```

A execução dos experimentos e da busca final pode levar vários minutos, pois envolve múltiplos treinamentos. Os comandos acima reproduzem os procedimentos e podem sobrescrever resultados existentes.

Os resultados da avaliação final são armazenados em:

```text
reports/final_test_metrics.json
```

## 11. Resultados e arquivos gerados

Os principais resultados produzidos durante o desenvolvimento estão organizados nos seguintes diretórios:

| Diretório | Conteúdo |
|---|---|
| `experiments/classification/` | Resultados e históricos dos experimentos de classificação |
| `experiments/regression/` | Resultados e históricos dos experimentos de regressão |
| `models/classification/` | Checkpoint e pré-processador do modelo de classificação |
| `models/regression/` | Checkpoint e pré-processador do modelo de regressão |
| `reports/figures/classification/` | Gráficos dos experimentos de classificação |
| `reports/figures/regression/` |