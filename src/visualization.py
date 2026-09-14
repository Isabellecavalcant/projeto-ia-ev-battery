import matplotlib.pyplot as plt
import seaborn as sns


def grafico_valores_ausentes(df):
    """Gera gráfico do percentual de valores ausentes por variável."""
    percentual = df.isnull().mean().mul(100)
    percentual = percentual[percentual > 0].sort_values(ascending=False)

    plt.figure(figsize=(14, 6))
    percentual.plot(kind="bar")
    plt.ylabel("Valores ausentes (%)")
    plt.xlabel("Variáveis")
    plt.title("Percentual de valores ausentes por variável")
    plt.xticks(rotation=90)
    plt.tight_layout()
    return plt.gcf()


def grafico_correlacao(df):
    """Gera mapa de calor da correlação entre variáveis numéricas."""
    correlacao = df.select_dtypes(include="number").corr()

    plt.figure(figsize=(16, 12))
    sns.heatmap(correlacao, cmap="coolwarm", center=0)
    plt.title("Matriz de correlação")
    plt.tight_layout()
    return plt.gcf()


def boxplot(df, colunas):
    """Gera boxplots para um conjunto de variáveis numéricas."""
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df[colunas])
    plt.xticks(rotation=45)
    plt.title("Distribuição e identificação de outliers")
    plt.tight_layout()
    return plt.gcf()