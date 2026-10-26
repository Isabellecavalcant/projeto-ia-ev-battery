"""Sistema especialista baseado em regras SE-ENTÃO.

A BASE DE CONHECIMENTO fica em arquivo de texto (rules/*.txt)
e o MOTOR DE INFERÊNCIA fica neste módulo.

Formato das regras:
    PADRÃO: battery_failure = 0
    R01 | heurística | SE capacity_loss_percent > 30 E cell_voltage_std > 0.03 | ENTÃO battery_failure = 1

Tipos aceitos:
    relação, recomendação, diretriz, estratégia, heurística.

Operadores:
    >, >=, <, <=, ==, !=

Regras podem usar E ou OU, mas não ambos na mesma regra.
Linhas iniciadas por # são comentários.
O valor padrão é usado quando nenhuma regra conclui o alvo.
"""

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

import numpy as np


TIPOS = {
    "relacao",
    "recomendacao",
    "diretriz",
    "estrategia",
    "heuristica",
}

# Variável, operador e valor da condição.
_COND = re.compile(
    r"^\s*(\w+)\s*(>=|<=|==|!=|>|<)\s*(.*?)\s*$"
)

# Conclusão no formato: ENTÃO variavel = valor
_CONC = re.compile(
    r"^\s*ENT[AÃ]O\s+(\w+)\s*=\s*(.*?)\s*$",
    re.IGNORECASE,
)

# Valor padrão no formato: PADRÃO: variavel = valor
_PADRAO = re.compile(
    r"^\s*PADR[AÃ]O\s*:\s*(\w+)\s*=\s*(.*?)\s*$",
    re.IGNORECASE,
)


def _norm(txt):
    """Remove acentos e converte o texto para minúsculas."""
    return "".join(
        c
        for c in unicodedata.normalize("NFD", str(txt).lower())
        if unicodedata.category(c) != "Mn"
    )


def _valor(txt):
    """Converte valores numéricos para float e mantém textos como strings."""
    txt = txt.strip().strip("'\"")

    try:
        return float(txt)
    except ValueError:
        return txt


def _ausente(valor):
    """Verifica se um valor está ausente."""
    if valor is None:
        return True

    try:
        return bool(np.isnan(valor))
    except (TypeError, ValueError):
        return False


_OPS = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


@dataclass
class Regra:
    """Representa uma regra da base de conhecimento."""

    id: str
    tipo: str
    conectivo: str
    condicoes: list
    conclusao: tuple
    texto: str


class BaseConhecimento:
    """Lê e valida o arquivo de regras."""

    def __init__(self, caminho):
        self.caminho = str(caminho)
        self.regras = []
        self.padrao = None

        # Abre o arquivo e percorre suas linhas.
        with open(caminho, "r", encoding="utf-8") as arquivo:
            for n, linha in enumerate(arquivo, 1):
                linha = linha.strip()

                # Ignora linhas vazias e comentários.
                if not linha or linha.startswith("#"):
                    continue

                # Lê a declaração do valor padrão.
                if linha.upper().startswith("PADRÃO:") or (
                    _norm(linha).startswith("padrao:")
                ):
                    mp = _PADRAO.match(linha)

                    if not mp:
                        raise ValueError(
                            f"linha {n}: use "
                            "'PADRÃO: variavel = valor'"
                        )

                    self.padrao = (
                        mp.group(1),
                        _valor(mp.group(2)),
                    )
                    continue

                # Todas as demais linhas devem ser regras.
                self.regras.append(self._parse(linha, n))

        if self.padrao is None:
            raise ValueError(
                f"O arquivo '{caminho}' precisa declarar "
                "'PADRÃO: variavel = valor'."
            )

        if not self.regras:
            raise ValueError(
                f"O arquivo '{caminho}' não contém nenhuma regra."
            )

    @staticmethod
    def _parse(linha, n):
        """Converte uma linha de texto em um objeto Regra."""
        partes = [p.strip() for p in linha.split("|")]

        if len(partes) != 4:
            raise ValueError(
                f"linha {n}: use "
                "'ID | tipo | SE ... | ENTÃO ...'"
            )

        rid, tipo, se, entao = partes

        if _norm(tipo) not in TIPOS:
            raise ValueError(
                f"linha {n}: tipo '{tipo}' inválido. "
                f"Tipos aceitos: {sorted(TIPOS)}"
            )

        if not re.match(r"^\s*SE\b", se, re.IGNORECASE):
            raise ValueError(
                f"linha {n}: a premissa deve começar com 'SE'."
            )

        # Remove a palavra SE do começo da premissa.
        corpo = re.sub(
            r"^\s*SE\b\s*",
            "",
            se,
            flags=re.IGNORECASE,
        )

        tem_e = (
            re.search(r"\s+E\s+", corpo, re.IGNORECASE)
            is not None
        )
        tem_ou = (
            re.search(r"\s+OU\s+", corpo, re.IGNORECASE)
            is not None
        )

        if tem_e and tem_ou:
            raise ValueError(
                f"linha {n}: não misture E e OU na mesma regra."
            )

        conectivo = "E" if tem_e else ("OU" if tem_ou else "")

        if conectivo:
            separador = (
                r"\s+E\s+"
                if conectivo == "E"
                else r"\s+OU\s+"
            )
            pedacos = re.split(
                separador,
                corpo,
                flags=re.IGNORECASE,
            )
        else:
            pedacos = [corpo]

        condicoes = []

        for pedaco in pedacos:
            m = _COND.match(pedaco)

            if not m:
                raise ValueError(
                    f"linha {n}: condição inválida '{pedaco}'."
                )

            variavel = m.group(1)
            operador = m.group(2)
            valor = _valor(m.group(3))

            condicoes.append((variavel, operador, valor))

        mc = _CONC.match(entao)

        if not mc:
            raise ValueError(
                f"linha {n}: conclusão deve seguir o formato "
                "'ENTÃO variavel = valor'."
            )

        conclusao = (
            mc.group(1),
            _valor(mc.group(2)),
        )

        return Regra(
            id=rid,
            tipo=_norm(tipo),
            conectivo=conectivo,
            condicoes=condicoes,
            conclusao=conclusao,
            texto=linha,
        )


class MotorInferencia:
    """Executa o encadeamento para frente sobre a base de conhecimento."""

    def __init__(
        self,
        base: BaseConhecimento,
        alvo: str,
        max_passes: int = 10,
    ):
        self.base = base
        self.alvo = alvo
        self.max_passes = max_passes

        if base.padrao is None or base.padrao[0] != alvo:
            raise ValueError(
                f"Defina 'PADRÃO: {alvo} = valor' "
                "no arquivo de regras."
            )

        self.padrao = base.padrao[1]

        self.variaveis = sorted(
            {
                variavel
                for regra in base.regras
                for variavel, _, _ in regra.condicoes
            }
        )

        self.disparos = Counter()

    @staticmethod
    def _cond(fatos, var, op, val):
        """Avalia uma condição. Valores ausentes tornam a condição falsa."""
        a = fatos.get(var)

        if _ausente(a):
            return False

        try:
            return _OPS[op](a, val)
        except (TypeError, ValueError):
            # Exemplo: comparar texto com número.
            return False

    def _vale(self, regra, fatos):
        """Verifica se todas as condições E, ou alguma condição OU, são verdadeiras."""
        resultados = [
            self._cond(fatos, variavel, operador, valor)
            for variavel, operador, valor in regra.condicoes
        ]

        if regra.conectivo == "OU":
            return any(resultados)

        return all(resultados)

    def inferir(self, fatos: dict):
        """Aplica regras até não surgirem novas conclusões.

        Retorna:
            fatos_finais: fatos originais e novos fatos inferidos.
            usados: identificadores das regras disparadas.
        """
        fatos = dict(fatos)
        usados = []

        for _ in range(self.max_passes):
            mudou = False

            for regra in self.base.regras:
                variavel, valor = regra.conclusao

                # Um fato já existente não é sobrescrito.
                # Assim, a primeira conclusão válida prevalece.
                if variavel in fatos:
                    continue

                if self._vale(regra, fatos):
                    fatos[variavel] = valor
                    usados.append(regra.id)
                    mudou = True

            if not mudou:
                break

        return fatos, usados

    def _linhas(self, X):
        """Seleciona as variáveis utilizadas nas condições das regras."""
        colunas = [c for c in self.variaveis if c in X.columns]
        return X[colunas].to_dict("records")

    def predict(self, X):
        """Gera uma predição por linha do DataFrame."""
        saida = []

        for linha in self._linhas(X):
            fatos, usados = self.inferir(linha)
            self.disparos.update(usados)
            saida.append(fatos.get(self.alvo, self.padrao))

        return np.asarray(saida)

    def predict_score(self, X, classe_positiva=1):
        """Calcula uma pontuação heurística entre 0 e 1 para classificação.

        A pontuação é a fração das regras que concluem a classe positiva
        e cujas condições são satisfeitas pela amostra. É uma aproximação
        heurística, não uma probabilidade calibrada.
        """
        positivas = [
            regra
            for regra in self.base.regras
            if regra.conclusao[0] == self.alvo
            and regra.conclusao[1] == classe_positiva
        ]

        if not positivas:
            return np.zeros(len(X))

        saida = []

        for linha in self._linhas(X):
            fatos, _ = self.inferir(linha)

            # Reavalia as condições com os dados de entrada originais.
            fatos_avaliacao = dict(linha)
            fatos_avaliacao.update(
                {
                    k: v
                    for k, v in fatos.items()
                    if k != self.alvo
                }
            )

            pontuacao = sum(
                self._vale(regra, fatos_avaliacao)
                for regra in positivas
            ) / len(positivas)

            saida.append(pontuacao)

        return np.asarray(saida)

    def cobertura(self):
        """Retorna um DataFrame com a quantidade de disparos por regra."""
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "regra": regra.id,
                    "tipo": regra.tipo,
                    "disparos": self.disparos.get(regra.id, 0),
                }
                for regra in self.base.regras
            ]
        )