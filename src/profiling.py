import os
import threading
import time
import tracemalloc
from contextlib import contextmanager

import joblib
import numpy as np
import psutil

FATOR_CO2_BR_G_KWH = 45.8 
POTENCIA_CPU_W = 15.0

class _AmostradorRSS:

    def __init__(self, intervalo=0.01):
        self.proc = psutil.Process(os.getpid())
        self.intervalo = intervalo
        self.pico = self.proc.memory_info().rss
        self.base = self.pico
        self._parar = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._parar.is_set():
            self.pico = max(self.pico, self.proc.memory_info().rss)
            time.sleep(self.intervalo)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._parar.set()
        self._t.join()
        self.pico = max(self.pico, self.proc.memory_info().rss)


def estimar_energia_kwh(tempo_s: float, potencia_w: float = POTENCIA_CPU_W) -> float:
    return potencia_w * tempo_s / 3_600_000.0


def emissao_co2_g(energia_kwh: float, fator_g_kwh: float = FATOR_CO2_BR_G_KWH) -> float:
    return energia_kwh * fator_g_kwh


@contextmanager
def rastrear_codecarbon(ativo: bool = True):
    res = {"energia_kwh": None, "co2_kg": None}
    tracker = None
    if ativo:
        try:
            from codecarbon import EmissionsTracker
            tracker = EmissionsTracker(save_to_file=False, log_level="error")
            tracker.start()
        except Exception:
            tracker = None
    try:
        yield res
    finally:
        if tracker is not None:
            res["co2_kg"] = tracker.stop()
            data = getattr(tracker, "final_emissions_data", None)
            res["energia_kwh"] = getattr(data, "energy_consumed", None)


def medir_treinamento(funcao_treino, potencia_w: float = POTENCIA_CPU_W):
    with _AmostradorRSS() as amostra:
        inicio = time.perf_counter()
        resultado = funcao_treino()
        tempo = time.perf_counter() - inicio
    energia = estimar_energia_kwh(tempo, potencia_w)
    return resultado, {
        "treino_tempo_s": tempo,
        "treino_pico_ram_mb": (amostra.pico - amostra.base) / 1024**2,
        "treino_energia_kwh": energia,
        "treino_co2_g": emissao_co2_g(energia),
    }

def medir_inferencia(funcao_predicao, X, n_rep: int = 30,
                     potencia_w: float = POTENCIA_CPU_W):
    funcao_predicao(X) 

    tempos = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        funcao_predicao(X)
        tempos.append(time.perf_counter() - t0)

    tracemalloc.start()         
    funcao_predicao(X)
    _, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    media, desvio = float(np.mean(tempos)), float(np.std(tempos))
    n = len(X)
    energia = estimar_energia_kwh(media, potencia_w)
    return {
        "inf_tempo_medio_s": media,
        "inf_tempo_desvio_s": desvio,
        "inf_tempo_por_amostra_ms": 1000 * media / n,
        "inf_vazao_amostras_s": n / media,
        "inf_pico_memoria_kb": pico / 1024,
        "inf_energia_kwh": energia,
        "inf_co2_g": emissao_co2_g(energia),
    }

def tamanho_em_disco_kb(objeto, caminho: str = "/tmp/_modelo.joblib") -> float:
    if isinstance(objeto, str) and os.path.exists(objeto):
        return os.path.getsize(objeto) / 1024
    joblib.dump(objeto, caminho, compress=0)
    tam = os.path.getsize(caminho) / 1024
    os.remove(caminho)
    return tam