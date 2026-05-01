# In[ ]:
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
import random
import pandas as pd
import os
from datetime import datetime
from least_entropy_antialiasing_fixed import get_phases
from scipy.signal import find_peaks
from GeneralFourierSeries import FourierSeries

# ─── Parámetros ───────────────────────────────────────────────────────────────
GRAFICAR_OG          = False  # Mostrar curvas foldeadas SIN detrending
GRAFICAR_DT          = True   # Mostrar curvas foldeadas CON detrending
GRAFICAR_DIAGNOSTICO = True   # Mostrar curva original + modelo de tendencia larga

LONG_PERIOD_FACTOR = 8.5   # T_largo debe ser > FACTOR × T_corto (separa ambas oscilaciones)
N_TERMS_LONG       = 6   # Términos de Fourier para el modelo de tendencia larga
# ─────────────────────────────────────────────────────────────────────────────

files = [
    "Lab8/data/mira/OGLE-BLG-LPV-060312.dat",
    "Lab8/data/mira/OGLE-BLG-LPV-086116.dat",
    "Lab8/data/mira/OGLE-BLG-LPV-168070.dat",
    "Lab8/data/mira/OGLE-BLG-LPV-192393.dat",
]

# Parámetros de detrending por archivo (None → usa el valor global de arriba)
FILE_PARAMS = [
    {"LONG_PERIOD_FACTOR": None, "N_TERMS_LONG": None},  # 060312
    {"LONG_PERIOD_FACTOR": 9, "N_TERMS_LONG": 1},  # 086116
    {"LONG_PERIOD_FACTOR": 6, "N_TERMS_LONG": 3},  # 168070
    {"LONG_PERIOD_FACTOR": 6, "N_TERMS_LONG": 4},  # 192393
]

for file in files:
        df = pd.read_csv(file, sep=" ", names=["hjd","I","inc_I"])
        id = file.split("-")[3]
        t, mag = df["hjd"], df["I"]
        frequency, power = LombScargle(t, mag).autopower(
                method="fast")
        periods = 1 / frequency
        # Normalizar
        power = power / power.max()

        best_frequency = frequency[np.argmax(power)]
        best_period    = 1 / best_frequency
        phases0 = get_phases(t,mag,best_period)
        phases1 = phases0 + 1
        phases   = np.append(phases0, phases1)
        mag_plot = np.append(mag, mag.copy())

        if GRAFICAR_OG:
                plt.scatter(phases, mag_plot, c="black", s=2)
                plt.tight_layout()
                plt.title(f"Estrella: {id}, período: {best_period}:.4f")
                plt.show()


# ─── Detrending con Fourier ───────────────────────────────────────────────────
for file, fparams in zip(files, FILE_PARAMS):
        df = pd.read_csv(file, sep=" ", names=["hjd","I","inc_I"])
        id = file.split("-")[3]
        t, mag = df["hjd"].values, df["I"].values
        mag = mag - mag.mean()

        # Parámetros efectivos: usa el valor por archivo si está definido, si no el global
        lpf = fparams["LONG_PERIOD_FACTOR"] if fparams["LONG_PERIOD_FACTOR"] is not None else LONG_PERIOD_FACTOR
        ntl = fparams["N_TERMS_LONG"]       if fparams["N_TERMS_LONG"]       is not None else N_TERMS_LONG

        # Período corto (igual que arriba)
        frequency, power = LombScargle(t, mag).autopower(method="fast")
        best_frequency = frequency[np.argmax(power)]
        best_period = 1.0 / best_frequency

        # Período largo: buscar solo en frecuencias < best_frequency / lpf
        min_freq_long = 1.0 / (t.max() - t.min())
        max_freq_long = best_frequency / lpf
        if max_freq_long <= min_freq_long:
            span = t.max() - t.min()
            print(
                f"[{id}] ERROR: lpf={lpf} deja el rango de búsqueda vacío.\n"
                f"  Duración total : {span:.0f} d  |  T_corto = {best_period:.1f} d\n"
                f"  lpf máximo usable: {span / best_period:.1f}  (= T_span / T_corto)"
            )
            continue
        freq_long, power_long = LombScargle(t, mag).autopower(
                method="fast",
                minimum_frequency=min_freq_long,
                maximum_frequency=max_freq_long,
        )
        long_freq = freq_long[np.argmax(power_long)]
        long_period = 1.0 / long_freq

        # Modelar la oscilación larga con FourierSeries
        fs = FourierSeries(n_terms=ntl, base_freq=long_freq)
        fs.fit(t, mag)
        trend = fs(t)

        # Gráfica diagnóstico: curva original + modelo de tendencia larga
        if GRAFICAR_DIAGNOSTICO:
                t_ord = np.argsort(t)
                plt.figure(figsize=(10, 4))
                plt.scatter(t, mag, c="black", s=1, label="Datos originales")
                plt.plot(
                        t[t_ord], trend[t_ord],
                        c="red", lw=1.5,
                        label=f"Modelo Fourier (T≈{long_period:.1f} d, {ntl} términos)",
                )
                plt.xlabel("HJD")
                plt.ylabel("I")
                plt.gca().invert_yaxis()
                plt.legend()
                plt.title(f"Estrella: {id} — evaluación del detrending  [lpf={lpf}, ntl={ntl}]")
                plt.tight_layout()
                plt.show()

        # Detrending: restar la tendencia larga conservando la magnitud media
        mag_detrended = mag - trend + np.mean(trend)

        # Foldear los datos detrended con el período corto
        phases0 = get_phases(t, mag_detrended, best_period)
        phases1 = phases0 + 1
        phases   = np.append(phases0, phases1)
        mag_plot = np.append(mag_detrended, mag_detrended.copy())

        if GRAFICAR_DT:
                plt.scatter(phases, mag_plot, c="black", s=2)
                plt.tight_layout()
                plt.title(
                        f"Estrella: {id} [detrended]\n"
                        f"T_corto={best_period:.2f} d  |  T_largo≈{long_period:.1f} d  [lpf={lpf}, ntl={ntl}]"
                )
                plt.show()
