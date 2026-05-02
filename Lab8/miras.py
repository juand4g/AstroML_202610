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
LONG_PERIOD_FACTOR = 9  # T_largo debe ser > FACTOR × T_corto (separa ambas oscilaciones)
N_TERMS_LONG       = 2 # Términos de Fourier para el modelo de tendencia larga
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = "Lab8/data_selected"
N_FILES  = 5  # cuántos archivos más pesados usar

_all_csvs = [
    os.path.join(DATA_DIR, f)
    for f in os.listdir(DATA_DIR)
    if f.endswith(".csv")
]
files = sorted(_all_csvs, key=os.path.getsize, reverse=True)[:N_FILES]

for file in files:
        df = pd.read_csv(file)
        id = os.path.splitext(os.path.basename(file))[0].split("-")[-1]
        t, mag = df["t"].values, df["y"].values
        mag = mag - mag.mean()

        lpf = LONG_PERIOD_FACTOR
        ntl = N_TERMS_LONG

        # Período corto
        frequency, power = LombScargle(t, mag).autopower(method="fast")
        best_frequency = frequency[np.argmax(power)]
        best_period    = 1.0 / best_frequency

        # Período largo
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
        long_freq   = freq_long[np.argmax(power_long)]
        long_period = 1.0 / long_freq

        # Modelo de tendencia larga
        fs = FourierSeries(n_terms=ntl, base_freq=long_freq)
        fs.fit(t, mag)
        trend = fs(t)

        # Detrending
        mag_detrended = mag - trend + np.mean(trend)

        # Fases sin detrending
        phases0_og = get_phases(t, mag, best_period)
        phases_og  = np.append(phases0_og, phases0_og + 1)
        mag_og     = np.append(mag, mag)

        # Fases con detrending
        phases0_dt = get_phases(t, mag_detrended, best_period)
        phases_dt  = np.append(phases0_dt, phases0_dt + 1)
        mag_dt     = np.append(mag_detrended, mag_detrended)

        # ── Figura con 3 paneles ──────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.suptitle(f"Estrella: {id}  |  T_corto={best_period:.2f} d  |  T_largo≈{long_period:.1f} d  [lpf={lpf}, ntl={ntl}]")

        # Panel 1: serie de tiempo + modelo largo
        t_ord = np.argsort(t)
        axes[0].scatter(t, mag, c="black", s=1, label="Datos")
        axes[0].plot(t[t_ord], trend[t_ord], c="red", lw=1.5,
                     label=f"Modelo (T≈{long_period:.1f} d)")
        axes[0].invert_yaxis()
        axes[0].set_xlabel("t")
        axes[0].set_ylabel("mag")
        axes[0].set_title("Serie de tiempo")
        axes[0].legend(fontsize=7)

        # Panel 2: curva foldeada SIN detrending
        axes[1].scatter(phases_og, mag_og, c="black", s=2)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Fase")
        axes[1].set_ylabel("mag")
        axes[1].set_title("Foldeada (sin detrending)")

        # Panel 3: curva foldeada CON detrending
        axes[2].scatter(phases_dt, mag_dt, c="black", s=2)
        axes[2].invert_yaxis()
        axes[2].set_xlabel("Fase")
        axes[2].set_ylabel("mag")
        axes[2].set_title("Foldeada (con detrending)")

        plt.tight_layout()
        plt.show()
