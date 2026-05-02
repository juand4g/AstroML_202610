import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from least_entropy_antialiasing_fixed import get_phases
from scipy.signal import find_peaks

route = "Lab8/SEMANA.12/T.TAURI.TESS/TIC_264739351.dat"
df = pd.read_csv(route, names=["t","flux"], sep=" ")

# ── Parámetros ajustables ─────────────────────────────────────────────────────
PEAKS_DISTANCE = 2
PEAKS_HEIGHT   = 0.005
N              = 10
N_PRUEBA       = 3
P_FIN          = df["t"].max()
P_INI          = 0.1

plt.scatter(df["t"], df["flux"])
plt.xlabel("Tiempo")
plt.ylabel("Flujo")

t, flux = df["t"], df["flux"]

P_NUM = int(np.ptp(df["t"]) * 30)

frequency = np.linspace(1 / P_FIN, 1 / P_INI, P_NUM)
power = LombScargle(t, flux).power(frequency, method="fastchi2")
power = power / power.max()

peaks_idx, _ = find_peaks(power, distance=PEAKS_DISTANCE, height=PEAKS_HEIGHT)

top_idx    = peaks_idx[np.argsort(power[peaks_idx])[::-1][:N]]
top_freqs  = frequency[top_idx]
top_powers = power[top_idx]

best_frequency = frequency[np.argmax(power)]
best_period    = 1 / best_frequency

# ── fase ──────────────────────────────────────────────────────────────────────
phases1  = get_phases(t, flux, best_period)
phases2  = phases1 + 1
phases   = np.append(phases1, phases2)
mag_plot = np.append(flux, flux.copy())

# ── figura ────────────────────────────────────────────────────────────────────
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Panel 0 — curva de luz en fase
axs[0].scatter(phases, mag_plot, c="k", s=1)
axs[0].invert_yaxis()
axs[0].set_xlim(0, 2)
axs[0].set_xlabel("Fase")
axs[0].set_ylabel("Flux")
axs[0].set_title(f"P = {best_period:.8f} d")

# Panel 1 — periodograma en frecuencia
axs[1].plot(frequency, power, "k-", lw=0.8)
axs[1].axvline(best_frequency, color="r", linestyle="--", label=f"f₁ = {best_frequency:.4f}")
for rank, (f, pw) in enumerate(zip(top_freqs, top_powers), start=1):
    axs[1].axvline(f, color="steelblue", linestyle=":", alpha=0.7)
    axs[1].scatter(f, pw, zorder=5, s=40)
    axs[1].annotate(f"f{rank}\n{f:.3f}", xy=(f, pw),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=7, color="steelblue")
axs[1].set_xlim(0, 2)
axs[1].set_xlabel("Frequency (d⁻¹)")
axs[1].set_ylabel("Normalized power")
axs[1].set_title("Periodogram (Frequency)")
axs[1].legend(fontsize=8)

plt.tight_layout()
plt.show()

# ── N_PRUEBA frecuencias más bajas entre los picos ────────────────────────────
# Tomar los N_PRUEBA picos de menor frecuencia (mayor periodo)
prueba_idx   = peaks_idx[np.argsort(frequency[peaks_idx])[:N_PRUEBA]]
prueba_freqs = frequency[prueba_idx]
prueba_pows  = power[prueba_idx]

for rank, (f, pw) in enumerate(zip(prueba_freqs, prueba_pows), start=1):
    p = 1 / f
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"Prueba {rank} — f = {f:.4f} d⁻¹  |  P = {p:.4f} d  |  potencia = {pw:.3f}", fontsize=11)

    # Panel izquierdo — curva doblada en fase
    ph1 = get_phases(t, flux, p)
    ph2 = ph1 + 1
    ph  = np.append(ph1, ph2)
    fl  = np.append(flux, flux)
    axs[0].scatter(ph, fl, c="k", s=1)
    axs[0].invert_yaxis()
    axs[0].set_xlim(0, 2)
    axs[0].set_xlabel("Fase")
    axs[0].set_ylabel("Flux")
    axs[0].set_title("Curva de luz en fase")

    # Panel derecho — periodograma del residuo tras sustraer esta frecuencia
    ls       = LombScargle(t, flux)
    residual = flux - ls.model(t, f)
    pow_res  = LombScargle(t, residual).power(frequency, method="fastchi2")
    pow_res  = pow_res / pow_res.max()
    axs[1].plot(frequency, pow_res, "k-", lw=0.8)
    axs[1].axvline(f, color="r", linestyle="--", alpha=0.7, label=f"f sustraída = {f:.4f}")
    axs[1].set_xlim(0, 2)
    axs[1].set_xlabel("Frequency (d⁻¹)")
    axs[1].set_ylabel("Normalized power (residuo)")
    axs[1].set_title("Periodograma del residuo")
    axs[1].legend(fontsize=8)

    plt.tight_layout()
    plt.show()
