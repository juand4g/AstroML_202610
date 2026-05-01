import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from least_entropy_antialiasing_fixed import get_phases
from scipy.signal import find_peaks


route = "Lab8/SEMANA.12/T.TAURI.TESS/TIC_264739351.dat"
df = pd.read_csv(route, names=["t","flux"], sep=" ")


plt.scatter(df["t"],df["flux"])
plt.xlabel("Tiempo")
plt.ylabel("Flujo")

t, flux = df["t"], df["flux"]

P_NUM=int(np.ptp(df["t"])*30)
P_FIN=5
P_INI=0.1
N=7

frequency = np.linspace(
            1 / P_FIN,
            1 / P_INI,
            P_NUM
        )

power = LombScargle(t, flux).power(
            frequency,
            method="fastchi2"
        )

power = power / power.max()
periods = 1 / frequency
best_frequency = frequency[np.argmax(power)]
best_period = 1 / best_frequency

# ── fase ─────────────────────────────────────────
phases1 = get_phases(t, flux, best_period)
phases2 = phases1 + 1
phases = np.append(phases1, phases2)
mag_plot = np.append(flux, flux.copy())

fig, axs = plt.subplots(1, 3, figsize=(15, 4))

axs[0].scatter(phases, mag_plot, c="k", s=1)
axs[0].invert_yaxis()
axs[0].set_title(
    f"P = {best_period:.8f} d"
)

axs[1].plot(periods, power, "k-")
axs[1].axvline(best_period, color="r", linestyle="--")
axs[1].set_title("Periodogram (Period)")

axs[2].plot(frequency, power, "k-")
axs[2].axvline(best_frequency, color="r", linestyle="--")
axs[2].set_title("Periodogram (Frequency)")

plt.tight_layout()
plt.show()

# find_peaks sobre el power (ya normalizado 0-1)
# 'distance' evita picos demasiado juntos; ajusta según resolución
peaks_idx, _ = find_peaks(power, distance=2, height=0.01)

# Ordenar por potencia y tomar los N mejores
top_idx = peaks_idx[np.argsort(power[peaks_idx])[::-1][:N]]
top_periods   = periods[top_idx]
top_freqs     = frequency[top_idx]
top_powers    = power[top_idx]

# El mejor sigue siendo el máximo global
best_frequency = frequency[np.argmax(power)]
best_period    = 1 / best_frequency

# ── fase ─────────────────────────────────────────────────────────────────
phases1 = get_phases(t, flux, best_period)
phases2 = phases1 + 1
phases  = np.append(phases1, phases2)
mag_plot = np.append(flux, flux.copy())

# ── figura ───────────────────────────────────────────────────────────────
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

# Panel 0 — curva de luz en fase
axs[0].scatter(phases, mag_plot, c="k", s=1)
axs[0].invert_yaxis()
axs[0].set_title(f"P = {best_period:.8f} d")

# Panel 1 — periodograma en período
axs[1].plot(periods, power, "k-", lw=0.8)
axs[1].axvline(best_period, color="r", linestyle="--", label=f"P₁ = {best_period:.4f} d")
for rank, (p, pw) in enumerate(zip(top_periods, top_powers), start=1):
    axs[1].axvline(p, color="steelblue", linestyle=":", alpha=0.7)
    axs[1].scatter(p, pw, zorder=5, s=40)
    axs[1].annotate(f"P{rank}\n{p:.3f}", xy=(p, pw),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=7, color="steelblue")
axs[1].set_xlabel("Period (d)")
axs[1].set_ylabel("Normalized power")
axs[1].set_title("Periodogram (Period)")
axs[1].legend(fontsize=8)

# Panel 2 — periodograma en frecuencia
axs[2].plot(frequency, power, "k-", lw=0.8)
axs[2].axvline(best_frequency, color="r", linestyle="--", label=f"f₁ = {best_frequency:.4f}")
for rank, (f, pw) in enumerate(zip(top_freqs, top_powers), start=1):
    axs[2].axvline(f, color="steelblue", linestyle=":", alpha=0.7)
    axs[2].scatter(f, pw, zorder=5, s=40)
    axs[2].annotate(f"f{rank}\n{f:.3f}", xy=(f, pw),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=7, color="steelblue")
axs[2].set_xlabel("Frequency (d⁻¹)")
axs[2].set_ylabel("Normalized power")
axs[2].set_title("Periodogram (Frequency)")
axs[2].legend(fontsize=8)

plt.tight_layout()
plt.show()

# ── Modelo sinusoidal por frecuencia (análogo a componente de Fourier) ────
def ls_component(t, flux, freq):
    """Ajusta y devuelve el modelo sinusoidal de LS para una frecuencia dada."""
    ls = LombScargle(t, flux)
    t_fit = np.linspace(t.min(), t.max(), 1000)
    y_fit = ls.model(t_fit, freq)          # modelo A·cos + B·sin en t_fit
    y_on_t = ls.model(t, freq)             # mismo modelo evaluado en t original
    return t_fit, y_fit, y_on_t

# ── Calcular componente para cada top frecuencia ──────────────────────────
components = []
flux_residual = flux.copy()

for rank, (f, p) in enumerate(zip(top_freqs, top_periods), start=1):
    t_fit, y_fit, y_on_t = ls_component(t, flux_residual, f)
    components.append({
        "rank": rank,
        "freq": f,
        "period": p,
        "t_fit": t_fit,
        "y_fit": y_fit,
        "y_on_t": y_on_t,
    })
    flux_residual = flux_residual - y_on_t   # sustraer para la siguiente (como Fourier iterativo)

# ── Figura: una fila por componente ──────────────────────────────────────
fig, axs = plt.subplots(len(components), 2, figsize=(14, 3 * len(components)))

for i, comp in enumerate(components):
    phase1 = (t * comp["freq"]) % 1.0
    phase2 = phase1 + 1
    phases_c  = np.append(phase1, phase2)
    flux_c    = np.append(comp["y_on_t"], comp["y_on_t"])

    # Panel izquierdo — serie temporal con modelo superpuesto
    axs[i, 0].scatter(t, flux, c="gray", s=1, alpha=0.4, label="flux original")
    axs[i, 0].plot(comp["t_fit"], comp["y_fit"], color="steelblue", lw=1.5,
                   label=f"componente {comp['rank']}")
    axs[i, 0].set_ylabel("Flux")
    axs[i, 0].set_title(f"Componente {comp['rank']} — P = {comp['period']:.5f} d  |  f = {comp['freq']:.4f} d⁻¹")
    axs[i, 0].legend(fontsize=8)

    # Panel derecho — curva de luz doblada en fase
    axs[i, 1].scatter(phases_c, flux_c, c="steelblue", s=2)
    axs[i, 1].set_xlabel("Fase")
    axs[i, 1].set_ylabel("Flux")
    axs[i, 1].set_title(f"Fase — componente {comp['rank']}")

# Etiqueta eje x solo en la última fila
axs[-1, 0].set_xlabel("Tiempo (d)")

plt.tight_layout()
plt.show()

# ── Residuo final ─────────────────────────────────────────────────────────
plt.figure(figsize=(12, 3))
plt.scatter(t, flux_residual, c="k", s=1, alpha=0.6)
plt.axhline(0, color="r", lw=0.8, linestyle="--")
plt.xlabel("Tiempo (d)")
plt.ylabel("Flux residual")
plt.title(f"Residuo tras sustraer {len(components)} componentes")
plt.tight_layout()
plt.show()



