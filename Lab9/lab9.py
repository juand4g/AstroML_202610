import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# ------- PARÁMETROS ----------
FAP          = 5e-3   # compromiso: detecta el grupo débil ~0.0068 sin inundar el espectro
gap_abs      = 0.2  # d⁻¹ — separación mínima entre grupos distintos
min_peak_sep = 0.01   # d⁻¹ — separación mínima entre picos detectados (evita encadenamiento)

# -------------------------------

file = "Lab9/gamma cas_s_all.csv"

data = pd.read_csv(file)
plt.scatter(data["time"], data["mag"], s=2, c="k")
plt.gca().invert_yaxis()
plt.title("Curva de luz de Gamma Cas")
plt.show()

t, mag = data["time"], data["mag"]
f_min = 1.0 / (t.max() - t.min())

# Rayleigh resolution — minimum frequency separation the dataset can resolve
freq_resolution = f_min

# LS is correct here (not FFT): ground-based data has gaps so plain FFT
# would introduce aliasing. autopower picks the optimal frequency grid.
ls = LombScargle(t, mag)
freq_hr, power_hr = ls.autopower(
    minimum_frequency=f_min,
    maximum_frequency=9.0,
    samples_per_peak=10,
)
fap_level = float(ls.false_alarm_level(FAP))

plt.plot(freq_hr, power_hr, lw=0.7)
plt.axhline(fap_level, color="red", ls="--", lw=1.0, label=f"FAP {FAP*100:.1f}%")
plt.title("Periodograma Gamma Cas (alta resolución)")
plt.ylabel("Potencia LS")
plt.xlabel("Frecuencia (d⁻¹)")
plt.legend()
plt.grid()
plt.show()

best_frequency = freq_hr[np.argmax(power_hr)]
best_period    = 1.0 / best_frequency

# =============================================================
# GRUPOS DE FRECUENCIA  (contraste con Labadie-Bartz+2022)
#
# El artículo usa "iterative pre-whitening" (VARTOOLS): ajusta y resta
# sinusoides sucesivamente hasta FAP > 1e-2, produciendo una lista
# discreta (f_i, A_i); luego agrupa esas frecuencias con un algoritmo
# de clustering (Appendix A) que calcula centro ponderado y amplitud
# neta por grupo.
#
# Aquí se adopta un enfoque directo equivalente:
#   1. Máximos locales significativos en el LS de alta resolución
#   2. Agrupamiento por proximidad (brecha > gap_abs d⁻¹ → nuevo grupo)
#   3. Ajuste gaussiano por grupo → centro f_c y anchura σ
#
# Configuración típica del artículo para estrellas Be de tipo temprano
# (Fig. 6-7):  g1 en 0.5-3 d⁻¹,  g2 ≈ 2×g1,  f_g2/f_g1 ≈ 2.0
# =============================================================

df       = freq_hr[1] - freq_hr[0]
min_dist = max(1, int(min_peak_sep / df))      # al menos 0.05 d⁻¹ entre picos consecutivos
peaks_idx, _ = find_peaks(power_hr, height=fap_level, distance=min_dist)
peak_freqs   = freq_hr[peaks_idx]
peak_powers  = power_hr[peaks_idx]

groups = []
if len(peaks_idx):
    grp = [peaks_idx[0]]
    for i in range(1, len(peaks_idx)):
        if peak_freqs[i] - peak_freqs[i - 1] < gap_abs:
            grp.append(peaks_idx[i])
        else:
            groups.append(grp)
            grp = [peaks_idx[i]]
    groups.append(grp)

def _gauss(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

group_results = []
for grp in groups:
    f0 = freq_hr[grp[0]]  - 2 * freq_resolution
    f1 = freq_hr[grp[-1]] + 2 * freq_resolution
    mask = (freq_hr >= f0) & (freq_hr <= f1)
    fw, pw = freq_hr[mask], power_hr[mask]
    im = np.argmax(pw)
    try:
        popt, _ = curve_fit(
            _gauss, fw, pw,
            p0=[pw[im], fw[im], max(freq_resolution, (f1 - f0) / 4)],
            bounds=([0, f0, 0], [np.inf, f1, f1 - f0]),
            maxfev=10000,
        )
        group_results.append({"center": popt[1], "sigma": abs(popt[2]),
                               "amp": popt[0], "n": len(grp)})
    except RuntimeError:
        # Gaussian didn't converge — use amplitude-weighted centroid (paper's fallback)
        center = np.average(freq_hr[grp], weights=power_hr[grp])
        group_results.append({"center": center, "sigma": freq_resolution,
                               "amp": power_hr[grp].max(), "n": len(grp)})

# Visualización de grupos
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(freq_hr, power_hr, lw=0.6, c="steelblue", label="LS")
ax.axhline(fap_level, ls="--", c="red", lw=1.2, label=f"FAP {FAP*100:.1f}%")
colors = plt.cm.tab10(np.linspace(0, 0.8, max(len(group_results), 1)))
for i, gr in enumerate(group_results):
    ax.axvspan(gr["center"] - 3 * gr["sigma"],
               gr["center"] + 3 * gr["sigma"],
               alpha=0.15, color=colors[i],
               label=f"G{i+1}: {gr['center']:.3f} d⁻¹  ({gr['n']} picos)")
    ax.axvline(gr["center"], c=colors[i], lw=1.2, ls="--")
ax.set_xlabel("Frecuencia (d⁻¹)")
ax.set_ylabel("Potencia LS")
ax.set_title("Grupos de frecuencia — Gamma Cas")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()

print(f"\nResolución: {freq_resolution:.5f} d⁻¹  (T_baseline = {1/freq_resolution:.1f} d)")
print(f"Nivel FAP {FAP*100:.0f}%: {fap_level:.4f}")
print(f"Grupos detectados: {len(group_results)}")
for i, gr in enumerate(group_results):
    print(f"  G{i+1}: f_c = {gr['center']:.4f} d⁻¹ | σ = {gr['sigma']:.4f} d⁻¹ | {gr['n']} picos")

if len(group_results) >= 2:
    by_amp = sorted(group_results, key=lambda g: g["amp"], reverse=True)
    g1, g2 = by_amp[0], by_amp[1]
    ratio  = g2["center"] / g1["center"]
    print(f"\n  g1 (más fuerte): {g1['center']:.3f} d⁻¹")
    print(f"  g2 (segundo):    {g2['center']:.3f} d⁻¹")
    print(f"  f_g2/f_g1 = {ratio:.2f}  (artículo espera ≈2.0 para B temprano)")

