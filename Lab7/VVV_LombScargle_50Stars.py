# # VVV — Análisis de Periodo con Lomb-Scargle
# ### 50 estrellas aleatorias · Curva de luz · Periodograma · Curva de fase

import numpy as np
import matplotlib.pyplot as plt
import random

import pandas as pd

from astropy.io import fits
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks

# ## 1 · Carga de datos

VVV_data = fits.open('Lab7/VVV_Sample.fits')

Id_b278,  Id_b279  = VVV_data[1].data,  VVV_data[11].data

Ks_b278,  Ks_b279  = VVV_data[2].data,  VVV_data[12].data
dKs_b278, dKs_b279 = VVV_data[5].data,  VVV_data[15].data
mjd_b278, mjd_b279 = VVV_data[8].data,  VVV_data[18].data

# ## 2 · Limpieza de NaNs

def remove_nans(array2d):
    """Elimina NaNs fila a fila."""
    return [row[~np.isnan(row)] for row in array2d]

Ks_b278  = remove_nans(Ks_b278)
dKs_b278 = remove_nans(dKs_b278)
mjd_b278 = remove_nans(mjd_b278)

Ks_b279  = remove_nans(Ks_b279)
dKs_b279 = remove_nans(dKs_b279)
mjd_b279 = remove_nans(mjd_b279)

# ## 3 · Parámetros clave de Lomb-Scargle
# Ajustar el rango y resolución de la búsqueda de periodos.

# ──────────────────────────────────────────────
#   PARÁMETROS DE LOMB-SCARGLE
# ──────────────────────────────────────────────

P_MIN   = 0.1      # Periodo mínimo de prueba (días)
P_MAX   = 10.0    # Periodo máximo de prueba (días)
N_FREQ  = 15000   # Número de frecuencias de prueba

N_STARS = 1      # Cantidad de estrellas aleatorias a analizar

# ──────────────────────────────────────────────
#   PARÁMETROS DE find_peaks (scipy)
# ──────────────────────────────────────────────

N_PEAKS         = 3   # Número de candidatos de periodo a mostrar
PEAK_DISTANCE   = 30   # Distancia mínima entre picos (en muestras de frecuencia)
PEAK_PROMINENCE = 0.05  # Prominencia mínima de cada pico

# ──────────────────────────────────────────────
#   Frecuencias derivadas de los periodos
# ──────────────────────────────────────────────
FREQ_MIN = 1.0 / P_MAX   # frecuencia mínima  (1/día)
FREQ_MAX = 1.0 / P_MIN   # frecuencia máxima  (1/día)

# Grilla de frecuencias equiespaciada
test_frequencies = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQ)

# ## 4 · Funciones auxiliares

def get_phases(t, mag, period):
    """
    Calcula las fases plegadas al periodo dado.
    El tiempo de referencia t0 se ancla al percentil 2% de la magnitud
    (punto más brillante aproximado).
    """
    threshold = np.percentile(mag, 2)
    t0 = t[np.argmin(np.abs(mag - threshold))]
    return ((t - t0) / period) % 1


def plot_star(star_id, t, mag, dmag, freq_grid,
              ax_lc, ax_pgram, axes_phase,
              n_peaks=1, peak_distance=None, peak_prominence=None):
    """
    Calcula el periodograma Lomb-Scargle y genera:
      · ax_lc         : curva de luz  (MJD vs magnitud)
      · ax_pgram      : periodograma con N candidatos marcados en distintos colores
      · axes_phase[i] : N curvas de fase, una por candidato
    """
    # ── Lomb-Scargle ────────────────────────────────────────────────────
    ls    = LombScargle(t, mag, dmag)
    power = ls.power(freq_grid)

    # ── Búsqueda de N picos con find_peaks ──────────────────────────────
    pk_kwargs = {}
    if peak_distance is not None:
        pk_kwargs['distance'] = peak_distance
    if peak_prominence is not None:
        pk_kwargs['prominence'] = peak_prominence

    peak_idx, _ = find_peaks(power, **pk_kwargs)

    # Fallback: si no hay picos con los criterios dados, usar el máximo global
    if len(peak_idx) == 0:
        peak_idx = np.array([np.argmax(power)])

    # Ordenar por potencia descendente y tomar los N mejores
    sorted_by_power = peak_idx[np.argsort(power[peak_idx])[::-1]]
    top_idx = sorted_by_power[:n_peaks]

    cand_freqs   = freq_grid[top_idx]
    cand_periods = 1.0 / cand_freqs

    best_freq   = cand_freqs[0]
    best_period = cand_periods[0]

    # Un color distinto por candidato (tab10)
    cmap   = plt.cm.get_cmap('tab10', len(top_idx))
    colors = [cmap(k) for k in range(len(top_idx))]

    # ── Curva de luz ─────────────────────────────────────────────────────
    ax_lc.scatter(t, mag, c='k', s=2)
    ax_lc.set_xlabel('MJD (días)')
    ax_lc.set_ylabel('Ks (mag)')
    ax_lc.set_title(f'ID {star_id}  —  Curva de luz', fontsize=9)
    ax_lc.invert_yaxis()

    # ── Periodograma ─────────────────────────────────────────────────────
    periods_grid = 1.0 / freq_grid
    sort_idx = np.argsort(periods_grid)
    ax_pgram.plot(periods_grid[sort_idx], power[sort_idx], c='steelblue', lw=0.7)

    for k, (period, color) in enumerate(zip(cand_periods, colors)):
        ax_pgram.axvline(period, color=color, linestyle='--', lw=1.2,
                         label=f'P{k+1} = {period:.4f} d')

    ax_pgram.set_xlabel('Periodo (días)')
    ax_pgram.set_ylabel('Potencia LS')
    ax_pgram.set_title(f'Periodograma  —  P₁ = {best_period:.4f} d', fontsize=9)
    ax_pgram.legend(fontsize=7, loc='upper right')

    # ── Curvas de fase (0 → 2) ────────────────────────────────────────────
    for k, (ax_ph, period, color) in enumerate(zip(axes_phase, cand_periods, colors)):
        phase  = get_phases(t, mag, period)
        phase2 = np.concatenate([phase, phase + 1])
        mag2   = np.concatenate([mag, mag])

        ax_ph.scatter(phase2, mag2, c='k', s=2)
        ax_ph.set_xlabel('Fase')
        ax_ph.set_ylabel('Ks (mag)')
        ax_ph.set_title(f'P{k+1} = {period:.5f} d', fontsize=9)
        ax_ph.set_xlim(0, 2)
        ax_ph.invert_yaxis()

        # Borde del mismo color que la línea en el periodograma
        for spine in ax_ph.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)

    return best_freq, best_period, cand_periods

# ## 5 · Selección aleatoria y análisis

# Excluir estrellas que ya están en el catálogo Excel
df_excel    = pd.read_excel('Lab7/periods_vvv_ls.xlsx')
catalog_ids = set(df_excel['id'].values)

# Selección aleatoria reproducible entre las que NO están en el catálogo
random.seed(28)
available_idx = [i for i, sid in enumerate(Id_b278) if sid not in catalog_ids]
selected_idx  = random.sample(available_idx, min(N_STARS, len(available_idx)))

print(f"IDs en el catálogo Excel       : {len(catalog_ids)}")
print(f"Estrellas disponibles en b278  : {len(available_idx)}")
print(f"Estrellas seleccionadas        : {len(selected_idx)}")
print(f"Rango de periodos de prueba    : [{P_MIN}, {P_MAX}] días")
print(f"Número de frecuencias          : {N_FREQ}")

# Tabla de resultados: [índice original, ID, mejor frecuencia, mejor periodo]
resultados = np.zeros((N_STARS, 4))

for i, idx in enumerate(selected_idx):

    t   = mjd_b278[idx]
    ks  = Ks_b278[idx]
    dks = dKs_b278[idx]
    sid = Id_b278[idx]

    # Saltar si la curva de luz tiene menos de 5 puntos válidos
    if len(t) < 5:
        print(f"[{i+1:02d}/{N_STARS}] ID {sid} — muy pocos puntos ({len(t)}), se omite.")
        continue

    # ── Figura con layout dinámico según N_PEAKS ─────────────────────────
    # Fila 0: Curva de luz (col 0) | Periodograma (cols 1..ncols-1)
    # Fila 1: N_PEAKS curvas de fase (una por columna)
    ncols = max(2, N_PEAKS)
    fig   = plt.figure(figsize=(5 * ncols, 7))
    fig.suptitle(f'Estrella {i+1}/{N_STARS}  —  ID {sid}', fontsize=11, fontweight='bold')
    gs = fig.add_gridspec(2, ncols, hspace=0.45, wspace=0.3)

    ax_lc    = fig.add_subplot(gs[0, 0])
    ax_pgram = fig.add_subplot(gs[0, 1:])

    if N_PEAKS == 1:
        axes_phase = [fig.add_subplot(gs[1, :])]
    else:
        axes_phase = [fig.add_subplot(gs[1, k]) for k in range(N_PEAKS)]

    best_freq, best_period, cand_periods = plot_star(
        star_id         = sid,
        t               = t,
        mag             = ks,
        dmag            = dks,
        freq_grid       = test_frequencies,
        ax_lc           = ax_lc,
        ax_pgram        = ax_pgram,
        axes_phase      = axes_phase,
        n_peaks         = N_PEAKS,
        peak_distance   = PEAK_DISTANCE,
        peak_prominence = PEAK_PROMINENCE,
    )

    resultados[i] = [idx, sid, best_freq, best_period]

    plt.tight_layout()
    plt.show()

    candidatos_str = "  |  ".join(f"P{k+1} = {p:.8f} d" for k, p in enumerate(cand_periods))
    print(f"[{i+1:02d}/{N_STARS}] ID {sid:>10.0f}  →  {candidatos_str}")
    plt.show()

# ## 6 · Resumen de resultados


df = pd.DataFrame(
    resultados,
    columns=['idx_original', 'star_id', 'best_frequency_1d', 'best_period_d']
)
df = df[df['best_period_d'] > 0]   # eliminar filas vacías (estrellas omitidas)
df = df.reset_index(drop=True)

print(df.to_string(index=False))
print(f"\nPeriodo medio  : {df['best_period_d'].mean():.4f} d")
print(f"Periodo mediano: {df['best_period_d'].median():.4f} d")
print(f"Rango          : [{df['best_period_d'].min():.4f}, {df['best_period_d'].max():.4f}] d")

# Histograma de periodos encontrados
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df['best_period_d'], bins=20, color='steelblue', edgecolor='white', linewidth=0.6)
ax.axvline(df['best_period_d'].median(), color='red', linestyle='--', lw=1.5,
           label=f"Mediana = {df['best_period_d'].median():.3f} d")
ax.set_xlabel('Periodo (días)')
ax.set_ylabel('N estrellas')
ax.set_title('Distribución de periodos — 50 estrellas aleatorias VVV')
ax.legend()
plt.tight_layout()
plt.show()

# ## 7 · Figuras para artículo — Curvas de fase del catálogo Excel

# Lookup rápido id → índice en b278
id_to_idx = {sid: i for i, sid in enumerate(Id_b278)}

# Recopilar entradas usando el periodo del catálogo Excel
art_entries = []
for _, row in df_excel.iterrows():
    sid_ex = row['id']
    stype  = str(row['type']) if pd.notna(row['type']) else '—'
    period = row['period']
    if sid_ex not in id_to_idx:
        continue
    idx = id_to_idx[sid_ex]
    t   = mjd_b278[idx]
    mag = Ks_b278[idx]
    if len(t) < 5:
        continue
    art_entries.append((sid_ex, period, stype, t, mag))

# Distribuir en dos figuras
n_total = len(art_entries)
n_fig1  = (n_total + 1) // 2
halves  = [art_entries[:n_fig1], art_entries[n_fig1:]]

NCOLS_ART = 4   # columnas por figura

for fig_num, entries in enumerate(halves, start=1):
    if not entries:
        continue
    nrows = (len(entries) + NCOLS_ART - 1) // NCOLS_ART
    fig, axes = plt.subplots(nrows, NCOLS_ART, figsize=(8.5, 11.0))
    axes = np.array(axes).reshape(-1)

    for ax, (sid_ex, period, stype, t, mag) in zip(axes, entries):
        phase  = get_phases(t, mag, period)
        phase2 = np.concatenate([phase, phase + 1])
        mag2   = np.concatenate([mag,   mag])
        ax.scatter(phase2, mag2, c='k', s=1, rasterized=True)
        ax.set_xlim(0, 2)
        ax.invert_yaxis()
        ax.set_xlabel('Fase', fontsize=6)
        ax.set_ylabel('$K_s$', fontsize=6)
        ax.tick_params(labelsize=5)
        ax.set_title(f'ID {int(sid_ex)}\nP={period:.5f} d  ·  {stype}', fontsize=5.5)

    for ax in axes[len(entries):]:
        ax.set_visible(False)

    fig.suptitle('Curvas de fase — Catálogo VVV', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'Lab7/fig_phase_curves_{fig_num}.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Figura {fig_num}/2 guardada  ({len(entries)} estrellas)  →  Lab7/fig_phase_curves_{fig_num}.png")
