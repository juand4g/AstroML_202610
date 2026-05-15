import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import kpss
from tqdm import tqdm
import webbrowser
import plotly.express as px

# =============================================================================
# CONFIGURACIÓN DE GRÁFICAS
# =============================================================================
SHOW = {
    "conteo_por_clase":   False,
    "skewness":           False,
    "curtosis":           False,
    "kpss":               False,
    "amplitud":           False,
    "deriv_sign":         False,
    "periodos":           False,
    "ln_periodo":         False,
    "colour":             True,
    "scatter_3d":         False,
}
# =============================================================================

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "list.dat")
stars_data = pd.read_fwf(
    data_path, header=None,
    names=["id", "ra", "dec", "class", "class2", "I", "V", "P", "I_A", "P2"]
)

# --- Filtrado y registro de eliminadas ---
removed_parts = []

mask_class = stars_data["class"].isin(["T2CEP", "ACER", "ACEP"])
removed_parts.append(
    stars_data[mask_class][["id", "class"]].assign(reason="clase_excluida")
)
stars_data = stars_data[~mask_class].copy()

stars_data["P"] = pd.to_numeric(stars_data["P"], errors='coerce')
mask_no_period = stars_data["P"].isna()
removed_parts.append(
    stars_data[mask_no_period][["id", "class"]].assign(reason="sin_periodo")
)
stars_data = stars_data[~mask_no_period].copy()

removed = pd.concat(removed_parts, ignore_index=True)
removed.to_csv(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "removed_stars.csv"),
    index=False
)

stars_data["ln_P"]   = np.log(stars_data["P"])
stars_data["colour"] = pd.to_numeric(stars_data["V"], errors='coerce') \
                     - pd.to_numeric(stars_data["I"], errors='coerce')

# --- Reclasificación ---
stars_data["class"] = stars_data["class"].replace({
    "RRLYR": "PULSATING", "DSCT": "PULSATING", "DCEP": "PULSATING",
    "ECL":   "BINARY",    "ELL":  "BINARY",
})

# --- Resumen ---
print(f"Total de estrellas en la muestra: {len(stars_data)}")
print(f"Estrellas eliminadas: {len(removed)}  "
      f"(clases excluidas: {(removed['reason']=='clase_excluida').sum()}, "
      f"sin período: {(removed['reason']=='sin_periodo').sum()})")

# --- Histograma de conteo por clase ---
if SHOW["conteo_por_clase"]:
    class_counts = stars_data["class"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(class_counts.index, class_counts.values, color='steelblue', edgecolor='black')
    ax.set_xlabel("Clase")
    ax.set_ylabel("Número de estrellas")
    ax.set_title("Distribución de estrellas por clase")
    for bar, count in zip(bars, class_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(count), ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "histogram_by_class.png"))
    plt.show()

colors = {"PULSATING": "steelblue", "BINARY": "tomato",
          "LPV": "mediumseagreen", "OTHER": "goldenrod"}

# --- Histograma de magnitud I por clase ---
# Descartado como feature: la magnitud depende de factores externos (distancia, extinción, etc.)
# stars_data["I"] = pd.to_numeric(stars_data["I"], errors='coerce')
#
# fig, ax = plt.subplots(figsize=(8, 5))
# for cls in sorted(stars_data["class"].unique()):
#     mag = stars_data[stars_data["class"] == cls]["I"].dropna()
#     ax.hist(mag, bins=30, color=colors.get(cls, "gray"), alpha=0.5, label=cls)
# ax.set_xlabel("Magnitud I")
# ax.set_ylabel("N")
# ax.set_title("Distribución de magnitud I por clase")
# ax.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "histogram_mag_by_class.png"))
# plt.show()

# --- Features de la serie de tiempo por estrella ---
i_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "I")
CACHE_COLS = ["skew_lc", "kurt_lc", "kpss_lc", "amp_lc", "deriv_sign_lc"]

def lc_features(star_id):
    fpath = os.path.join(i_folder, f"{star_id}.dat")
    nan_row = pd.Series([np.nan] * len(CACHE_COLS), index=CACHE_COLS)
    try:
        lc = pd.read_csv(fpath, sep=r'\s+', header=None, names=["hjd", "I", "eI"])
        mag = pd.to_numeric(lc["I"], errors='coerce').dropna().values
        if len(mag) < 10:
            return nan_row
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat = kpss(mag, regression='c', nlags='auto')[0]
        amplitude = mag.max() - mag.min()
        diffs = np.sign(np.diff(mag))
        diffs = diffs[diffs != 0]   # ignorar plateaus
        sign_changes = np.sum(np.diff(diffs) != 0) / len(mag)
        return pd.Series([skew(mag), kurtosis(mag), kpss_stat, amplitude, sign_changes],
                         index=CACHE_COLS)
    except FileNotFoundError:
        return nan_row

cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lc_features_cache.csv")

if os.path.exists(cache_path):
    cache_df = pd.read_csv(cache_path)
    if all(c in cache_df.columns for c in CACHE_COLS):
        cache_df = cache_df.set_index("id")
        for col in CACHE_COLS:
            stars_data[col] = stars_data["id"].map(cache_df[col])
        print(f"\nFeatures leídas desde caché ({cache_path})")
    else:
        os.remove(cache_path)
        print("\nCaché incompleto, recalculando...")
else:
    print()

if not os.path.exists(cache_path):
    tqdm.pandas(desc="Calculando features de curvas de luz")
    feats = stars_data["id"].progress_apply(lc_features)
    stars_data[CACHE_COLS] = feats
    stars_data[["id"] + CACHE_COLS].to_csv(cache_path, index=False)
    print(f"Features guardadas en caché ({cache_path})")

n_found = stars_data["skew_lc"].notna().sum()
print(f"Curvas de luz encontradas: {n_found} / {len(stars_data)}")

if SHOW["skewness"]:
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in sorted(stars_data["class"].unique()):
        s = stars_data[stars_data["class"] == cls]["skew_lc"].dropna()
        ax.hist(s, bins=30, color=colors.get(cls, "gray"), alpha=0.5, label=cls)
    ax.set_xlabel("Skewness (serie de tiempo I)")
    ax.set_ylabel("N")
    ax.set_title("Skewness de curvas de luz por clase")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "histogram_skew_lc_by_class.png"))
    plt.show()

if SHOW["curtosis"]:
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in sorted(stars_data["class"].unique()):
        k = stars_data[stars_data["class"] == cls]["kurt_lc"].dropna()
        ax.hist(k, bins=30, color=colors.get(cls, "gray"), alpha=0.5, label=cls)
    ax.set_xlabel("Curtosis (serie de tiempo I)")
    ax.set_ylabel("N")
    ax.set_title("Curtosis de curvas de luz por clase")
    ax.legend()
    ax.set_xlim(-10, 25)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "histogram_kurt_lc_by_class.png"))
    plt.show()

if SHOW["kpss"]:
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in sorted(stars_data["class"].unique()):
        k = stars_data[stars_data["class"] == cls]["kpss_lc"].dropna()
        ax.hist(k, bins=30, color=colors.get(cls, "gray"), alpha=0.5, label=cls)
    ax.set_xlabel("Estadístico KPSS (serie de tiempo I)")
    ax.set_ylabel("N")
    ax.set_title("KPSS de curvas de luz por clase")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "histogram_kpss_lc_by_class.png"))
    plt.show()

if SHOW["amplitud"]:
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in sorted(stars_data["class"].unique()):
        a = stars_data[stars_data["class"] == cls]["amp_lc"].dropna()
        ax.hist(a, bins=30, color=colors.get(cls, "gray"), alpha=0.5, label=cls)
    ax.set_xlabel("Amplitud (mag)")
    ax.set_ylabel("N")
    ax.set_title("Amplitud de curvas de luz por clase")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "histogram_amp_lc_by_class.png"))
    plt.show()

if SHOW["deriv_sign"]:
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in sorted(stars_data["class"].unique()):
        d = stars_data[stars_data["class"] == cls]["deriv_sign_lc"].dropna()
        ax.hist(d, bins=30, color=colors.get(cls, "gray"), alpha=0.5, label=cls)
    ax.set_xlabel("Cambios de signo en derivada / N")
    ax.set_ylabel("N")
    ax.set_title("Tasa de extremos locales por clase")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "histogram_deriv_sign_by_class.png"))
    plt.show()

# --- Scatter 3D interactivo ---
# Features disponibles: "kurt_lc", "skew_lc", "amp_lc", "P", "ln_P", "kpss_lc", "deriv_sign_lc"
FEATURE_LABELS = {
    "kurt_lc":      "Curtosis",
    "skew_lc":      "Skewness",
    "amp_lc":       "Amplitud (mag)",
    "P":            "Período (días)",
    "ln_P":         "ln(Período)",
    "colour":       "Colour (V−I)",
    "kpss_lc":      "KPSS",
    "deriv_sign_lc":"Tasa extremos locales",
}

def scatter3d(df, x, y, z):
    plot_df = df[["id", "class", x, y, z]].dropna()
    fig3d = px.scatter_3d(
        plot_df, x=x, y=y, z=z,
        color="class",
        hover_name="id",
        color_discrete_map={"PULSATING": "steelblue", "BINARY": "tomato",
                            "LPV": "mediumseagreen", "OTHER": "goldenrod"},
        labels={x: FEATURE_LABELS.get(x, x),
                y: FEATURE_LABELS.get(y, y),
                z: FEATURE_LABELS.get(z, z)},
        title=f"3D: {FEATURE_LABELS.get(x,x)} · {FEATURE_LABELS.get(y,y)} · {FEATURE_LABELS.get(z,z)}",
        opacity=0.8,
    )
    fig3d.update_traces(marker=dict(size=4))
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scatter3d.html")
    fig3d.write_html(html_path)
    print(f"\nGráfica 3D guardada en: {html_path}")
    print("Abriendo en el navegador...")
    webbrowser.open(html_path)
    print("Listo. Si no se abrió sola, abre el archivo HTML manualmente.")

if SHOW["scatter_3d"]:
    scatter3d(stars_data, x="kpss_lc", y="colour", z="ln_P")

if SHOW["periodos"]:
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in sorted(stars_data["class"].unique()):
        p = stars_data[stars_data["class"] == cls]["P"].dropna()
        ax.hist(p, bins=30, color=colors.get(cls, "gray"), alpha=0.5, label=cls)
    ax.set_xlabel("Período (días)")
    ax.set_ylabel("N")
    ax.set_title("Distribución de períodos por clase")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "histogram_period_by_class.png"))
    plt.show()

if SHOW["ln_periodo"]:
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in sorted(stars_data["class"].unique()):
        lp = stars_data[stars_data["class"] == cls]["ln_P"].dropna()
        ax.hist(lp, bins=30, color=colors.get(cls, "gray"), alpha=0.5, label=cls)
    ax.set_xlabel("ln(Período)")
    ax.set_ylabel("N")
    ax.set_title("Distribución de ln(Período) por clase")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "histogram_lnP_by_class.png"))
    plt.show()

if SHOW["colour"]:
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in sorted(stars_data["class"].unique()):
        c = stars_data[stars_data["class"] == cls]["colour"].dropna()
        ax.hist(c, bins=30, color=colors.get(cls, "gray"), alpha=0.5, label=cls)
    ax.set_xlabel("Colour (V−I)")
    ax.set_ylabel("N")
    ax.set_title("Distribución de colour (V−I) por clase")
    ax.legend()
    ax.set_xlim(0,10)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "histogram_colour_by_class.png"))
    plt.show()
