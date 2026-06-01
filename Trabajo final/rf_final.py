"""
rf_final.py  —  Modelo final para el informe.

Features (todos derivados de las series de tiempo, sin usar catálogo):
  - colour_bw   : biweight_location(V) - biweight_location(I)
  - ln_P_calc   : ln del período calculado con Lomb-Scargle sobre banda I
  - skew_lc     : asimetría de la curva de luz en banda I
  - amp_lc      : amplitud fotométrica en banda I

Filtrado: solo estrellas con curva de luz disponible en I y en V.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

BASE     = os.path.dirname(os.path.abspath(__file__))
I_FOLDER = os.path.join(BASE, "I")
V_FOLDER = os.path.join(BASE, "V")

# ---------------------------------------------------------------------------
# Carga y preprocesamiento
# ---------------------------------------------------------------------------
stars_data = pd.read_fwf(
    os.path.join(BASE, "list.dat"), header=None,
    names=["id", "ra", "dec", "class", "class2", "I", "V", "P", "I_A", "P2"],
)
stars_data = stars_data[~stars_data["class"].isin(["T2CEP", "ACER", "ACEP"])].copy()
stars_data["class"] = stars_data["class"].replace({
    "RRLYR": "PULSATING", "DSCT": "PULSATING", "DCEP": "PULSATING",
    "ECL":   "BINARY",    "ELL":  "BINARY",
})

# Filtro: solo estrellas con curva de luz en AMBAS bandas
mask = stars_data["id"].apply(
    lambda sid: (
        os.path.exists(os.path.join(I_FOLDER, f"{sid}.dat")) and
        os.path.exists(os.path.join(V_FOLDER, f"{sid}.dat"))
    )
)
stars_data = stars_data[mask].copy().reset_index(drop=True)
print(f"Estrellas con curvas en I y V: {len(stars_data)}")
print(stars_data["class"].value_counts())

# ---------------------------------------------------------------------------
# Cargar features biweight (colour_bw + LC features)
# ---------------------------------------------------------------------------
BW_COLS = ["colour_bw", "skew_lc", "kurt_lc", "kpss_lc", "amp_lc", "deriv_sign_lc"]
BW_CACHE = os.path.join(BASE, "lc_biweight_cache.csv")
if not os.path.exists(BW_CACHE):
    raise FileNotFoundError("No se encontró lc_biweight_cache.csv. Corre rf_biweight.py primero.")
bw_df = pd.read_csv(BW_CACHE).set_index("id")
for col in BW_COLS:
    if col in bw_df.columns:
        stars_data[col] = stars_data["id"].map(bw_df[col])

# ---------------------------------------------------------------------------
# Cargar período calculado con Lomb-Scargle
# ---------------------------------------------------------------------------
PERIOD_CACHE = os.path.join(BASE, "lc_period_cache.csv")
if not os.path.exists(PERIOD_CACHE):
    raise FileNotFoundError("No se encontró lc_period_cache.csv. Corre rf_period_search.py primero.")
pcache = pd.read_csv(PERIOD_CACHE).set_index("id")
stars_data["period_adopted"] = stars_data["id"].map(pcache["period_adopted"])
stars_data["ln_P_calc"] = np.log(stars_data["period_adopted"].clip(lower=1e-6))

print(f"Estrellas con features completas: {stars_data[['colour_bw','ln_P_calc','skew_lc','amp_lc']].notna().all(axis=1).sum()}")

# ---------------------------------------------------------------------------
# Paletas y etiquetas
# ---------------------------------------------------------------------------
COLORS = {
    "PULSATING": "steelblue",
    "BINARY":    "tomato",
    "LPV":       "mediumseagreen",
    "OTHER":     "goldenrod",
}
CLASS_ES = {
    "PULSATING": "Pulsantes",
    "BINARY":    "Binarias",
    "LPV":       "LPV",
    "OTHER":     "Otras",
}
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Histogramas en español de los 4 features finales
# ---------------------------------------------------------------------------
hist_cfg = [
    ("colour_bw", "Color BW (V−I)",    "Distribución de color (V−I) por clase",        "histogram_colour_bw_by_class.png", None),
    ("ln_P_calc", "ln(Período LS)",     "Distribución de ln(Período LS) por clase",     "histogram_lnP_ls_by_class.png",    None),
    ("skew_lc",   "Asimetría",          "Distribución de asimetría por clase",          "histogram_skew_lc_by_class.png",   None),
    ("amp_lc",    "Amplitud (mag)",     "Distribución de amplitud por clase",           "histogram_amp_lc_by_class.png",    None),
]

for feat, xlabel, title, fname, xlim in hist_cfg:
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in sorted(stars_data["class"].unique()):
        vals = stars_data[stars_data["class"] == cls][feat].dropna()
        ax.hist(vals, bins=30, color=COLORS.get(cls, "gray"), alpha=0.5,
                label=CLASS_ES.get(cls, cls))
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("N", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend()
    if xlim:
        ax.set_xlim(*xlim)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, fname), dpi=150)
    plt.close()
    print(f"Histograma guardado: {fname}")

# ---------------------------------------------------------------------------
# Modelo Random Forest — 4 features
# ---------------------------------------------------------------------------
FEATURES_4 = ["colour_bw", "ln_P_calc", "skew_lc", "amp_lc"]
FEATURE_LABELS_ES = {
    "colour_bw": "Color BW (V−I)",
    "ln_P_calc": "ln(Período LS)",
    "skew_lc":   "Asimetría",
    "amp_lc":    "Amplitud (mag)",
}

df4 = stars_data[["class"] + FEATURES_4].dropna()
X   = df4[FEATURES_4].to_numpy(dtype=float)
y   = df4["class"].to_numpy(dtype=str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
rf = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

acc    = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
print(f"\n{'='*60}")
print(f"  Modelo final 4 features  —  Exactitud: {acc:.4f}")
print(f"{'='*60}")
print(classification_report(y_test, y_pred))

# Importancias
importances = pd.DataFrame({
    "feature":    FEATURES_4,
    "importance": rf.feature_importances_,
}).sort_values("importance", ascending=False).reset_index(drop=True)
importances.to_csv(os.path.join(BASE, "rf_final4_importances.csv"), index=False)

# Métricas
rows = []
for cls, vals in report.items():
    if isinstance(vals, dict):
        rows.append({"class": cls, **vals})
pd.DataFrame(rows).to_csv(os.path.join(BASE, "rf_final4_report.csv"), index=False)

# ---------------------------------------------------------------------------
# Gráfica de importancias
# ---------------------------------------------------------------------------
labels = [FEATURE_LABELS_ES[f] for f in importances["feature"]]
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(labels[::-1], importances["importance"].values[::-1],
        color="steelblue", edgecolor="black")
ax.set_xlabel("Importancia (Gini)", fontsize=12)
ax.set_title("Importancia de variables — modelo final (4 features)", fontsize=12)
ax.set_xlim(0, importances["importance"].max() * 1.28)
for i, v in enumerate(importances["importance"].values[::-1]):
    ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "rf_final4_importances.png"), dpi=150)
plt.close()
print("Importancias guardadas: rf_final4_importances.png")

# ---------------------------------------------------------------------------
# Matriz de confusión
# ---------------------------------------------------------------------------
classes_order = ["BINARY", "LPV", "OTHER", "PULSATING"]
labels_es     = ["Binarias", "LPV", "Otras", "Pulsantes"]

cm = confusion_matrix(y_test, y_pred, labels=classes_order)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_es)
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_xlabel("Clase predicha", fontsize=12)
ax.set_ylabel("Clase real", fontsize=12)
ax.set_title("Matriz de confusión — modelo final (4 features)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "rf_final4_confusion.png"), dpi=150)
plt.close()
print("Matriz de confusión guardada: rf_final4_confusion.png")

print(f"\nExactitud global: {acc:.4f}")
print("Listo.")
