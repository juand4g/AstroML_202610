import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import kpss
from astropy.stats import biweight_location
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# =============================================================================
# CARGA Y PREPROCESAMIENTO
# =============================================================================
BASE     = os.path.dirname(os.path.abspath(__file__))
I_FOLDER = os.path.join(BASE, "I")
V_FOLDER = os.path.join(BASE, "V")

data_path = os.path.join(BASE, "list.dat")
stars_data = pd.read_fwf(
    data_path, header=None,
    names=["id", "ra", "dec", "class", "class2", "I", "V", "P", "I_A", "P2"]
)

stars_data = stars_data[~stars_data["class"].isin(["T2CEP", "ACER", "ACEP"])].copy()
stars_data["P"] = pd.to_numeric(stars_data["P"], errors="coerce")
stars_data["ln_P"] = np.log(stars_data["P"])  # NaN para estrellas sin período en catálogo
stars_data["class"] = stars_data["class"].replace({
    "RRLYR": "PULSATING", "DSCT": "PULSATING", "DCEP": "PULSATING",
    "ECL":   "BINARY",    "ELL":  "BINARY",
})

# =============================================================================
# PREFILTRADO: conservar solo estrellas con .dat en V/ y en I/
# =============================================================================
def has_both_bands(star_id):
    return (
        os.path.exists(os.path.join(I_FOLDER, f"{star_id}.dat")) and
        os.path.exists(os.path.join(V_FOLDER, f"{star_id}.dat"))
    )

mask_both = stars_data["id"].apply(has_both_bands)
stars_data = stars_data[mask_both].copy().reset_index(drop=True)
print(f"Estrellas con curvas en ambas bandas (I y V): {len(stars_data)}")
print(f"Distribución de clases:\n{stars_data['class'].value_counts()}\n")

# =============================================================================
# CÓMPUTO DE FEATURES (con caché propio)
# =============================================================================
BW_CACHE_COLS = ["colour_bw", "skew_lc", "kurt_lc", "kpss_lc", "amp_lc", "deriv_sign_lc"]
BW_CACHE_PATH = os.path.join(BASE, "lc_biweight_cache.csv")

def biweight_features(star_id):
    """Lee ambas bandas, calcula colour via Biweight Location + features de la banda I."""
    i_path = os.path.join(I_FOLDER, f"{star_id}.dat")
    v_path = os.path.join(V_FOLDER, f"{star_id}.dat")
    nan_row = pd.Series([np.nan] * len(BW_CACHE_COLS), index=BW_CACHE_COLS)
    try:
        mag_i = pd.to_numeric(
            pd.read_csv(i_path, sep=r'\s+', header=None, names=["hjd","mag","e"])["mag"],
            errors="coerce"
        ).dropna().values
        mag_v = pd.to_numeric(
            pd.read_csv(v_path, sep=r'\s+', header=None, names=["hjd","mag","e"])["mag"],
            errors="coerce"
        ).dropna().values

        if len(mag_i) < 10 or len(mag_v) < 10:
            return nan_row

        colour_bw = float(biweight_location(mag_v)) - float(biweight_location(mag_i))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat = kpss(mag_i, regression="c", nlags="auto")[0]

        amplitude = mag_i.max() - mag_i.min()
        diffs = np.sign(np.diff(mag_i))
        diffs = diffs[diffs != 0]
        sign_changes = np.sum(np.diff(diffs) != 0) / len(mag_i)

        return pd.Series(
            [colour_bw, skew(mag_i), kurtosis(mag_i), kpss_stat, amplitude, sign_changes],
            index=BW_CACHE_COLS
        )
    except FileNotFoundError:
        return nan_row

if os.path.exists(BW_CACHE_PATH):
    cache_df = pd.read_csv(BW_CACHE_PATH)
    if all(c in cache_df.columns for c in BW_CACHE_COLS):
        cache_df = cache_df.set_index("id")
        for col in BW_CACHE_COLS:
            stars_data[col] = stars_data["id"].map(cache_df[col])
        print(f"Features leídas desde caché: {BW_CACHE_PATH}")
    else:
        os.remove(BW_CACHE_PATH)
        print("Caché incompleto, recalculando...")

if not os.path.exists(BW_CACHE_PATH):
    tqdm.pandas(desc="Calculando features (biweight colour + I-band)")
    feats = stars_data["id"].progress_apply(biweight_features)
    stars_data[BW_CACHE_COLS] = feats
    stars_data[["id"] + BW_CACHE_COLS].to_csv(BW_CACHE_PATH, index=False)
    print(f"Features guardadas en caché: {BW_CACHE_PATH}")

n_valid = stars_data["colour_bw"].notna().sum()
print(f"Estrellas con features completas: {n_valid} / {len(stars_data)}\n")

# =============================================================================
# CONFIGURACIÓN DE EXPERIMENTOS
# =============================================================================
FEATURES_3   = ["colour_bw", "ln_P", "skew_lc"]
FEATURES_ALL = ["colour_bw", "ln_P", "skew_lc", "kurt_lc", "amp_lc"]

FEATURE_LABELS = {
    "colour_bw":     "Colour BW (V−I)",
    "ln_P":          "ln(Período)",
    "skew_lc":       "Skewness",
    "kurt_lc":       "Curtosis",
    "amp_lc":        "Amplitud (mag)",
    "kpss_lc":       "KPSS",
    "deriv_sign_lc": "Tasa extremos locales",
}

RANDOM_STATE = 42

# =============================================================================
# RANDOM FOREST
# =============================================================================
def run_rf(feature_set, label):
    df = stars_data[["class"] + feature_set].dropna()
    X  = df[feature_set].to_numpy(dtype=float)
    y  = df["class"].to_numpy(dtype=str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"\n{'='*60}")
    print(f"  {label}  —  Accuracy: {acc:.4f}")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred))

    importances = pd.DataFrame({
        "feature":    feature_set,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return rf, y_test, y_pred, importances, report, acc

rf3,    y_t3,   y_p3,   imp3,   rep3,   acc3   = run_rf(FEATURES_3,   "3 features: colour_bw · ln_P · skew")
rf_all, y_tall, y_pall, imp_all, rep_all, acc_all = run_rf(FEATURES_ALL, "5 features (sin KPSS ni tasa extremos) — colour BW")

# =============================================================================
# EXPORTAR RESULTADOS
# =============================================================================

# Importancias
imp3["experiment"]    = "3feat_bw"
imp_all["experiment"] = "5feat_bw"
imp_combined = pd.concat([imp3, imp_all], ignore_index=True)
imp_path = os.path.join(BASE, "rf_bw_feature_importances.csv")
imp_combined.to_csv(imp_path, index=False)
print(f"\nImportancias guardadas en: {imp_path}")

# Métricas globales
metrics = pd.DataFrame([
    {"experiment": "3feat_bw",  "accuracy": acc3,    "n_features": len(FEATURES_3),   "features": str(FEATURES_3)},
    {"experiment": "5feat_bw",  "accuracy": acc_all, "n_features": len(FEATURES_ALL), "features": str(FEATURES_ALL)},
])
metrics_path = os.path.join(BASE, "rf_bw_metrics.csv")
metrics.to_csv(metrics_path, index=False)
print(f"Métricas globales guardadas en: {metrics_path}")

# Reporte por clase
rows = []
for exp_label, report in [("3feat_bw", rep3), ("5feat_bw", rep_all)]:
    for cls, vals in report.items():
        if isinstance(vals, dict):
            rows.append({"experiment": exp_label, "class": cls, **vals})
pd.DataFrame(rows).to_csv(os.path.join(BASE, "rf_bw_classification_report.csv"), index=False)
print(f"Reporte por clase guardado en: {os.path.join(BASE, 'rf_bw_classification_report.csv')}")

# =============================================================================
# GRÁFICAS
# =============================================================================

# Importancias
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, imp, title in [
    (axes[0], imp3,   "Importancia — colour BW · ln_P · skew"),
    (axes[1], imp_all, "Importancia — 5 features (colour BW)"),
]:
    labels = [FEATURE_LABELS.get(f, f) for f in imp["feature"]]
    ax.barh(labels[::-1], imp["importance"].values[::-1], color="darkorange", edgecolor="black")
    ax.set_xlabel("Importancia (Gini)")
    ax.set_title(title)
    ax.set_xlim(0, imp["importance"].max() * 1.2)
    for i, v in enumerate(imp["importance"].values[::-1]):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)

plt.tight_layout()
fig_path = os.path.join(BASE, "rf_bw_feature_importances.png")
plt.savefig(fig_path, dpi=150)
plt.show()
print(f"Gráfica de importancias guardada en: {fig_path}")

# Matrices de confusión
for y_t, y_p, title, suffix in [
    (y_t3,   y_p3,   "Confusión — colour BW · ln_P · skew",        "bw_3feat"),
    (y_tall, y_pall, "Confusión — 5 features (colour BW)",          "bw_5feat"),
]:
    classes = sorted(set(y_t) | set(y_p))
    cm  = confusion_matrix(y_t, y_p, labels=classes)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(
        ax=ax_cm, colorbar=False, cmap="Oranges"
    )
    ax_cm.set_title(title)
    plt.tight_layout()
    cm_path = os.path.join(BASE, f"rf_confusion_matrix_{suffix}.png")
    plt.savefig(cm_path, dpi=150)
    plt.show()
    print(f"Matriz de confusión guardada en: {cm_path}")

print("\nListo.")
