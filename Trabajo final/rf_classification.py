import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import kpss
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# =============================================================================
# CARGA DE DATOS  (replica el preprocesamiento de code.py sin recomputar curvas)
# =============================================================================
BASE = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(BASE, "list.dat")
stars_data = pd.read_fwf(
    data_path, header=None,
    names=["id", "ra", "dec", "class", "class2", "I", "V", "P", "I_A", "P2"]
)

stars_data = stars_data[~stars_data["class"].isin(["T2CEP", "ACER", "ACEP"])].copy()
stars_data["P"] = pd.to_numeric(stars_data["P"], errors="coerce")
stars_data = stars_data[stars_data["P"].notna()].copy()

stars_data["ln_P"]   = np.log(stars_data["P"])
stars_data["colour"] = (
    pd.to_numeric(stars_data["V"], errors="coerce")
    - pd.to_numeric(stars_data["I"], errors="coerce")
)
stars_data["class"] = stars_data["class"].replace({
    "RRLYR": "PULSATING", "DSCT": "PULSATING", "DCEP": "PULSATING",
    "ECL":   "BINARY",    "ELL":  "BINARY",
})

# Cargar features de curvas de luz desde caché
CACHE_COLS = ["skew_lc", "kurt_lc", "kpss_lc", "amp_lc", "deriv_sign_lc"]
cache_path = os.path.join(BASE, "lc_features_cache.csv")

if os.path.exists(cache_path):
    cache_df = pd.read_csv(cache_path).set_index("id")
    for col in CACHE_COLS:
        stars_data[col] = stars_data["id"].map(cache_df[col])
    print(f"Features leídas desde caché: {cache_path}")
else:
    raise FileNotFoundError(
        "No se encontró lc_features_cache.csv. "
        "Corre code.py primero para generar el caché."
    )

# =============================================================================
# PREPARACIÓN
# =============================================================================
FEATURES_3   = ["colour", "ln_P", "skew_lc"]
FEATURES_ALL = ["colour", "ln_P", "skew_lc", "kurt_lc", "amp_lc"]
FEATURE_LABELS = {
    "kurt_lc":       "Curtosis",
    "skew_lc":       "Skewness",
    "amp_lc":        "Amplitud (mag)",
    "ln_P":          "ln(Período)",
    "colour":        "Colour (V−I)",
    "kpss_lc":       "KPSS",
    "deriv_sign_lc": "Tasa extremos locales",
}

COLORS = {
    "PULSATING": "steelblue",
    "BINARY":    "tomato",
    "LPV":       "mediumseagreen",
    "OTHER":     "goldenrod",
}

RANDOM_STATE = 42

def run_rf(feature_set, label):
    """Entrena RF con split 80/20 estratificado y devuelve métricas e importancias."""
    df = stars_data[["class"] + feature_set].dropna()
    X = df[feature_set].to_numpy(dtype=float)
    y = df["class"].to_numpy(dtype=str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"\n{'='*60}")
    print(f"  {label}  —  Accuracy: {acc:.4f}")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred))

    # Importancias
    importances = pd.DataFrame({
        "feature": feature_set,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return rf, y_test, y_pred, importances, report, acc

# =============================================================================
# EXPERIMENTO 1: 3 features
# =============================================================================
rf3, y_test3, y_pred3, imp3, report3, acc3 = run_rf(FEATURES_3, "3 features: colour · ln_P · skew_lc")

# =============================================================================
# EXPERIMENTO 2: todas las features
# =============================================================================
rf_all, y_test_all, y_pred_all, imp_all, report_all, acc_all = run_rf(FEATURES_ALL, "5 features (sin KPSS ni tasa extremos)")

# =============================================================================
# EXPORTAR IMPORTANCIAS A CSV
# =============================================================================
imp3["experiment"]   = "3_features"
imp_all["experiment"] = "all_features"
imp_combined = pd.concat([imp3, imp_all], ignore_index=True)
imp_path = os.path.join(BASE, "rf_feature_importances.csv")
imp_combined.to_csv(imp_path, index=False)
print(f"\nImportancias guardadas en: {imp_path}")

# Métricas globales
metrics = pd.DataFrame([
    {"experiment": "3_features_colour_lnP_skew",        "accuracy": acc3,    "n_features": len(FEATURES_3),   "features": str(FEATURES_3)},
    {"experiment": "5_features_sin_kpss_deriv",         "accuracy": acc_all, "n_features": len(FEATURES_ALL), "features": str(FEATURES_ALL)},
])
metrics_path = os.path.join(BASE, "rf_metrics.csv")
metrics.to_csv(metrics_path, index=False)
print(f"Métricas globales guardadas en: {metrics_path}")

# Reporte detallado por clase
rows = []
for exp_label, report, features in [
    ("3_features_colour_lnP_skew",  report3,    FEATURES_3),
    ("5_features_sin_kpss_deriv",   report_all, FEATURES_ALL),
]:
    for cls, vals in report.items():
        if isinstance(vals, dict):
            rows.append({"experiment": exp_label, "class": cls, **vals})
report_df = pd.DataFrame(rows)
report_path = os.path.join(BASE, "rf_classification_report.csv")
report_df.to_csv(report_path, index=False)
print(f"Reporte por clase guardado en: {report_path}")

# =============================================================================
# GRÁFICAS
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, imp, title in [
    (axes[0], imp3,   "Importancia — colour · ln_P · skew"),
    (axes[1], imp_all, "Importancia — 5 features (sin KPSS ni tasa extremos)"),
]:
    labels = [FEATURE_LABELS.get(f, f) for f in imp["feature"]]
    ax.barh(labels[::-1], imp["importance"].values[::-1], color="steelblue", edgecolor="black")
    ax.set_xlabel("Importancia (Gini)")
    ax.set_title(title)
    ax.set_xlim(0, imp["importance"].max() * 1.2)
    for i, v in enumerate(imp["importance"].values[::-1]):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)

plt.tight_layout()
fig_path = os.path.join(BASE, "rf_feature_importances.png")
plt.savefig(fig_path, dpi=150)
plt.show()
print(f"Gráfica de importancias guardada en: {fig_path}")

# Matrices de confusión
for y_t, y_p, title, suffix in [
    (y_test3,    y_pred3,    "Matriz de confusión — colour · ln_P · skew",        "3feat"),
    (y_test_all, y_pred_all, "Matriz de confusión — 5 features (sin KPSS ni deriv)", "5feat"),
]:
    classes = sorted(set(y_t) | set(y_p))
    cm = confusion_matrix(y_t, y_p, labels=classes)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
    ax_cm.set_title(title)
    plt.tight_layout()
    cm_path = os.path.join(BASE, f"rf_confusion_matrix_{suffix}.png")
    plt.savefig(cm_path, dpi=150)
    plt.show()
    print(f"Matriz de confusión guardada en: {cm_path}")

print("\nListo.")
