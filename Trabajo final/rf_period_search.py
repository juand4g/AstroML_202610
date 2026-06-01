import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from astropy.timeseries import LombScargle
# from entropypf import find_best_period  # reservado para estrategia LS+ME futura
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# =============================================================================
# HIPERPARÁMETROS DE BÚSQUEDA DE PERÍODO  (ajusta aquí)
# =============================================================================
P_MIN        = 0.1      # límite inferior del rango de búsqueda (días)
P_MAX        = 150.0    # límite superior del rango de búsqueda (días)
N_TRIALS     = 20_000   # puntos de la grilla de frecuencias para LS
# N_CANDIDATES = 5      # (futuro) candidatos de entropypf a comparar con LS
# MATCH_TOL    = 0.05   # (futuro) tolerancia relativa para acuerdo LS–ME
# ENT_ALIASES  = (1,)   # (futuro) aliases a enmascarar en entropypf
RECOVER_TOL  = 0.05     # tolerancia para la evaluación de recuperación de períodos
RANDOM_STATE = 42

# =============================================================================
# CARGA Y PREPROCESAMIENTO
# =============================================================================
BASE     = os.path.dirname(os.path.abspath(__file__))
I_FOLDER = os.path.join(BASE, "I")
V_FOLDER = os.path.join(BASE, "V")

stars_data = pd.read_fwf(
    os.path.join(BASE, "list.dat"), header=None,
    names=["id", "ra", "dec", "class", "class2", "I", "V", "P", "I_A", "P2"]
)
stars_data = stars_data[~stars_data["class"].isin(["T2CEP", "ACER", "ACEP"])].copy()
stars_data["P_catalog"] = pd.to_numeric(stars_data["P"], errors="coerce")
# No se filtra por período de catálogo: el período se calcula de la curva de luz
stars_data["class"] = stars_data["class"].replace({
    "RRLYR": "PULSATING", "DSCT": "PULSATING", "DCEP": "PULSATING",
    "ECL":   "BINARY",    "ELL":  "BINARY",
})

# Prefiltrado: solo estrellas con .dat en ambas bandas
mask = stars_data["id"].apply(
    lambda sid: (
        os.path.exists(os.path.join(I_FOLDER, f"{sid}.dat")) and
        os.path.exists(os.path.join(V_FOLDER, f"{sid}.dat"))
    )
)
stars_data = stars_data[mask].copy().reset_index(drop=True)
print(f"Estrellas con I y V: {len(stars_data)}")
print(f"Distribución de clases:\n{stars_data['class'].value_counts()}\n")

# =============================================================================
# CARGA DE FEATURES DE BIWEIGHT (colour_bw + LC features)
# =============================================================================
BW_COLS       = ["colour_bw", "skew_lc", "kurt_lc", "kpss_lc", "amp_lc", "deriv_sign_lc"]
BW_CACHE_PATH = os.path.join(BASE, "lc_biweight_cache.csv")

if os.path.exists(BW_CACHE_PATH):
    bw_cache = pd.read_csv(BW_CACHE_PATH).set_index("id")
    for col in BW_COLS:
        if col in bw_cache.columns:
            stars_data[col] = stars_data["id"].map(bw_cache[col])
    print(f"Biweight features cargadas desde: {BW_CACHE_PATH}")
else:
    raise FileNotFoundError(
        "No se encontró lc_biweight_cache.csv. Corre rf_biweight.py primero."
    )

# =============================================================================
# BÚSQUEDA DE PERÍODO: LOMB-SCARGLE + ENTROPÍA MÍNIMA
# =============================================================================
PERIOD_COLS  = ["period_ls", "period_ent", "period_adopted", "ls_ent_agree"]
PERIOD_CACHE = os.path.join(BASE, "lc_period_cache.csv")

# Grilla de frecuencias para LS (se computa una sola vez)
LS_FREQS = np.linspace(1.0 / P_MAX, 1.0 / P_MIN, N_TRIALS)

def search_period(star_id):
    """LS + entropypf sobre la banda I. Devuelve Series con los 4 campos."""
    i_path  = os.path.join(I_FOLDER, f"{star_id}.dat")
    nan_row = pd.Series([np.nan, np.nan, np.nan, 0], index=PERIOD_COLS)
    try:
        lc = pd.read_csv(i_path, sep=r'\s+', header=None, names=["hjd", "mag", "e"])
        lc["hjd"] = pd.to_numeric(lc["hjd"], errors="coerce")
        lc["mag"] = pd.to_numeric(lc["mag"], errors="coerce")
        lc = lc.dropna(subset=["hjd", "mag"])
        t   = lc["hjd"].values
        mag = lc["mag"].values
        if len(t) < 10:
            return nan_row

        # --- Lomb-Scargle (máximo absoluto de potencia) ---
        power     = LombScargle(t, mag).power(LS_FREQS)
        period_ls = float(1.0 / LS_FREQS[np.argmax(power)])

        # --- (futuro) Entropía mínima + lógica de acuerdo LS–ME ---
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        #     ent_periods, _ = find_best_period(
        #         t, mag, p0=P_MIN, p1=P_MAX, p_num=N_TRIALS,
        #         L=7, K=7, aliases=ENT_ALIASES, n_candidates=N_CANDIDATES,
        #     )
        # period_ent_best = float(ent_periods[0])
        # period_adopted = period_ls
        # agree = 0
        # for p_ent in ent_periods:
        #     if abs(period_ls - float(p_ent)) / period_ls < MATCH_TOL:
        #         period_adopted = float(p_ent); agree = 1; break

        return pd.Series([period_ls, np.nan, period_ls, 0],
                         index=PERIOD_COLS)
    except Exception:
        return nan_row


if os.path.exists(PERIOD_CACHE):
    pcache = pd.read_csv(PERIOD_CACHE)
    if all(c in pcache.columns for c in PERIOD_COLS):
        pcache = pcache.set_index("id")
        for col in PERIOD_COLS:
            stars_data[col] = stars_data["id"].map(pcache[col])
        print(f"Períodos leídos desde caché: {PERIOD_CACHE}")
    else:
        os.remove(PERIOD_CACHE)
        print("Caché de períodos incompleto, recalculando...")

if not os.path.exists(PERIOD_CACHE):
    tqdm.pandas(desc="Buscando períodos (Lomb-Scargle)")
    period_feats = stars_data["id"].progress_apply(search_period)
    stars_data[PERIOD_COLS] = period_feats
    stars_data[["id"] + PERIOD_COLS].to_csv(PERIOD_CACHE, index=False)
    print(f"Períodos guardados en caché: {PERIOD_CACHE}")

stars_data["ls_ent_agree"] = stars_data["ls_ent_agree"].astype(float).fillna(0).astype(bool)
stars_data["ln_P_calc"]    = np.log(stars_data["period_adopted"].clip(lower=1e-6))

n_found = stars_data["period_adopted"].notna().sum()
print(f"\nPeríodos calculados (LS): {n_found} / {len(stars_data)}\n")

# =============================================================================
# DIAGNÓSTICO: RECUPERACIÓN DE PERÍODOS  (solo para inspección — no entra al RF)
# =============================================================================
def period_recovered(p_calc, p_cat, tol=RECOVER_TOL):
    """True si p_calc está dentro de `tol` relativo de p_cat, p_cat/2 o 2·p_cat."""
    for factor in [1.0, 0.5, 2.0]:
        if abs(p_calc - factor * p_cat) / (factor * p_cat) < tol:
            return True
    return False

eval_mask = stars_data["period_adopted"].notna() & stars_data["P_catalog"].notna()
eval_df   = stars_data[eval_mask].copy()
eval_df["recovered"]    = eval_df.apply(
    lambda r: period_recovered(r["period_adopted"], r["P_catalog"]), axis=1
)
eval_df["period_ratio"] = eval_df["period_adopted"] / eval_df["P_catalog"]

total_rec = eval_df["recovered"].sum()
total_val = len(eval_df)
print("=" * 60)
print("  RECUPERACIÓN DE PERÍODOS (diagnóstico — no entra al RF)")
print("=" * 60)
print(f"  Global : {total_rec}/{total_val}  ({100*total_rec/total_val:.1f}%)  "
      f"[tolerancia ±{int(RECOVER_TOL*100)}%, incluyendo mitad/doble período]")
class_colors = {"PULSATING": "steelblue", "BINARY": "tomato",
                "LPV": "mediumseagreen", "OTHER": "goldenrod"}
for cls, grp in eval_df.groupby("class"):
    r, n = grp["recovered"].sum(), len(grp)
    print(f"  {cls:12s}: {r:4d}/{n:4d}  ({100*r/n:.1f}%)")
print()

rec_path = os.path.join(BASE, "rf_ps_period_recovery.csv")
eval_df[[
    "id", "class", "P_catalog", "period_ls", "period_ent",
    "period_adopted", "period_ratio", "recovered", "ls_ent_agree"
]].to_csv(rec_path, index=False)
print(f"Tabla de recuperación guardada en: {rec_path}")

# =============================================================================
# RANDOM FOREST  (con ln_P_calc en lugar de ln_P del catálogo)
# =============================================================================
FEATURES_3   = ["colour_bw", "ln_P_calc", "skew_lc"]
FEATURES_ALL = ["colour_bw", "ln_P_calc", "skew_lc", "amp_lc"]

FEATURE_LABELS = {
    "colour_bw":  "Colour BW (V−I)",
    "ln_P_calc":  "ln(P calculado)",
    "skew_lc":    "Skewness",
    "kurt_lc":    "Curtosis",
    "amp_lc":     "Amplitud (mag)",
}

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

rf3,    y_t3,   y_p3,   imp3,    rep3,    acc3    = run_rf(
    FEATURES_3,   "3 feat.: colour_bw · ln_P_calc · skew"
)
rf_all, y_tall, y_pall, imp_all, rep_all, acc_all = run_rf(
    FEATURES_ALL, "4 feat. — período calculado (sin KPSS ni deriv)"
)

# =============================================================================
# EXPORTAR RESULTADOS
# =============================================================================
imp3["experiment"]    = "3feat_ps"
imp_all["experiment"] = "4feat_ps"
pd.concat([imp3, imp_all], ignore_index=True).to_csv(
    os.path.join(BASE, "rf_ps_feature_importances.csv"), index=False
)
pd.DataFrame([
    {"experiment": "3feat_ps",  "accuracy": acc3,    "n_features": len(FEATURES_3),   "features": str(FEATURES_3)},
    {"experiment": "4feat_ps",  "accuracy": acc_all, "n_features": len(FEATURES_ALL), "features": str(FEATURES_ALL)},
]).to_csv(os.path.join(BASE, "rf_ps_metrics.csv"), index=False)

rows = []
for exp_label, report in [("3feat_ps", rep3), ("4feat_ps", rep_all)]:
    for cls, vals in report.items():
        if isinstance(vals, dict):
            rows.append({"experiment": exp_label, "class": cls, **vals})
pd.DataFrame(rows).to_csv(os.path.join(BASE, "rf_ps_classification_report.csv"), index=False)

print("\nCSVs guardados.")

# =============================================================================
# GRÁFICAS
# =============================================================================
# — Importancias de features —
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, imp, title in [
    (axes[0], imp3,   "Importancia — 3 feat. (P calculado)"),
    (axes[1], imp_all, "Importancia — 4 feat. (P calculado)"),
]:
    labels = [FEATURE_LABELS.get(f, f) for f in imp["feature"]]
    ax.barh(labels[::-1], imp["importance"].values[::-1], color="mediumseagreen", edgecolor="black")
    ax.set_xlabel("Importancia (Gini)")
    ax.set_title(title)
    ax.set_xlim(0, imp["importance"].max() * 1.2)
    for i, v in enumerate(imp["importance"].values[::-1]):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "rf_ps_feature_importances.png"), dpi=150)
plt.show()

# — Matrices de confusión —
for y_t, y_p, title, suffix in [
    (y_t3,   y_p3,   "Confusión — 3 feat. (P calc.)", "ps_3feat"),
    (y_tall, y_pall, "Confusión — 4 feat. (P calc.)", "ps_5feat"),
]:
    classes = sorted(set(y_t) | set(y_p))
    cm = confusion_matrix(y_t, y_p, labels=classes)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(
        ax=ax_cm, colorbar=False, cmap="Greens"
    )
    ax_cm.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, f"rf_confusion_matrix_{suffix}.png"), dpi=150)
    plt.show()

# — Histograma de ratio período calculado / catálogo —
fig_r, ax_r = plt.subplots(figsize=(9, 4))
for cls, grp in eval_df.groupby("class"):
    ax_r.hist(
        grp["period_ratio"].clip(0, 3), bins=80,
        alpha=0.5, color=class_colors.get(cls, "gray"), label=cls
    )
for xval, ls, lbl in [(1.0, "--", "P_calc = P_cat"),
                       (0.5, ":",  "P_calc = P_cat/2"),
                       (2.0, ":",  "P_calc = 2·P_cat")]:
    ax_r.axvline(xval, color="black", ls=ls, lw=1.5, label=lbl)
ax_r.set_xlabel("P_calc / P_catálogo")
ax_r.set_ylabel("N")
ax_r.set_title("Ratio período calculado vs. catálogo (diagnóstico)")
ax_r.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "rf_ps_period_ratio.png"), dpi=150)
plt.show()

print("\nListo.")
