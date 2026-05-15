import os
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt

# HIPERPARÁMETROS
SIZE=5
SAMPLES=3

archivo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HOJASCLASIFICAR.xlsx")
df = pd.read_excel(archivo)
df["ratio"] = df["largo"]/df["ancho"]
clustering_on = ["n_vertices", "largo", "ancho"]

# Datos de la región, escalados
data = RobustScaler().fit_transform(df[clustering_on])

# Modelo HDBSCAN
hd = HDBSCAN(min_cluster_size=SIZE, min_samples=SAMPLES, metric='euclidean').fit(data)

# Reclasificar clusters con más de 1000 elementos como ruido
labels = hd.labels_.copy()

df['label_hb'] = labels
unique_labels = sorted(set(labels))
num_clusters = len(unique_labels)

# --- Métricas ---
CLUSTER_TO_CLASS = {0: 2, 1: 0, 2: 1, -1: 3}
CLASS_NAMES = {0: "Alargada", 1: "Ovalada", 2: "Estrellada", 3: "Helechosa"}

y_true = df["clase"].values
y_pred = np.array([CLUSTER_TO_CLASS[l] for l in labels])

print("\n=== Accuracy por clase (% de aciertos sobre el total real de la clase) ===")
for cls in sorted(CLASS_NAMES):
    total = np.sum(y_true == cls)
    aciertos = np.sum((y_true == cls) & (y_pred == cls))
    acc = aciertos / total * 100 if total > 0 else 0.0
    print(f"  {CLASS_NAMES[cls]:12s} (clase {cls}): {acc:.1f}%  ({aciertos}/{total})")

print("\n=== F1 score ===")
nombres = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]
print(classification_report(y_true, y_pred, target_names=nombres))

cmap = plt.cm.get_cmap('tab10', max(num_clusters, 1))

def cluster_label_name(lbl):
    return CLASS_NAMES[CLUSTER_TO_CLASS[lbl]]

if len(clustering_on) == 3:
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    for lbl in unique_labels:
        mask = labels == lbl
        color = 'gray' if lbl == -1 else cmap(lbl)
        ax.scatter(
            df.loc[mask, clustering_on[0]], df.loc[mask, clustering_on[1]], df.loc[mask, clustering_on[2]],
            c=[color], label=cluster_label_name(lbl), s=20, alpha=0.6
        )
    ax.set_xlabel(clustering_on[0])
    ax.set_ylabel(clustering_on[1])
    ax.set_zlabel(clustering_on[2])
    ax.set_title('HDBSCAN — clasificación de hojas')
    ax.legend(loc='best', fontsize=8)

elif len(clustering_on) == 2:
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    for lbl in unique_labels:
        mask = labels == lbl
        color = 'gray' if lbl == -1 else cmap(lbl)
        ax.scatter(
            df.loc[mask, clustering_on[0]], df.loc[mask, clustering_on[1]],
            c=[color], label=cluster_label_name(lbl), s=20, alpha=0.6
        )
    ax.set_xlabel(clustering_on[0])
    ax.set_ylabel(clustering_on[1])
    ax.set_title('HDBSCAN — clasificación de hojas')
    ax.legend(loc='best', fontsize=8)

else:
    raise ValueError(f"clustering_on debe tener 2 o 3 elementos, tiene {len(clustering_on)}")

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hdbscan_clustering.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nGráfica guardada en: {out_path}")
plt.show()
