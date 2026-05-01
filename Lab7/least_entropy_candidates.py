import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Literal
from scipy.signal import find_peaks


def get_test_periods(period_i, period_f, n):

    THRESHOLD   = 30.0
    FINE_WEIGHT = 2.0

    if period_f <= THRESHOLD:
        return np.linspace(period_i, period_f, n)

    if period_i >= THRESHOLD:
        u = np.sort(np.random.uniform(0, 1, n))
        u = u ** 1.8
        return period_i + u * (period_f - period_i)

    # Rango mixto
    fine_fraction     = (THRESHOLD - period_i) / (period_f - period_i)
    weighted_fraction = (fine_fraction * FINE_WEIGHT) / (
        fine_fraction * FINE_WEIGHT + (1 - fine_fraction)
    )

    n_fine   = int(round(n * weighted_fraction))
    n_sparse = n - n_fine

    fine_periods = (
        np.linspace(period_i, THRESHOLD, n_fine, endpoint=(n_sparse == 0))
        if n_fine > 0 else np.array([])
    )

    if n_sparse > 0:
        u     = np.sort(np.random.uniform(0, 1, n_sparse))
        u     = u ** 1.8
        noise = np.random.uniform(-0.5 / n_sparse, 0.5 / n_sparse, n_sparse)
        u     = np.sort(np.clip(u + noise, 0, 1))
        sparse_periods = THRESHOLD + u * (period_f - THRESHOLD)
    else:
        sparse_periods = np.array([])

    return np.concatenate([fine_periods, sparse_periods])


def get_phases(t_data, u_data, trial_period, star_type: Literal["cefeid", "rrlyrae", "binary"]):
    threshold = np.percentile(u_data, 2)
    idx       = np.argmin(np.abs(u_data - threshold))
    t0        = t_data[idx]
    return ((t_data - t0) / trial_period) % 1


def get_probabilities(phases, u_data, t_parts, u_parts):
    """Histograma 2D en el cuadrado unitario: fase en [0,1), magnitud normalizada en [0,1]."""
    phi = np.array(phases)             # ya en [0, 1)
    u   = np.array(u_data, dtype=float)

    u_min, u_max = u.min(), u.max()
    u_norm = (u - u_min) / (u_max - u_min) if u_max > u_min else np.zeros_like(u)

    N_total = len(phi)
    H, _, _ = np.histogram2d(phi, u_norm,
                              bins=[t_parts, u_parts],
                              range=[[0, 1], [0, 1]])
    return H / N_total, None   # None mantiene compatibilidad con código que desempaqueta 2 valores


def get_entropy(Mu):
    Mu_nz = Mu[Mu > 0]
    return (-Mu_nz * np.log(Mu_nz)).sum()


def find_best_period(
    data, p0, p1, p_num,
    star_type: Literal["cefeid", "rrlyrae", "binary"],
    t_parts=4, u_parts=4,
    plot_entropies=False,
    eps=0,
    plot_periods=False,
    name=None,
    p_teo=0,
    n_candidates=1,
    peak_prominence=0.01,
    peak_distance=10,
):
    testing_periods = get_test_periods(p0, p1, p_num)

    if plot_periods:
        plt.hist(testing_periods, bins=50)
        plt.xlabel("Período (días)")
        plt.title("Distribución de períodos de prueba")
        plt.show()

    # ── eps por tipo ──────────────────────────────────────────────────────────
    if eps == 0:
        eps_by_type = {
            "rrlyrae": 0.01,
            "cefeid":  0.01,
            "binary":  0.05,
        }
        eps = eps_by_type[star_type]

    # ── Centros de alias ──────────────────────────────────────────────────────
    alias_centers = (
        [n * 0.5 for n in range(1, int(p1 / 0.5) + 2)] +
        [n / 3   for n in range(1, int(p1 * 3)  + 2)]
    )

    YEARLY_EPS   = 20.0
    SMALL_YR_EPS =  7.0

    yearly_centers       = []
    half_yearly_centers  = []
    two_third_yr_centers = []
    third_yr_centers     = []
    fifth_yr_centers     = []

    if star_type == "binary":
        yearly_centers      = [n * 365.25       for n in range(1, int(p1 / 365.25)       + 2)]
        half_yearly_centers = [n * 182.625      for n in range(1, int(p1 / 182.625)      + 2)]
        # two_third_yr_centers = [n * 243.5      for n in range(1, int(p1 / 243.5)        + 2)]
        # third_yr_centers     = [n*(365.25/3)   for n in range(1, int(p1/(365.25/3))     + 2)]
        # fifth_yr_centers     = [n*(365.25/5)   for n in range(1, int(p1/(365.25/5))     + 2)]
        alias_centers += (yearly_centers + half_yearly_centers +
                          two_third_yr_centers + third_yr_centers + fifth_yr_centers)

    alias_centers        = [a for a in alias_centers        if p0 <= a <= p1]
    yearly_centers       = set(a for a in yearly_centers       if p0 <= a <= p1)
    half_yearly_centers  = set(a for a in half_yearly_centers  if p0 <= a <= p1)
    two_third_yr_centers = set(a for a in two_third_yr_centers if p0 <= a <= p1)
    third_yr_centers     = set(a for a in third_yr_centers     if p0 <= a <= p1)
    fifth_yr_centers     = set(a for a in fifth_yr_centers     if p0 <= a <= p1)

    large_yr_set = yearly_centers | half_yearly_centers | two_third_yr_centers
    small_yr_set = third_yr_centers | fifth_yr_centers

    aliasing_mask = np.zeros(len(testing_periods), dtype=bool)
    for ac in alias_centers:
        if ac in large_yr_set:
            window = YEARLY_EPS
        elif ac in small_yr_set:
            window = SMALL_YR_EPS
        else:
            window = eps
        aliasing_mask |= np.abs(testing_periods - ac) < window

    testing_periods = testing_periods[~aliasing_mask]

    # ── Calcular entropías ────────────────────────────────────────────────────
    entropies = np.zeros(len(testing_periods))
    for i, p in tqdm(enumerate(testing_periods), total=len(testing_periods)):
        phases        = get_phases(data["t"], data["u"], p, star_type)
        Mu, _         = get_probabilities(phases, data["u"], t_parts, u_parts)
        entropies[i]  = get_entropy(Mu)

    entropies = entropies / entropies.max()

    # ── buscar mínimos locales ───────────────────────────────
    peaks, properties = find_peaks(
        -entropies,
        prominence=peak_prominence,
        distance=peak_distance
    )

    if len(peaks) == 0:
        peaks = [np.argmin(entropies)]

    candidate_periods = testing_periods[peaks]
    candidate_entropies = entropies[peaks]

    order = np.argsort(candidate_entropies)

    candidate_periods = candidate_periods[order][:n_candidates]
    candidate_entropies = candidate_entropies[order][:n_candidates]

    # fases del mejor candidato
    best_period = candidate_periods[0]
    final_phases = get_phases(data["t"], data["u"], best_period, star_type)
    phases2 = final_phases + 1

    if plot_entropies:

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(testing_periods, entropies)

        if p_teo != 0:
            ax1.axvline(p_teo, c="r", linestyle="--", label="Período real")

        for i, p in enumerate(candidate_periods):
            ax1.axvline(p, linestyle="--", label=f"Candidato {i+1}")

        ax1.set_title("Periodograma")
        ax1.set_xlabel("Período (días)")
        ax1.set_ylabel("Entropía normalizada")
        ax1.legend()

        ax2.plot(1 / testing_periods, entropies)

        if p_teo != 0:
            ax2.axvline(1/p_teo, c="r", linestyle="--", label="Frecuencia real")

        for i, p in enumerate(candidate_periods):
            ax2.axvline(1/p, linestyle="--", label=f"Candidato {i+1}")

        ax2.set_title("Periodograma en frecuencia")
        ax2.set_xlabel("Frecuencia (días⁻¹)")
        ax2.set_ylabel("Entropía normalizada")
        ax2.legend()

        plt.tight_layout()
        plt.show()

    return candidate_periods, candidate_entropies, final_phases, phases2



