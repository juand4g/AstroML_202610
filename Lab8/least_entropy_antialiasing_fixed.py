import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Literal

def get_test_periods(period_i,period_f,n):
    return np.linspace(period_i,period_f,n)

def get_phases(t_data, u_data, trial_period):
    threshold = np.percentile(u_data, 2)  # percentil 2% = los más bajos
    idx = np.argmin(np.abs(u_data - threshold))  # índice del valor más cercano al percentil
    t0 = t_data[idx]

    return ((t_data - t0) / trial_period) % 1

def get_probabilities(phases, u_data, t_parts, u_parts):
    t = np.array(phases)
    u = np.array(u_data)
    
    N_total = len(t)
    
    t_min, t_max = t.min(), t.max()
    u_min, u_max = u.min(), u.max()
    
    t_edges = np.linspace(t_min, t_max, t_parts + 1)
    u_edges = np.linspace(u_min, u_max, u_parts + 1)
    
    H, _, _ = np.histogram2d(t, u, bins=[t_edges, u_edges])
    
    Mu = H / N_total
    
    return Mu, (t_edges, u_edges)

def get_entropy(Mu):
    Mu = Mu[np.nonzero(Mu)]
    S = -Mu*np.log(Mu)
    entropy = S.sum()

    return entropy

def find_best_period(data, p0, p1, p_num, t_parts=4, u_parts=4, plot_entropies=False, eps=0.0025):
    testing_periods = get_test_periods(p0, p1, p_num)

    # Filtrar aliases
    mask_half_day = (
        np.isclose(testing_periods % 0.5, 0, atol=1e-3) |
        np.isclose(testing_periods % 0.5, 0.5, atol=1e-3)
    )
    mask_third_day = np.isclose(testing_periods % (1/3), 0, atol=1e-3)
    
    # Expandir ventana eps alrededor de cada alias
    aliasing_mask = mask_half_day | mask_third_day
    N = testing_periods.shape[0]
    idx_window = int(eps * N)
    for idx in np.where(aliasing_mask)[0]:
        window_start = max(0, idx - idx_window)
        window_end = min(N, idx + idx_window + 1)
        aliasing_mask[window_start:window_end] = True

    testing_periods = testing_periods[~aliasing_mask]

    entropies = np.zeros_like(testing_periods)
    for i, p in tqdm(enumerate(testing_periods)):
        phases = get_phases(data["t"], data["u"], p)
        Probabilities, _ = get_probabilities(phases, data["u"], t_parts=t_parts, u_parts=u_parts)
        entropies[i] = get_entropy(Probabilities)

    entropies = entropies / entropies.max()
    

    min_idx = np.argmin(entropies)  # Encuentra el índice del mínimo
    best_period = testing_periods[min_idx]  # Obtiene el período en ese índice
    min_entropy = entropies[min_idx]  # Obtiene el valor mínimo

    final_phases = get_phases(data["t"], data["u"], best_period)
    phases2 = final_phases+1

    if plot_entropies:
        plt.plot(testing_periods,entropies)
        plt.title("Periodograma")
        plt.xlim(0.475,0.525)
        plt.xlabel("Período de prueba")
        plt.ylabel("Entropía")
        #for idx in aliasing_idx:
            #plt.axvline(testing_periods[idx], color='red', linestyle='--', alpha=0.5, label='Alias' if idx == aliasing_idx[0] else "")
        plt.show()

    return best_period, min_entropy, final_phases, phases2










