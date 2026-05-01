import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os 
from astropy.timeseries import LombScargle

# ------- PARÁMETROS ----------
N_freqs = int(1e3)

# -------------------------------

filenames = [
    "Lab9/LMC570.14.103.dat",
    "Lab9/LMC562.02.7937.dat",
    "Lab9/LMC562.12.10123.dat"
]

for file in filenames:
    data = pd.read_csv(file, sep=" ", names=["hjd", "I", "inc_I"])
    name = file.split(sep="/")[1]
    plt.scatter(data["hjd"], data["I"], s=2, c="k")
    plt.gca().invert_yaxis()
    plt.title(f"Curva de luz de estrella {name}")
    plt.show()

    t, mag = data["hjd"], data["I"]
    f_min = 1.0/ (t.max() - t.min())
    frequency = np.linspace(f_min,9,num=N_freqs) # d^-1
    power = LombScargle(t, mag).power(frequency, method="fastchi2")
    best_frequency = frequency[np.argmax(power)]
    best_period = 1.0 / best_frequency

    plt.plot(frequency, power)
    plt.title(f"Períodograma {name}")
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency (1/d)")
    plt.grid()
    plt.show()



