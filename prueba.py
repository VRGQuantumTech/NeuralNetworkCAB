from sctlib.analysis import Trace

trace = Trace()
trace.load_trace(source='cab')
results = trace.do_fit(baseline=(3, 0.7), mode="one-shot", verbose=True)
print(results["one-shot"].final)
 

""" import numpy as np
import matplotlib.pyplot as plt

def load_dat_real_imag(path: str):
    # 1) Intento rápido: saltar 1 línea (cabecera)
    try:
        data = np.loadtxt(path, skiprows=1)
    except ValueError:
        # 2) Alternativa robusta: genfromtxt ignora líneas no numéricas
        data = np.genfromtxt(path, skip_header=1)

    # Si hubiera columnas extra, nos quedamos con las 3 primeras
    data = data[:, :3]

    f  = data[:, 0].astype(np.float64)
    re = data[:, 1].astype(np.float64)
    im = data[:, 2].astype(np.float64)

    return f, re, im


# ---- CARGA DEL .dat ----
path = r"src/sctlib/analysis/trace/NeuralNetwork/Real_traces/Line4_DAS_PRIMA_LER8_VNAmeas_PDUT-99.0dBm_T20.0mK_0.8492GHzto0.8502GHz_IF20Hz.dat"
f, re, im = load_dat_real_imag(path)

# ---- MAGNITUD ----
mag = np.abs(re + 1j * im)

# ---- PLOT ----
plt.figure()
plt.plot(f, re, label="Re(S21)")
plt.plot(f, im, label="Im(S21)")
plt.plot(f, mag, "--", label="|S21|")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.title("Measured trace (.dat)")
plt.legend()

ax = plt.gca()
ax.tick_params(direction="in", which="both")

plt.show() """