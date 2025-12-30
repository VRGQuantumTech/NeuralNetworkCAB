import numpy as np
import matplotlib.pyplot as plt
import torch

from lorentzian_generator import lorentzian_generator
from network import Net


def load_trained_model(model_path="kc_predictor.pt"):
    ckpt = torch.load(model_path, map_location="cpu")

    net = Net(
        input_dim=ckpt["input_dim"],
        output_dim=ckpt["output_dim"],
        n_units=ckpt["n_units"],
        epochs=1,      # no se usan para inferencia
        lr=1e-3,
    )
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()  # importante
    return net, ckpt


def main():
    # --- mismos parámetros que tu training.py ---
    cavity_params = {
        "ac": 1.0,
        "dt": 0.0,
        "phi": 0.0,
        "dphi": 0.0,
        "kappai": 1.0e3,
        "fr": 1e9,
    }

    kc_limits = (1e4, 1e5)
    ki = cavity_params["kappai"]
    fr = cavity_params["fr"]
    delta_f_max = 10 * kc_limits[1] + ki
    frequency_limits = (fr - delta_f_max, fr + delta_f_max)

    frequency_points = 2000

    # --- carga modelo entrenado ---
    net, ckpt = load_trained_model("kc_predictor.pt")

    # --- barrido de sigmas (misma escala que usabas: 0.001 a 0.05) ---
    sigmas = np.geomspace(1e-3, 5e-2, 12)

    # tamaño de test por sigma (ajusta si quieres más estabilidad)
    n_samples = 1000

    mae_rel = []
    rmse_rel = []
    p95_abs_rel = []

    # para reproducibilidad
    np.random.seed(0)

    for sigma in sigmas:
        # genera dataset con ESTE sigma
        f, X_meas, X_clean, kc_true = lorentzian_generator(
            n_samples=n_samples,
            cavity_params=cavity_params,
            kc_limits=kc_limits,
            frequency_limits=frequency_limits,
            frequency_points=frequency_points,
            noise_std_signal=float(sigma),
        )

        # mismo preprocesado que tú: y = log(kc)
        X = X_meas.astype(np.float32)
        y_true = np.log(kc_true).reshape(-1, 1).astype(np.float32)

        # predicción (sale en log(Kc))
        y_pred = net.predict(X).astype(np.float32)

        # volvemos a Kc lineal
        kc_pred = np.exp(y_pred).flatten()
        kc_true_lin = np.exp(y_true).flatten()

        # error relativo por muestra
        rel = (kc_pred - kc_true_lin) / kc_true_lin
        abs_rel = np.abs(rel)

        mae_rel.append(np.mean(abs_rel))
        rmse_rel.append(np.sqrt(np.mean(rel**2)))
        p95_abs_rel.append(np.percentile(abs_rel, 95))

    mae_rel = np.array(mae_rel)
    rmse_rel = np.array(rmse_rel)
    p95_abs_rel = np.array(p95_abs_rel)

    # --- gráfica error vs sigma ---
    plt.figure()
    plt.plot(sigmas, mae_rel, marker="o", label="MAE relativo (|err|)")
    plt.plot(sigmas, rmse_rel, marker="o", label="RMSE relativo")
    plt.plot(sigmas, p95_abs_rel, marker="o", label="P95 |err| relativo")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Sigma (ruido)")
    plt.ylabel("Error relativo")
    plt.title("Robustez al ruido: error de predicción vs sigma")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("error_vs_sigma.png", dpi=200)
    plt.show()

    print("Saved plot: error_vs_sigma.png")


if __name__ == "__main__":
    main()
