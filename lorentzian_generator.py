import numpy as np
from lorentzian import lorentzian as lorentzian_cy
import matplotlib.pyplot as plt
from sctlib.analysis import Trace



def plot_kappa_window_debug(f, I, Q, f_win, I_win, Q_win, info):
    """
    Visualiza la traza completa y la ventana recortada por padder_half_kappa_window.
    """
    amp = np.sqrt(I**2 + Q**2)
    amp_win = np.sqrt(I_win**2 + Q_win**2)

    idx_min = info["idx_min"]
    idx_left = info["idx_left"]
    idx_right = info["idx_right"]
    A_base = info["A_base"]
    A_min = info["A_min"]
    A_half = info["A_half"]

    plt.figure(figsize=(10, 4.5), dpi=200)

    # Traza completa
    plt.plot(f, amp, color="gray", alpha=0.6, label="|S21| completo")

    # Ventana recortada
    plt.plot(f_win, amp_win, color="crimson", linewidth=2.2, label="Ventana recortada")

    # Líneas horizontales
    plt.axhline(A_base, color="black", linestyle="--", linewidth=1.2, label="Baseline")
    if A_half is not None:
        plt.axhline(A_half, color="blue", linestyle=":", linewidth=1.2, label="Nivel mitad")

    # Marcas verticales
    plt.axvline(f[idx_left], color="green", linestyle="--", linewidth=1.2, label="Corte izq.")
    plt.axvline(f[idx_right], color="green", linestyle="--", linewidth=1.2, label="Corte der.")

    # Mínimo
    plt.scatter(f[idx_min], A_min, color="black", zorder=5, label="Mínimo")

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("|S21|")
    plt.title("Ventana alrededor del dip (κ aproximado)")
    plt.legend(frameon=False)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

def padder_1db_tail(
    f, I, Q,
    max_F,
    db_threshold=1.0,
    noise_std=0.0,
):
    """
    Padding físico desde el final de la traza.
    
    Parámetros
    ----------
    f, I, Q : arrays (Fi,)
        Traza real
    max_F : int
        Longitud final deseada
    db_threshold : float
        Umbral en dB (por defecto 1 dB)
    noise_std : float
        Ruido gaussiano opcional en el padding

    Returns
    -------
    f_pad, I_pad, Q_pad : arrays (max_F,)
    mask : (max_F,)  -> 1 real, 0 padding
    """
    Fi = len(f)
    assert Fi <= max_F

    amp = np.sqrt(I**2 + Q**2)

    # umbral = 1 dB
    amp_ref = amp[-1]
    amp_min = amp_ref * 10 ** (-db_threshold / 20)

    idx = Fi - 1
    while idx > 0 and amp[idx] > amp_min:
        idx -= 1

    # tramo válido para padding
    I_tail = I[idx:Fi]
    Q_tail = Q[idx:Fi]
    f_tail = f[idx:Fi]

    pad_len = max_F - Fi

    if pad_len > 0:
        # repetir el tramo final
        rep = int(np.ceil(pad_len / len(I_tail)))

        I_pad_tail = np.tile(I_tail, rep)[:pad_len]
        Q_pad_tail = np.tile(Q_tail, rep)[:pad_len]

        if noise_std > 0:
            I_pad_tail += np.random.normal(0, noise_std, pad_len)
            Q_pad_tail += np.random.normal(0, noise_std, pad_len)

        # freq extrapolada
        df = f[-1] - f[-2] if Fi > 1 else 1.0
        f_pad_tail = f[-1] + df * np.arange(1, pad_len + 1)

        I_out = np.concatenate([I, I_pad_tail])
        Q_out = np.concatenate([Q, Q_pad_tail])
        f_out = np.concatenate([f, f_pad_tail])

    else:
        I_out, Q_out, f_out = I, Q, f

    # Mask
    mask = np.zeros(max_F, dtype=np.float32)
    mask[:Fi] = 1.0

    return f_out, I_out, Q_out, mask


def padder_half_kappa_window(
    f: np.ndarray,
    I: np.ndarray,
    Q: np.ndarray,
    baseline_mode: str = "tail_median",
    tail_frac: float = 0.15,
    extra_margin_frac: float = 0.30,
    min_window_pts: int = 128,
):
    """
    Ventana alrededor del dip para fit:
      1) mínimo de |S21|
      2) nivel mitad: A_half = A_min + 0.5*(A_base - A_min)
      3) cruces a izquierda/derecha con A_half
      4) recorte [left, right] con margen extra

    Parámetros
    ----------
    db_floor : si no es None, ignora dips muy poco profundos (ej. 1 dB). En amplitud:
              profundidad_dB = 20*log10(A_base/A_min).
    baseline_mode : "tail_median" o "max"
    tail_frac : fracción final para baseline (si tail_median)
    extra_margin_frac : margen extra respecto al ancho encontrado (0.30 = 30%)
    min_window_pts : tamaño mínimo de ventana, por si el dip es raro

    Returns
    -------
    f_win, I_win, Q_win, mask_win, info
    """
    f = np.asarray(f, dtype=np.float64)
    I = np.asarray(I, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    assert f.ndim == I.ndim == Q.ndim == 1
    assert len(f) == len(I) == len(Q)

    Fi = len(f)
    amp = np.sqrt(I**2 + Q**2)

    # Baseline
    if baseline_mode == "tail_median":  #mediana del tail_frc % final
        n_tail = max(8, int(np.ceil(tail_frac * Fi)))
        A_base = float(np.median(amp[-n_tail:]))
    elif baseline_mode == "max":
        A_base = float(np.max(amp))
    else:
        raise ValueError("baseline_mode debe ser 'tail_median' o 'max'")

    idx_min = int(np.argmin(amp))
    A_min = float(amp[idx_min])

    A_half = A_min + 0.5 * (A_base - A_min)

    # buscar cruce a la izquierda: primer punto (desde el mínimo hacia 0) con amp >= A_half
    left = idx_min
    while left > 0 and amp[left] < A_half:
        left -= 1

    # buscar cruce a la derecha 
    right = idx_min
    while right < Fi - 1 and amp[right] < A_half:
        right += 1

    # fuerza una ventana mínima centrada 
    if left == 0 and amp[left] < A_half:
        left = max(0, idx_min - min_window_pts // 2)
    if right == Fi - 1 and amp[right] < A_half:
        right = min(Fi - 1, idx_min + min_window_pts // 2)

    # margen extra
    width = max(1, right - left)
    extra = int(np.ceil(extra_margin_frac * width))

    left2 = max(0, left - extra)
    right2 = min(Fi - 1, right + extra)

    # asegurar mínimo de puntos
    if (right2 - left2 + 1) < min_window_pts:
        center = idx_min
        half = min_window_pts // 2
        left2 = max(0, center - half)
        right2 = min(Fi - 1, left2 + min_window_pts - 1)
        left2 = max(0, right2 - min_window_pts + 1)

    f_win = f[left2:right2+1]
    I_win = I[left2:right2+1]
    Q_win = Q[left2:right2+1]
    mask_win = np.ones_like(f_win, dtype=np.float32)

    info = dict(
        idx_min=idx_min,
        idx_left=left2,
        idx_right=right2,
        A_half=A_half,
        A_base=A_base,
        A_min=A_min,
        approx_width_hz=float(f[right] - f[left]) if 0 <= left < Fi and 0 <= right < Fi else None
    )
    return f_win, I_win, Q_win, mask_win, info

def _random_poly_response(x: np.ndarray,
                          deg_min: int = 1,
                          deg_max: int = 5,
                          coeff_scale: float = 0.25
                          ) -> np.ndarray:
    
    # Removed ensure_positive. Magnitude always positive.
    
    deg = int(np.random.randint(deg_min, deg_max + 1))
    coef = np.random.normal(0.0, coeff_scale, size=deg + 1)
    coef[0] = 0.0

    a = np.zeros_like(x, dtype=np.float64)
    xp = np.ones_like(x, dtype=np.float64)
    for k in range(deg + 1):
        a += coef[k] * xp
        xp *= x
    
    # In this way, the polynomial is always positive (NEW)
    a = np.exp(a)
    
    # if ensure_positive:
    #     a = a - np.min(a)
    #     a = 0.2 + a
        
    a = a / (np.mean(np.abs(a)) + 1e-12)
    return a


def _apply_random_poly_to_magnitude_only(
    f: np.ndarray,
    s: np.ndarray,
    poly_deg_range: tuple[int, int] = (1, 5),
    poly_coeff_scale: float = 0.25,
) -> np.ndarray:
    
    f = f.astype(np.float64)

    mag = np.abs(s)
    phase = np.unwrap(np.angle(s))  

    # eje normalizado [-1,1] para el polinomio
    f0 = float(np.mean(f))
    span = float(np.ptp(f)) + 1e-12
    x = (f - f0) / (span / 2.0)

    a = _random_poly_response(
        x,
        deg_min=poly_deg_range[0],
        deg_max=poly_deg_range[1],
        coeff_scale=poly_coeff_scale,
    )

    mag2 = mag * a

    return mag2 * np.exp(1j * phase)


def lorentzian_generator(
    n_samples: int,
    cavity_params: dict,
    kc_limits: tuple[float, float],
    frequency_points=(2000, 5000, 6000, 10000, 15000, 20000),
    noise_std_signal: float | tuple[float, float] = 0.0,
    pad_db_threshold: float = 1.0,
    pad_noise_std: float = 1e-4,
):
    """
      - Genera cada muestra con un Fi aleatorio (de frequency_points).
      - Devuelve X_meas y X_clean con DIMENSIÓN FIJA: 2*max_F (padding físico).
      - Devuelve también F (frecuencias padded), F_len y mask.
    """
    frequency_points = np.asarray(frequency_points, dtype=int)
    max_F = int(frequency_points.max())

    kappai_true = np.zeros(n_samples, dtype=np.float32)

    log_lo, log_hi = np.log(kc_limits[0]), np.log(kc_limits[1])
    kc_true = np.exp(np.random.uniform(log_lo, log_hi, size=n_samples)).astype(np.float32)

    X_meas  = np.zeros((n_samples, 2 * max_F), dtype=np.float32)
    X_clean = np.zeros((n_samples, 2 * max_F), dtype=np.float32)

    F = np.zeros((n_samples, max_F), dtype=np.float64)
    F_len = np.zeros(n_samples, dtype=np.int32)
    mask = np.zeros((n_samples, max_F), dtype=np.float32)

    freqs = np.array(frequency_points, dtype=int)
    p = np.array([0.05, 0.05, 0.10, 0.25, 0.25, 0.30], dtype=float)  
    p = p / p.sum()

    for i, kc in enumerate(kc_true):
        Fi = int(np.random.choice(freqs, p=p))
        F_len[i] = Fi
        mask[i, :Fi] = 1.0

        ac = float(np.exp(np.random.uniform(np.log(cavity_params["ac"][0]),
                                            np.log(cavity_params["ac"][1]))))
        dt = float(np.random.uniform(*cavity_params["dt"]))
        fr = float(np.random.uniform(*cavity_params["fr"]))
        dphi = float(np.random.uniform(*cavity_params["dphi"]))

        kappai = float(np.exp(np.random.uniform(np.log(cavity_params["kappai"][0]),
                                                np.log(cavity_params["kappai"][1]))))
        kappai_true[i] = kappai

        phi = float(np.random.uniform(*cavity_params["phi"]))

        kc_f = float(kc)
        kappa = kappai + kc_f
        r = kc_f / kappa

        nRange = np.random.uniform(100, 500)
        delta_f_max = nRange * kappa

        f_i = np.linspace(fr - delta_f_max, fr + delta_f_max, Fi, dtype=np.float64)

        s0 = lorentzian_cy(f_i, ac, dt, phi, r, kappa, dphi, fr)

        s_clean = _apply_random_poly_to_magnitude_only(
            f_i, s0,
            poly_deg_range=(1, 5),
            poly_coeff_scale=np.random.uniform(0.02, 0.06),
        )

        c0 = (np.random.normal(0.0, 0.05) + 1j*np.random.normal(0.0, 0.05))
        s_clean = s_clean + c0

        eps = np.random.uniform(-0.03, 0.03)
        I_clean = s_clean.real * (1 + eps)
        Q_clean = s_clean.imag * (1 - eps)
        s_clean = I_clean + 1j * Q_clean

        s_meas = s_clean.copy()

        if isinstance(noise_std_signal, tuple):
            sig = float(np.random.uniform(noise_std_signal[0], noise_std_signal[1]))
        else:
            sig = float(noise_std_signal)

        if sig > 0.0:
            s_meas = s_meas + (
                np.random.normal(0.0, sig, size=Fi) +
                1j*np.random.normal(0.0, sig, size=Fi)
            )

        f_pad, I_clean_pad, Q_clean_pad, _mask_clean = padder_1db_tail(
            f_i,
            s_clean.real.astype(np.float64),
            s_clean.imag.astype(np.float64),
            max_F=max_F,
            db_threshold=pad_db_threshold,
            noise_std=pad_noise_std,
        ) 

        f_win, I_win, Q_win, mask_win, info = padder_half_kappa_window(
            f_i,
            s_meas.real,
            s_meas.imag,
            baseline_mode="tail_median",
            tail_frac=0.15,
            extra_margin_frac=0.30,
            min_window_pts=256,
        )

        plot_kappa_window_debug(
            f_i,
            s_meas.real,
            s_meas.imag,
            f_win,
            I_win,
            Q_win,
            info,
        )

        f_pad2, I_meas_pad, Q_meas_pad, _mask_meas = padder_1db_tail(
            f_i,
            s_meas.real.astype(np.float64),
            s_meas.imag.astype(np.float64),
            max_F=max_F,
            db_threshold=pad_db_threshold,
            noise_std=pad_noise_std,
        )

        F[i, :] = f_pad

        X_clean[i, :max_F] = I_clean_pad.astype(np.float32)
        X_clean[i, max_F:2*max_F] = Q_clean_pad.astype(np.float32)

        X_meas[i, :max_F] = I_meas_pad.astype(np.float32)
        X_meas[i, max_F:2*max_F] = Q_meas_pad.astype(np.float32)

    return F, X_meas, X_clean, kc_true, kappai_true, F_len, mask


if __name__ == "__main__":
    cavity_params = {
        "ac"     : (0.3, 1.8),
        "dt"     : (-1e-7, 0),
        "phi"    : (-np.pi, np.pi),
        "dphi"   : (-np.pi/4, np.pi/4),
        "kappai" : (1e2, 1e5), 
        "fr"     : (7.30e8 - 2e6, 7.50e8 + 2e6)
    }

    kc_limits = (1e4, 1e5)

    trace = Trace()
    trace.load_trace(source="cab")
    results = trace.do_fit(baseline=(3, 0.7), mode="one-shot", verbose=True)

    F, X_meas, X_clean, kc_true, kappai_true, F_len, mask = lorentzian_generator(
        n_samples=3,
        cavity_params=cavity_params,
        kc_limits=kc_limits,
        frequency_points=[2000, 5000, 6000, 10000, 15000, 20000],
        noise_std_signal=0.0,
    )

    i=2
    max_F = F.shape[1]

    re = X_meas[i, :max_F]
    im = X_meas[i, max_F:2*max_F]
    f_pad = np.arange(max_F)

    mag = np.sqrt(re**2 + im**2)

    plt.figure()
    plt.plot(f_pad, mag)
    plt.axvline(F_len[i], color="r", linestyle="--", label="end real data")
    plt.legend()
    plt.title("Trace with padding (what NN sees)")
    plt.show()

    i = 2
    Fi = F_len[i]       
    f = F[i, :Fi] 
    max_F = F.shape[1]
    re = X_meas[i, :Fi]
    im = X_meas[i, max_F:max_F + Fi]

    mag = np.sqrt(re**2 + im**2)
    phase = np.unwrap(np.arctan2(im, re))

    f_GHz = f * 1e-9

    fig, ax = plt.subplots(2, 1, dpi=300, figsize=(12, 7), constrained_layout=True, sharex=True)
    ax[0].plot(f_GHz, mag, linestyle="--")
    ax[1].plot(f_GHz, phase)
    ax[1].set_xlabel("Frequency [GHz]")
    ax[0].set_ylabel("Amplitude")
    ax[1].set_ylabel("Phase [rad]")
    ax[0].set_title(f"Magnitude (kc = {kc_true[i]:.2e})")
    ax[1].set_title(f"Phase (kc = {kc_true[i]:.2e})")
    ax[0].tick_params(direction='in', which='both')
    ax[1].tick_params(direction='in', which='both')
    plt.show()

    I = X_meas[i, :Fi]
    Q = X_meas[i, max_F:max_F + Fi]

    plt.figure()
    plt.plot(I, Q, label="IQ trajectory")
    plt.scatter(I[0], Q[0], label="Start", zorder=3)
    plt.scatter(I[-1], Q[-1], label="End", zorder=3)
    plt.xlabel("I (Re{S21})")
    plt.ylabel("Q (Im{S21})")
    plt.title(f"IQ plot (kc = {kc_true[i]:.2e})")
    plt.legend()
    plt.axis("equal")
    ax = plt.gca()
    ax.tick_params(direction='in', which='both')
    plt.show()


    

    mag = np.sqrt(I**2 + Q**2)        
    f_GHz = f * 1e-9

    fig, ax = plt.subplots(1, 1, dpi=300, figsize=(9.7, 4.0), constrained_layout=True)

    ax.plot(f_GHz, I, label="Re(S21)")
    ax.plot(f_GHz, Q, label="Im(S21)")
    ax.plot(f_GHz, mag, label="|S21|", linestyle="--")

    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Lorentzian clean (kc = {kc_true[i]:.2e})")
    ax.legend(frameon=False)

    ax.tick_params(direction="in", which="both", top=True, right=True)
    plt.show()

