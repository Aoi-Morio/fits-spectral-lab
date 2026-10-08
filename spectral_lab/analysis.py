"""Continuum and isolated Gaussian line measurement on original samples."""
from dataclasses import dataclass, asdict
import warnings

import numpy as np
from numpy.polynomial import Polynomial
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.signal import find_peaks

from .io import Spectrum


def continuum_fit(spectrum: Spectrum, degree=1, excluded=(), clip_sigma=3.0):
    if not 0 <= degree <= 5:
        raise ValueError("連続光の次数は0〜5を指定してください。")
    x, y = spectrum.x, spectrum.flux
    mask = np.ones(x.size, dtype=bool)
    for low, high in excluded:
        mask &= ~((x >= low) & (x <= high))
    for _ in range(8):
        if mask.sum() < max(degree + 2, 8):
            raise ValueError("連続光の推定に使える点が不足しています。")
        weights = 1 / spectrum.error[mask] if spectrum.error is not None else None
        model = Polynomial.fit(x[mask], y[mask], degree, w=weights)
        residual = y - model(x)
        center = np.median(residual[mask])
        scale = 1.4826 * np.median(np.abs(residual[mask] - center))
        if scale <= np.finfo(float).eps * max(np.finfo(float).tiny, np.max(np.abs(y))):
            break
        updated = mask & (np.abs(residual - center) <= clip_sigma * scale)
        if np.array_equal(updated, mask):
            break
        mask = updated
    if mask.sum() < max(degree + 2, 8):
        raise ValueError("クリッピング後の連続光データが不足しています。")
    weights = 1 / spectrum.error[mask] if spectrum.error is not None else None
    model = Polynomial.fit(x[mask], y[mask], degree, w=weights)
    return model(x), mask


@dataclass
class LineResult:
    center: float
    center_error: float | None
    amplitude: float
    sigma: float
    fwhm: float
    fwhm_error: float | None
    gaussian_flux: float
    gaussian_flux_error: float | None
    window_flux: float
    equivalent_width: float | None
    reduced_chi2: float | None
    samples: int
    uncertainty_method: str
    notes: list[str]

    def to_dict(self):
        return asdict(self)


def fit_line(spectrum: Spectrum, continuum, low, high, kind="emission"):
    if kind not in ("emission", "absorption") or low >= high:
        raise ValueError("線の種類と解析範囲を確認してください。")
    continuum = np.asarray(continuum, dtype=float)
    if continuum.shape != spectrum.flux.shape or not np.all(np.isfinite(continuum)):
        raise ValueError("連続光の配列が無効です。")
    selected = (spectrum.x >= low) & (spectrum.x <= high)
    x, y = spectrum.x[selected], (spectrum.flux - continuum)[selected]
    if x.size < 8:
        raise ValueError("解析範囲に8点以上必要です。")
    origin, span = float(np.mean(x)), float(np.ptp(x))
    t = (x - origin) / span
    yscale = float(np.max(np.abs(y)))
    if yscale <= np.finfo(float).tiny:
        raise ValueError("解析範囲に線の信号がありません。")
    scaled_y = y / yscale
    sign = 1 if kind == "emission" else -1
    peak = int(np.argmax(sign * scaled_y))
    if sign * scaled_y[peak] <= 0:
        raise ValueError("指定した種類の線が連続光から検出できません。")
    step = float(np.median(np.diff(x))) / span

    def model(axis, amplitude, center, sigma, offset):
        return amplitude * np.exp(-0.5 * ((axis - center) / sigma) ** 2) + offset

    lower_amplitude, upper_amplitude = (0, np.inf) if sign == 1 else (-np.inf, 0)
    initial = [scaled_y[peak], t[peak], max(step, 0.08), 0]
    error = spectrum.error[selected] / yscale if spectrum.error is not None else None
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", OptimizeWarning)
        params, covariance = curve_fit(
            model, t, scaled_y, p0=initial,
            bounds=([lower_amplitude, t.min(), step / 4, -np.inf],
                    [upper_amplitude, t.max(), 1.0, np.inf]),
            sigma=error, absolute_sigma=error is not None, maxfev=20000, x_scale="jac")
    amp, center, width, offset = params
    notes = ["単一ガウス＋定数残差を仮定。連続光推定の不確かさは含みません。"]
    if captured or not np.all(np.isfinite(covariance)):
        notes.append("共分散が不安定です。誤差値を解釈できない可能性があります。")
    if width * span < 2 * step * span:
        notes.append("線幅が2サンプル未満です。分解能不足の可能性があります。")
    if center - 3 * width < t.min() or center + 3 * width > t.max():
        notes.append("線の裾が解析範囲の外に出ます。範囲を広げて確認してください。")
    prediction = model(t, *params) * yscale + continuum[selected]
    factor = 2 * np.sqrt(2 * np.log(2))
    root_tau = np.sqrt(2 * np.pi)
    gradient = np.array([width, 0, amp, 0]) * root_tau * span * yscale
    flux_variance = float(gradient @ covariance @ gradient)

    def safe_error(variance):
        return float(np.sqrt(variance)) if np.isfinite(variance) and variance >= 0 else None

    local_continuum = continuum[selected]
    ew = float(trapezoid(1 - spectrum.flux[selected] / local_continuum, x)) if np.all(local_continuum > 0) else None
    if ew is None:
        notes.append("連続光に0以下の値があるため等価幅は計算しません。")
    chi2 = float(np.sum(((spectrum.flux[selected] - prediction) / spectrum.error[selected]) ** 2) / (len(x) - 4)) if error is not None else None
    result = LineResult(
        center=float(origin + center * span), center_error=safe_error(covariance[1, 1] * span**2),
        amplitude=float(amp * yscale), sigma=float(width * span),
        fwhm=float(factor * width * span), fwhm_error=safe_error(covariance[2, 2] * (factor * span)**2),
        gaussian_flux=float(amp * width * root_tau * span * yscale), gaussian_flux_error=safe_error(flux_variance),
        window_flux=float(trapezoid(y, x)), equivalent_width=ew, reduced_chi2=chi2,
        samples=int(len(x)), uncertainty_method="input_1sigma" if error is not None else "residual_estimate",
        notes=notes)
    return result, x, prediction


def line_candidates(spectrum: Spectrum, continuum, prominence=1.0, kind="emission"):
    residual = spectrum.flux - continuum
    sign = 1 if kind == "emission" else -1
    peaks, properties = find_peaks(sign * residual, prominence=prominence, distance=3)
    order = np.argsort(properties["prominences"])[::-1][:30]
    return [dict(position=float(spectrum.x[peaks[i]]),
                 flux=float(spectrum.flux[peaks[i]]), prominence=float(properties["prominences"][i])) for i in order]
