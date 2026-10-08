"""Read explicitly supported spectra without inventing wavelength calibration."""
from dataclasses import dataclass, field
from io import BytesIO, StringIO
import csv

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS


@dataclass
class Spectrum:
    x: np.ndarray
    flux: np.ndarray
    error: np.ndarray | None = None
    x_unit: str = "pixel"
    flux_unit: str = "arbitrary"
    source: str = ""
    notes: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        self.flux = np.asarray(self.flux, dtype=float)
        if self.x.ndim != 1 or self.x.shape != self.flux.shape:
            raise ValueError("軸とフラックスには同じ長さの1次元配列が必要です。")
        good = np.isfinite(self.x) & np.isfinite(self.flux)
        if self.error is not None:
            self.error = np.asarray(self.error, dtype=float)
            if self.error.shape != self.x.shape:
                raise ValueError("誤差配列の長さが一致しません。")
            good &= np.isfinite(self.error) & (self.error > 0)
        removed = int((~good).sum())
        self.x, self.flux = self.x[good], self.flux[good]
        if self.error is not None:
            self.error = self.error[good]
        if self.x.size < 8:
            raise ValueError("有効なサンプルが8点以上必要です。")
        order = np.argsort(self.x)
        self.x, self.flux = self.x[order], self.flux[order]
        if self.error is not None:
            self.error = self.error[order]
        if np.any(np.diff(self.x) <= 0):
            raise ValueError("軸に重複した値があります。単一のスペクトルを選択してください。")
        if removed:
            self.notes.append(f"非有限値または無効な誤差のある {removed} 点を除外しました。")


def fits_inventory(raw: bytes) -> list[dict]:
    with fits.open(BytesIO(raw), memmap=False) as hdus:
        return [dict(index=i, name=h.name, shape=getattr(h.data, "shape", None),
                     columns=list(h.columns.names) if isinstance(h, (fits.BinTableHDU, fits.TableHDU)) else [],
                     header=str(h.header)) for i, h in enumerate(hdus)]


def image_preview(raw: bytes, hdu_index: int) -> np.ndarray:
    with fits.open(BytesIO(raw), memmap=False) as hdus:
        data = np.asarray(hdus[hdu_index].data, dtype=float)
        if data.ndim != 2:
            raise ValueError("2次元画像を選択してください。")
        return data.copy()


def _wavelength(values, unit):
    """Normalize spectral wavelength/frequency coordinates to Angstrom."""
    if not unit:
        raise ValueError("波長列の単位を指定してください。")
    try:
        return (np.asarray(values) * u.Unit(unit)).to_value(u.AA, equivalencies=u.spectral())
    except (ValueError, u.UnitConversionError) as exc:
        raise ValueError("波長または周波数の単位が必要です（例: Angstrom, nm, um, Hz）。") from exc


def image_axis(header, count, fits_axis=1):
    pixels = np.arange(count, dtype=float)
    # SDSS legacy 1D products encode log10(wavelength/Angstrom).
    if fits_axis == 1 and "COEFF0" in header and "COEFF1" in header:
        return 10 ** (header["COEFF0"] + pixels * header["COEFF1"]), "Angstrom", []
    ctype = str(header.get(f"CTYPE{fits_axis}", "")).upper()
    unit = header.get(f"CUNIT{fits_axis}", "")
    has_reference = f"CRVAL{fits_axis}" in header and f"CRPIX{fits_axis}" in header
    spectral_type = ctype.startswith(("WAVE", "AWAV", "FREQ", "ENER", "WAVN"))
    if spectral_type and has_reference:
        try:
            wcs = WCS(header)
            world_index = next(i for i, name in enumerate(wcs.wcs.ctype)
                               if str(name).upper().startswith(("WAVE", "AWAV", "FREQ", "ENER", "WAVN")))
            # Do not silently select a different spectral axis or coupled spatial WCS.
            if world_index != fits_axis - 1:
                raise ValueError("選択した分散軸とWCSの分光軸が一致しません。")
            if np.count_nonzero(wcs.axis_correlation_matrix[world_index]) != 1:
                raise ValueError("空間軸と結合したWCSには対応していません。")
            coords = [np.full(count, val - 1) for val in wcs.wcs.crpix]
            coords[fits_axis - 1] = pixels
            world = wcs.all_pix2world(*coords, 0)
            x = _wavelength(world[world_index], wcs.world_axis_units[world_index])
            note = ["AWAVは空気中波長です。真空波長への変換は行っていません。"] if ctype.startswith("AWAV") else []
            return x, "Angstrom", note
        except Exception as exc:
            raise ValueError(f"分光WCSを解釈できません: {exc}") from exc
    # Some simple spectra provide linear calibration with CUNIT but no CTYPE.
    step_key = f"CDELT{fits_axis}"
    cd_key = f"CD{fits_axis}_{fits_axis}"
    if not ctype and unit and has_reference and (step_key in header or cd_key in header):
        if int(header.get("NAXIS", 1)) > 1 and any(
            key.startswith((f"CD{fits_axis}_", f"PC{fits_axis}_")) and
            key not in (cd_key, f"PC{fits_axis}_{fits_axis}") and float(header[key]) != 0
            for key in header.keys()
        ):
            raise ValueError("結合したWCSには対応していません。")
        step = header.get(cd_key, header.get(step_key, 1) * header.get(f"PC{fits_axis}_{fits_axis}", 1))
        x = header[f"CRVAL{fits_axis}"] + (pixels + 1 - header[f"CRPIX{fits_axis}"]) * step
        return _wavelength(x, unit), "Angstrom", ["CTYPEなし: 単位付きの線形校正を使用しました。"]
    return pixels, "pixel", ["波長校正が見つかりません。ピクセル軸を使用しています。"]


def read_fits(raw: bytes, hdu_index=0, *, dispersion_axis=1,
              aperture=None, background=None, wave_column=None, flux_column=None,
              error_column=None, wavelength_unit=None) -> Spectrum:
    with fits.open(BytesIO(raw), memmap=False) as hdus:
        hdu = hdus[hdu_index]
        if hdu.data is None:
            raise ValueError("このHDUにはデータがありません。")
        if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
            names = {name.lower(): name for name in hdu.columns.names}
            wave_column = wave_column or next((names[k] for k in ("wavelength", "wave", "lambda", "loglam", "pixel") if k in names), None)
            flux_column = flux_column or next((names[k] for k in ("flux", "flam", "intensity") if k in names), None)
            if not wave_column or not flux_column:
                raise ValueError("波長列とフラックス列を選択してください。")
            x, flux = np.asarray(hdu.data[wave_column]), np.asarray(hdu.data[flux_column])
            if x.ndim > 1:
                if len(x) != 1:
                    raise ValueError("複数のベクトル行には未対応です。1行のスペクトルに分割してください。")
                x = x[0]
            if flux.ndim > 1:
                if len(flux) != 1:
                    raise ValueError("複数のベクトル行には未対応です。")
                flux = flux[0]
            loglam = wave_column.lower() == "loglam"
            if loglam:
                x = 10 ** x
            unit = wavelength_unit or ("pixel" if wave_column.lower() == "pixel" else "Angstrom" if loglam else hdu.columns[wave_column].unit)
            if unit != "pixel":
                x = _wavelength(x, unit)
            if error_column is None:
                error_column = next((names[k] for k in ("error", "err", "sigma", "ivar") if k in names), None)
            error = None
            if error_column:
                error = np.asarray(hdu.data[error_column], dtype=float)
                if error.ndim > 1:
                    if len(error) != 1:
                        raise ValueError("複数の誤差ベクトル行には未対応です。")
                    error = error[0]
                if error_column.lower() == "ivar":
                    with np.errstate(divide="ignore", invalid="ignore"):
                        error = np.where(error > 0, 1 / np.sqrt(error), np.nan)
            return Spectrum(x, flux, error, "pixel" if unit == "pixel" else "Angstrom", hdu.columns[flux_column].unit or "arbitrary",
                            f"HDU {hdu_index}: {hdu.name}")
        data = np.asarray(hdu.data, dtype=float)
        notes = []
        if data.ndim == 1:
            flux = data
            dispersion_axis = 1
        elif data.ndim == 2:
            if dispersion_axis not in (1, 2):
                raise ValueError("FITS分散軸は1または2です。")
            image = data if dispersion_axis == 1 else data.T
            start, stop = aperture or (0, image.shape[0])
            if not 0 <= start < stop <= image.shape[0]:
                raise ValueError("抽出範囲が画像の範囲外です。")
            flux = np.sum(image[start:stop], axis=0)
            if background:
                bstart, bstop = background
                if not 0 <= bstart < bstop <= image.shape[0] or max(start, bstart) < min(stop, bstop):
                    raise ValueError("背景範囲は抽出範囲と重ならない範囲を指定してください。")
                flux = flux - (stop - start) * np.median(image[bstart:bstop], axis=0)
                notes.append("背景の行方向中央値を引きました。抽出誤差は未推定です。")
            notes.append(f"空間方向 [{start}, {stop}) を加算しました。FITS分散軸: {dispersion_axis}")
        else:
            raise ValueError("画像HDUは1次元・2次元に対応しています。キューブは未対応です。")
        x, unit, axis_notes = image_axis(hdu.header, flux.size, dispersion_axis)
        return Spectrum(x, flux, None, unit, hdu.header.get("BUNIT", "arbitrary"),
                        f"HDU {hdu_index}: {hdu.name}", notes + axis_notes)


def read_csv(raw: bytes, wavelength_unit="Angstrom") -> Spectrum:
    try:
        text = raw.decode("utf-8-sig")
        lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
        dialect = csv.Sniffer().sniff("\n".join(lines[:10]), delimiters=",;\t ")
        rows = list(csv.reader(lines, dialect))
        try:
            float(rows[0][0])
        except ValueError:
            rows = rows[1:]
        data = np.asarray([[float(v) for v in row if v.strip()] for row in rows], dtype=float)
        if data.ndim != 2 or data.shape[1] not in (2, 3):
            raise ValueError("列数")
    except (ValueError, IndexError, csv.Error, UnicodeError) as exc:
        raise ValueError("CSV/TXTは波長、フラックス、任意の1σ誤差の2〜3列にしてください。") from exc
    x = data[:, 0] if wavelength_unit == "pixel" else _wavelength(data[:, 0], wavelength_unit)
    return Spectrum(x, data[:, 1], data[:, 2] if data.shape[1] == 3 else None,
                    "pixel" if wavelength_unit == "pixel" else "Angstrom", source="CSV/TXT")


def export_csv(spectrum: Spectrum, continuum=None) -> str:
    buffer = StringIO()
    names = [f"wavelength_{spectrum.x_unit}", "flux"]
    columns = [spectrum.x, spectrum.flux]
    if spectrum.error is not None:
        names.append("error")
        columns.append(spectrum.error)
    if continuum is not None:
        names.extend(["continuum", "flux_minus_continuum"])
        columns.extend([continuum, spectrum.flux - continuum])
    np.savetxt(buffer, np.column_stack(columns), delimiter=",", header=",".join(names), comments="")
    return buffer.getvalue()


def export_fits(spectrum: Spectrum, continuum=None) -> bytes:
    columns = [fits.Column(name="WAVELENGTH" if spectrum.x_unit != "pixel" else "PIXEL",
                           format="D", unit=None if spectrum.x_unit == "pixel" else "Angstrom", array=spectrum.x),
               fits.Column(name="FLUX", format="D", array=spectrum.flux,
                           unit=None if spectrum.flux_unit == "arbitrary" else spectrum.flux_unit)]
    if spectrum.error is not None:
        columns.append(fits.Column(name="ERROR", format="D", array=spectrum.error))
    if continuum is not None:
        columns.append(fits.Column(name="CONTINUUM", format="D", array=continuum))
    buffer = BytesIO()
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns, name="SPECTRUM")]).writeto(buffer)
    return buffer.getvalue()


def demo_spectrum() -> Spectrum:
    rng = np.random.default_rng(42)
    x = np.linspace(6500, 6620, 1201)
    y = 10 + 0.008 * (x - 6560)
    for amplitude, center, sigma in [(24, 6562.8, 1.7), (8, 6548.1, 1.1), (12, 6583.4, 1.3)]:
        y += amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    error = np.full(x.size, 0.35)
    return Spectrum(x, y + rng.normal(0, error), error, "Angstrom", "arbitrary", "合成デモ / Hα付近")
