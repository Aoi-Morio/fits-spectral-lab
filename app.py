"""Run with: python -m streamlit run app.py"""
import hashlib
import json

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.ndimage import gaussian_filter1d

from spectral_lab.analysis import continuum_fit, fit_line, line_candidates
from spectral_lab.io import (
    Spectrum, demo_spectrum, export_csv, export_fits, fits_inventory,
    image_preview, read_csv, read_fits,
)

st.set_page_config(page_title="FITS Spectral Lab", page_icon="🔭", layout="wide")
st.markdown("""<style>
.block-container {padding-top:2rem; max-width:1500px;}
[data-testid="stMetric"] {background:rgba(72,133,165,.08);padding:16px;border-radius:12px;}
.eyebrow {color:#58a6c4;letter-spacing:.2em;font-size:12px;font-weight:700;}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="eyebrow">ASTRONOMICAL SPECTROSCOPY WORKBENCH</div>', unsafe_allow_html=True)
st.title("FITS Spectral Lab")
st.caption("観測ファイルから、スペクトルの形と線の性質を読み解く。データ処理はこのPC上で実行します。")

with st.sidebar:
    st.header("データを開く")
    upload = st.file_uploader("FITS / CSV / TXT", type=["fits", "fit", "fts", "gz", "csv", "txt", "tsv"])
    st.caption("最大100 MB。最初は合成スペクトルのデモを表示します。")

header = None
image = None
try:
    if upload is None:
        spectrum = demo_spectrum()
        dataset_id = "demo"
    else:
        raw = upload.getvalue()
        dataset_id = hashlib.sha256(raw).hexdigest()[:12]
        is_fits = upload.name.lower().endswith((".fits", ".fit", ".fts", ".fits.gz", ".fit.gz", ".fts.gz"))
        if len(raw) > 100 * 1024 * 1024:
            raise ValueError("100 MB以下のファイルを選択してください。")
        if is_fits:
            inventory = fits_inventory(raw)
            available = [item["index"] for item in inventory if item["shape"] is not None]
            if not available:
                raise ValueError("データのあるHDUが見つかりません。")
            with st.sidebar:
                hdu_index = st.selectbox("HDU", available,
                    format_func=lambda i: f'{i}: {inventory[i]["name"]}  {inventory[i]["shape"]}', key=f"hdu_{dataset_id}")
            info = inventory[hdu_index]
            header = info["header"]
            kwargs = {}
            if info["columns"]:
                columns = info["columns"]
                def default_column(choices):
                    return next((i for i, name in enumerate(columns) if name.lower() in choices), 0)
                with st.sidebar:
                    kwargs["wave_column"] = st.selectbox("波長列", columns, index=default_column(["wavelength", "wave", "lambda", "loglam", "pixel"]), key=f"wave_{dataset_id}_{hdu_index}")
                    kwargs["flux_column"] = st.selectbox("フラックス列", columns, index=default_column(["flux", "flam", "intensity"]), key=f"flux_{dataset_id}_{hdu_index}")
                    errors = ["なし"] + columns
                    default_error = next((i for i, name in enumerate(errors) if name.lower() in ["error", "err", "sigma", "ivar"]), 0)
                    error_name = st.selectbox("1σ誤差列 / IVAR", errors, index=default_error, key=f"err_{dataset_id}_{hdu_index}")
                    # Empty string explicitly disables auto-detection.
                    kwargs["error_column"] = error_name if error_name != "なし" else ""
                    unit = st.selectbox("波長列の単位", ["ヘッダーから", "Angstrom", "nm", "um", "Hz", "pixel"], key=f"unit_{dataset_id}_{hdu_index}")
                    if unit != "ヘッダーから":
                        kwargs["wavelength_unit"] = unit
            elif len(info["shape"]) == 2:
                image = image_preview(raw, hdu_index)
                with st.sidebar:
                    axis = st.selectbox("分散方向", [1, 2], format_func=lambda a: "横方向 (FITS axis 1)" if a == 1 else "縦方向 (FITS axis 2)")
                    spatial_size = image.shape[0 if axis == 1 else 1]
                    if spatial_size == 1:
                        aperture = (0, 1)
                    else:
                        aperture = st.slider("加算する空間範囲 [開始, 終了)", 0, spatial_size, (0, spatial_size), key=f"ap_{dataset_id}_{axis}")
                    kwargs.update(dispersion_axis=axis, aperture=aperture)
                    if st.checkbox("背景を差し引く"):
                        kwargs["background"] = st.slider("背景の空間範囲 [開始, 終了)", 0, spatial_size, (0, min(5, spatial_size)), key=f"bg_{dataset_id}_{axis}")
            spectrum = read_fits(raw, hdu_index, **kwargs)
            dataset_id += f"_{hdu_index}"
        else:
            with st.sidebar:
                unit = st.selectbox("第1列の単位", ["Angstrom", "nm", "um", "Hz", "pixel"])
            spectrum = read_csv(raw, unit)
        spectrum.source = f"{upload.name} / {spectrum.source}"
except Exception as exc:
    st.error(f"読み込みできませんでした: {exc}")
    st.info("HDU、波長列、単位、抽出範囲を確認してください。FITSキューブと複数ベクトル行は未対応です。")
    st.stop()

if spectrum.x_unit == "pixel":
    with st.sidebar:
        st.subheader("手動の線形波長校正")
        calibrate = st.checkbox("校正値を適用する")
        if calibrate:
            reference = st.number_input("基準ピクセル（0始まり）", value=0.0)
            wavelength = st.number_input("基準波長 [Å]", value=6500.0)
            increment = st.number_input("波長 / ピクセル [Å]", value=0.1, format="%.6f")
            if increment == 0:
                st.error("波長間隔は0以外にしてください。")
                st.stop()
            spectrum = Spectrum(wavelength + (spectrum.x - reference) * increment,
                                spectrum.flux, spectrum.error, "Angstrom", spectrum.flux_unit,
                                spectrum.source, ["手動の線形波長校正を適用しました。"])

for note in spectrum.notes:
    st.info(note)

st.caption(spectrum.source)
stats = st.columns(4)
stats[0].metric("有効サンプル", f"{spectrum.x.size:,}")
stats[1].metric("軸の範囲", f"{spectrum.x[0]:.3f} – {spectrum.x[-1]:.3f}")
stats[2].metric("軸の単位", "Å" if spectrum.x_unit == "Angstrom" else "pixel")
stats[3].metric("誤差データ", "1σ あり" if spectrum.error is not None else "なし")

view_tab, analysis_tab, export_tab = st.tabs(["スペクトル", "線の解析", "保存・ファイル情報"])

with analysis_tab:
    st.subheader("連続光と解析範囲")
    a, b, c = st.columns(3)
    lo, hi = float(spectrum.x.min()), float(spectrum.x.max())
    span = hi - lo
    default_center = 6562.8 if lo < 6562.8 < hi else (lo + hi) / 2
    low = a.number_input("解析範囲の開始", min_value=lo, max_value=hi,
                         value=max(lo, default_center - span / 15), format="%.6f", key=f"low_{dataset_id}_{spectrum.x_unit}")
    high = b.number_input("解析範囲の終了", min_value=lo, max_value=hi,
                          value=min(hi, default_center + span / 15), format="%.6f", key=f"high_{dataset_id}_{spectrum.x_unit}")
    kind_label = c.selectbox("線の種類", ["輝線", "吸収線"])
    kind = "emission" if kind_label == "輝線" else "absorption"
    degree = st.select_slider("連続光の多項式次数", options=[0, 1, 2, 3, 4, 5], value=1)
    st.caption("指定した線の範囲を除外し、3σの反復クリッピングで全体の連続光を推定します。複雑な連続光では範囲と次数を調整してください。")

try:
    if low >= high:
        raise ValueError("範囲の開始は終了より小さい値にしてください。")
    continuum, continuum_mask = continuum_fit(spectrum, degree=degree, excluded=[(low, high)])
except ValueError as exc:
    st.error(str(exc))
    st.stop()

result = None
fit_x, fit_y = None, None
with analysis_tab:
    st.caption(f"連続光推定に使用: {int(continuum_mask.sum()):,} 点")
    try:
        result, fit_x, fit_y = fit_line(spectrum, continuum, low, high, kind)
        metrics = st.columns(4)
        metrics[0].metric("中心", f"{result.center:.5f}", f"± {result.center_error:.5f}" if result.center_error is not None else None, delta_color="off")
        metrics[1].metric("FWHM", f"{result.fwhm:.5f}")
        metrics[2].metric("ガウス積分強度", f"{result.gaussian_flux:.5g}")
        metrics[3].metric("等価幅", f"{result.equivalent_width:.5f}" if result.equivalent_width is not None else "—")
        st.caption(f"中心・FWHM・等価幅の単位: {spectrum.x_unit} / 積分強度: {spectrum.flux_unit} × {spectrum.x_unit}。等価幅は吸収で正、輝線で負。")
        if spectrum.error is None:
            st.info("入力誤差がないため、フィット誤差は残差からの推定です。")
        for note in result.notes:
            st.caption(note)
        line_plot = go.Figure()
        selected = (spectrum.x >= low) & (spectrum.x <= high)
        line_plot.add_trace(go.Scatter(x=spectrum.x[selected], y=spectrum.flux[selected], name="観測", mode="markers", marker=dict(size=4), error_y=dict(type="data", array=spectrum.error[selected], visible=True) if spectrum.error is not None else None))
        line_plot.add_trace(go.Scatter(x=fit_x, y=fit_y, name="ガウスフィット", line=dict(color="#f69d50")))
        line_plot.add_trace(go.Scatter(x=spectrum.x[selected], y=continuum[selected], name="連続光", line=dict(dash="dash", color="#50b9ab")))
        line_plot.update_layout(height=380, xaxis_title=spectrum.x_unit, yaxis_title=spectrum.flux_unit, margin=dict(l=30, r=20, t=20, b=30), legend=dict(orientation="h"))
        st.plotly_chart(line_plot, use_container_width=True)
        with st.expander("測定値の詳細"):
            st.json(result.to_dict())
        if spectrum.x_unit == "Angstrom":
            rest = st.number_input("比較する静止波長 [Å]", min_value=0.000001, value=6562.8, format="%.6f")
            st.metric("赤方偏移 z = λ / λ₀ − 1", f"{result.center / rest - 1:.7f}")
            st.caption("同じ空気中／真空波長系の静止波長を入力してください。赤方偏移には観測者運動の補正を適用していません。")
    except (ValueError, RuntimeError) as exc:
        st.warning(f"この範囲はフィットできません: {exc}")
    with st.expander("線の候補を探す"):
        prominence = st.number_input("ピークの最小突出度（フラックス単位）", min_value=0.000001, value=max(0.000001, float(np.std(spectrum.flux - continuum))), format="%.6f")
        candidates = line_candidates(spectrum, continuum, prominence, kind)
        if candidates:
            st.dataframe(candidates, use_container_width=True)
        else:
            st.caption("この閾値の候補はありません。")
        st.caption("候補は画素上の極値です。線の同定や有意性の判定には範囲を指定してフィットしてください。")

with view_tab:
    smoothing = st.slider("表示の平滑化 σ（サンプル）", 0.0, 10.0, 0.0, 0.5)
    display_flux = gaussian_filter1d(spectrum.flux, smoothing) if smoothing > 0 else spectrum.flux
    plot = go.Figure()
    plot.add_trace(go.Scattergl(x=spectrum.x, y=display_flux, name="スペクトル", line=dict(color="#55a9e8", width=1.4)))
    plot.add_trace(go.Scattergl(x=spectrum.x, y=continuum, name="連続光", line=dict(color="#50b9ab", dash="dash")))
    plot.add_vrect(x0=low, x1=high, fillcolor="#e7a75a", opacity=0.12, line_width=0)
    if result is not None:
        plot.add_trace(go.Scatter(x=fit_x, y=fit_y, name="線のフィット", line=dict(color="#f69d50", width=2)))
    plot.update_layout(height=470, xaxis_title=f"波長 [{spectrum.x_unit}]" if spectrum.x_unit != "pixel" else "ピクセル",
                       yaxis_title=f"フラックス [{spectrum.flux_unit}]", hovermode="x unified", legend=dict(orientation="h"), margin=dict(l=30, r=20, t=25, b=40))
    st.plotly_chart(plot, use_container_width=True)
    st.caption("ドラッグで拡大、ダブルクリックでリセット。平滑化は表示だけに適用し、解析には元データを使用します。")
    if image is not None:
        with st.expander("2次元画像と抽出範囲", expanded=True):
            row_step, col_step = max(1, image.shape[0] // 500), max(1, image.shape[1] // 1000)
            shown = image[::row_step, ::col_step]
            finite = shown[np.isfinite(shown)]
            zmin, zmax = np.percentile(finite, [5, 99]) if finite.size else (0, 1)
            image_plot = go.Figure(go.Heatmap(z=shown, x=np.arange(0, image.shape[1], col_step), y=np.arange(0, image.shape[0], row_step), zmin=zmin, zmax=zmax, colorscale="Viridis"))
            aperture = kwargs["aperture"]
            if kwargs["dispersion_axis"] == 1:
                image_plot.add_hrect(y0=aperture[0] - 0.5, y1=aperture[1] - 0.5, line_color="#f69d50", fillcolor="#f69d50", opacity=0.2)
            else:
                image_plot.add_vrect(x0=aperture[0] - 0.5, x1=aperture[1] - 0.5, line_color="#f69d50", fillcolor="#f69d50", opacity=0.2)
            image_plot.update_layout(height=350, xaxis_title="画像のXピクセル", yaxis_title="画像のYピクセル")
            st.plotly_chart(image_plot, use_container_width=True)
            st.caption("色の範囲は5〜99パーセンタイル。大きい画像は表示時のみ間引きます。")

with export_tab:
    st.subheader("解析結果をダウンロード")
    left, middle, right = st.columns(3)
    left.download_button("スペクトル CSV", export_csv(spectrum, continuum), "spectrum.csv", "text/csv")
    middle.download_button("スペクトル FITS", export_fits(spectrum, continuum), "spectrum.fits", "application/octet-stream")
    if result is not None:
        report = dict(source=spectrum.source, axis_unit=spectrum.x_unit, flux_unit=spectrum.flux_unit,
                      fit_window=[low, high], line_kind=kind, continuum_degree=degree,
                      continuum_method="polynomial; 3-sigma MAD clipping; fit window excluded",
                      measurements=result.to_dict())
        right.download_button("線の測定 JSON", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), "line_measurement.json", "application/json")
    st.download_button("練習用デモ FITS", export_fits(demo_spectrum()), "demo_halpha.fits", "application/octet-stream")
    if header:
        with st.expander("FITSヘッダー"):
            st.code(header, language="text")
    st.caption("FITS保存は抽出した1次元スペクトルの新しいテーブルです。元ファイルの全HDUや装置固有ヘッダーは複製しません。")
    st.caption("v0.1.0 · 対応: 1D/2D FITS、波長＋フラックスのFITSテーブル、CSV/TXT。装置補正・キューブ・重なった線の同時フィットは未対応です。")
