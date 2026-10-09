"""Run with: python -m streamlit run app.py"""
import hashlib
from html import escape
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
from spectral_lab.workbench import apply_style, chart_style, document_strip, panel_heading, status_bar

st.set_page_config(page_title="FITS Spectral Lab", page_icon="🔭", layout="wide")
apply_style()
st.title("FITS Spectral Lab")
st.session_state.setdefault("workspace", "スペクトル")
st.session_state.setdefault("document_epoch", 0)
st.session_state.setdefault("view_epoch", 0)


def change_workspace(view):
    st.session_state.workspace = view


def reset_document():
    st.session_state.document_epoch += 1
    st.session_state.workspace = "スペクトル"


def reset_view():
    st.session_state.view_epoch += 1


with st.container(key="menubar"):
    menu = st.columns([1, 1, 1, 1, 1, 6], gap="small")
    with menu[0].popover("ファイル"):
        st.button("デモを開く", on_click=reset_document)
        st.download_button("デモFITSを保存", export_fits(demo_spectrum()), "demo_halpha.fits", "application/octet-stream", key="menu_demo")
        st.caption("観測ファイルは右の「ファイルを開く」から読み込めます。")
    with menu[1].popover("表示"):
        grid = st.checkbox("グリッドを表示", value=True, key="grid")
        drag = st.radio("ドラッグ操作", ["ズーム", "移動"], horizontal=True, key="drag")
        st.button("表示範囲をリセット", on_click=reset_view)
    menu[2].button("解析", on_click=change_workspace, args=("線の解析",))
    menu[3].button("保存", on_click=change_workspace, args=("保存・ファイル情報",))
    with menu[4].popover("ヘルプ"):
        st.markdown("右側で範囲と線の種類を指定し、中央でフィットを確認します。左のツールで画面を切り替えます。")
        st.caption("ドラッグでズーム、ダブルクリックで全体表示。平滑化とレイヤー切替は表示だけに適用します。")
    menu[5].markdown('<div class="workspace-label">ワークスペース：分光解析</div>', unsafe_allow_html=True)

rail_column, canvas_column, inspector_column = st.columns([0.45, 7.2, 2.8], gap="small")
with rail_column.container(key="tool_rail"):
    st.markdown('<div class="rail-brand">Sl</div>', unsafe_allow_html=True)
    for name, icon in [("スペクトル", "show_chart"), ("線の解析", "analytics"),
                       ("2D画像", "image"), ("保存・ファイル情報", "save")]:
        st.button(name, icon=f":material/{icon}:", help=name, key=f"tool_{name}",
                  type="primary" if st.session_state.workspace == name else "secondary",
                  width="stretch", on_click=change_workspace, args=(name,))
    st.markdown('<div class="rail-divider"></div>', unsafe_allow_html=True)
    st.button("全体表示", icon=":material/fit_screen:", help="表示範囲をリセット", key="tool_reset",
              width="stretch", on_click=reset_view)

canvas = canvas_column.container(key="canvas")
inspector = inspector_column.container(key="inspector")
with inspector:
    properties, layers, information = st.tabs(["プロパティ", "レイヤー", "情報"])
with properties:
    document_info = st.empty()
    with st.expander("ファイルを開く"):
        upload = st.file_uploader("FITS / CSV / TXT", type=["fits", "fit", "fts", "gz", "csv", "txt", "tsv"], key=f"upload_{st.session_state.document_epoch}")
        st.caption("最大100 MB · このPC上で処理")
with layers:
    panel_heading("表示するレイヤー")
    show_spectrum = st.checkbox("観測スペクトル", value=True, key="layer_spectrum")
    show_continuum = st.checkbox("連続光", value=True, key="layer_continuum")
    show_fit = st.checkbox("ガウスフィット", value=True, key="layer_fit")
    show_errors = st.checkbox("1σ誤差", value=False, key="layer_errors")
    show_selection = st.checkbox("解析範囲", value=True, key="layer_selection")
    st.caption("表示だけを切り替えます。測定値には影響しません。")

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
            with properties:
                hdu_index = st.selectbox("HDU", available,
                    format_func=lambda i: f'{i}: {inventory[i]["name"]}  {inventory[i]["shape"]}', key=f"hdu_{dataset_id}")
            info = inventory[hdu_index]
            header = info["header"]
            kwargs = {}
            if info["columns"]:
                columns = info["columns"]
                def default_column(choices):
                    return next((i for i, name in enumerate(columns) if name.lower() in choices), 0)
                with properties:
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
                with properties:
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
            with properties:
                unit = st.selectbox("第1列の単位", ["Angstrom", "nm", "um", "Hz", "pixel"])
            spectrum = read_csv(raw, unit)
        spectrum.source = f"{upload.name} / {spectrum.source}"
except Exception as exc:
    with canvas:
        st.error(f"読み込みできませんでした: {exc}")
        st.info("HDU、波長列、単位、抽出範囲を確認してください。FITSキューブと複数ベクトル行は未対応です。")
    st.stop()

if spectrum.x_unit == "pixel":
    with properties:
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

axis_label = "Å" if spectrum.x_unit == "Angstrom" else "pixel"
document_name = upload.name if upload is not None else "demo_halpha.fits"
document_info.markdown(f'<div class="document-info">{escape(document_name)}</div>', unsafe_allow_html=True)

with properties:
    panel_heading("線のプロパティ")
    lo, hi = float(spectrum.x.min()), float(spectrum.x.max())
    span = hi - lo
    default_center = 6562.8 if lo < 6562.8 < hi else (lo + hi) / 2
    a, b = st.columns(2, gap="small")
    low = a.number_input("解析範囲の開始", min_value=lo, max_value=hi,
                         value=max(lo, default_center - span / 15), format="%.3f", key=f"low_{dataset_id}_{spectrum.x_unit}")
    high = b.number_input("解析範囲の終了", min_value=lo, max_value=hi,
                          value=min(hi, default_center + span / 15), format="%.3f", key=f"high_{dataset_id}_{spectrum.x_unit}")
    kind_label = st.selectbox("線の種類", ["輝線", "吸収線"])
    kind = "emission" if kind_label == "輝線" else "absorption"
    degree = st.select_slider("連続光の次数", options=[0, 1, 2, 3, 4, 5], value=1,
                              help="解析範囲を除外し、3σクリッピングで全体の多項式連続光を推定します。")
    smoothing = st.slider("表示の平滑化 σ", 0.0, 10.0, 0.0, 0.5, help="表示だけを平滑化します。解析には元サンプルを使います。")

with information:
    panel_heading("ドキュメント情報")
    st.caption(spectrum.source)
    st.dataframe([dict(項目="サンプル", 値=f"{spectrum.x.size:,}"),
                  dict(項目="範囲", 値=f"{lo:.3f} – {hi:.3f} {axis_label}"),
                  dict(項目="フラックス単位", 値=spectrum.flux_unit),
                  dict(項目="入力誤差", 値="1σあり" if spectrum.error is not None else "なし")], hide_index=True, width="stretch")
    for note in spectrum.notes:
        st.info(note)
    if header:
        with st.expander("FITSヘッダー"):
            st.code(header, language="text")

with canvas:
    document_strip(document_name, st.session_state.workspace, axis_label)

try:
    if low >= high:
        raise ValueError("範囲の開始は終了より小さい値にしてください。")
    continuum, continuum_mask = continuum_fit(spectrum, degree=degree, excluded=[(low, high)])
except ValueError as exc:
    with canvas:
        st.error(str(exc))
    st.stop()

result, fit_x, fit_y = None, None, None
fit_error = None
try:
    result, fit_x, fit_y = fit_line(spectrum, continuum, low, high, kind)
except (ValueError, RuntimeError) as exc:
    fit_error = str(exc)

with properties:
    panel_heading("測定値")
    if result is not None:
        m1, m2 = st.columns(2, gap="small")
        m1.metric("中心", f"{result.center:.4f}", f"± {result.center_error:.4f}" if result.center_error is not None else None, delta_color="off")
        m2.metric("FWHM", f"{result.fwhm:.4f}")
        m3, m4 = st.columns(2, gap="small")
        m3.metric("ガウス積分強度", f"{result.gaussian_flux:.4g}")
        m4.metric("等価幅", f"{result.equivalent_width:.4f}" if result.equivalent_width is not None else "—")
        st.caption(f"中心・幅・等価幅: {axis_label} · 吸収の等価幅は正")
        if spectrum.x_unit == "Angstrom":
            with st.expander("赤方偏移"):
                rest = st.number_input("比較する静止波長 [Å]", min_value=0.000001, value=6562.8, format="%.6f")
                st.metric("赤方偏移 z = λ / λ₀ − 1", f"{result.center / rest - 1:.7f}")
                st.caption("同じ空気中／真空波長系の静止波長を入力してください。観測者運動の補正は未適用です。")
    else:
        st.warning(f"この範囲はフィットできません: {fit_error}")

with information:
    panel_heading("解析情報")
    st.caption(f"連続光推定に使用: {int(continuum_mask.sum()):,} 点")
    if spectrum.error is None:
        st.caption("入力誤差がないため、フィット誤差は残差からの推定です。")
    if result is not None:
        for note in result.notes:
            st.caption(note)
        with st.expander("測定値の詳細"):
            st.json(result.to_dict())

workspace = st.session_state.workspace
revision = f"{dataset_id}_{st.session_state.view_epoch}_{workspace}"
plot_config = dict(displaylogo=False, scrollZoom=True,
                   modeBarButtonsToRemove=["lasso2d", "select2d"], toImageButtonOptions=dict(filename="spectrum"))

with canvas:
    if workspace in ("スペクトル", "線の解析"):
        plot = go.Figure()
        selected = (spectrum.x >= low) & (spectrum.x <= high) if workspace == "線の解析" else np.ones(spectrum.x.size, dtype=bool)
        display_flux = gaussian_filter1d(spectrum.flux, smoothing) if smoothing > 0 else spectrum.flux
        if show_spectrum:
            errors = dict(type="data", array=spectrum.error[selected], visible=True, color="#737373", thickness=0.7) if show_errors and spectrum.error is not None else None
            plot.add_trace(go.Scattergl(x=spectrum.x[selected], y=display_flux[selected], name="観測スペクトル",
                                       mode="lines+markers" if workspace == "線の解析" else "lines",
                                       marker=dict(size=3), line=dict(color="#b7cddd", width=1.2), error_y=errors))
        if show_continuum:
            plot.add_trace(go.Scattergl(x=spectrum.x[selected], y=continuum[selected], name="連続光", line=dict(color="#9db99a", dash="dash", width=1.2)))
        if show_fit and result is not None:
            plot.add_trace(go.Scatter(x=fit_x, y=fit_y, name="ガウスフィット", line=dict(color="#dfac77", width=2)))
        if show_selection:
            plot.add_vrect(x0=low, x1=high, fillcolor="#6d97b8", opacity=0.10, line_width=1, line_color="#7195b1")
        plot.update_layout(xaxis_title=f"波長 [{axis_label}]" if spectrum.x_unit != "pixel" else "ピクセル", yaxis_title=f"フラックス [{spectrum.flux_unit}]")
        chart_style(plot, grid=grid, dragmode="zoom" if drag == "ズーム" else "pan", revision=revision)
        st.plotly_chart(plot, width="stretch", theme=None, config=plot_config)
        st.markdown(f'<div class="canvas-footer">{lo:.3f} — {hi:.3f} {axis_label} &nbsp; / &nbsp; 解析範囲 {low:.3f} — {high:.3f} &nbsp; / &nbsp; ドラッグで{escape(drag)}・ダブルクリックで全体表示</div>', unsafe_allow_html=True)
        if not any([show_spectrum, show_continuum, show_fit and result is not None]):
            st.info("表示レイヤーがありません。右の「レイヤー」で表示をオンにしてください。")
        if workspace == "線の解析":
            with st.expander("線の候補を探す"):
                prominence = st.number_input("ピークの最小突出度（フラックス単位）", min_value=0.000001,
                                             value=max(0.000001, float(np.std(spectrum.flux - continuum))), format="%.6f")
                candidates = line_candidates(spectrum, continuum, prominence, kind)
                if candidates:
                    st.dataframe(candidates, width="stretch")
                else:
                    st.caption("この閾値の候補はありません。")
                st.caption("候補は画素上の極値です。線の同定や有意性は範囲を指定して確認してください。")
    elif workspace == "2D画像":
        if image is None:
            st.info("右の「ファイルを開く」から2次元FITS画像を読み込むと、ここに画像と抽出範囲を表示します。")
        else:
            row_step, col_step = max(1, image.shape[0] // 500), max(1, image.shape[1] // 1000)
            shown = image[::row_step, ::col_step]
            finite = shown[np.isfinite(shown)]
            zmin, zmax = np.percentile(finite, [5, 99]) if finite.size else (0, 1)
            plot = go.Figure(go.Heatmap(z=shown, x=np.arange(0, image.shape[1], col_step),
                                       y=np.arange(0, image.shape[0], row_step), zmin=zmin, zmax=zmax, colorscale="Gray"))
            aperture = kwargs["aperture"]
            if kwargs["dispersion_axis"] == 1:
                plot.add_hrect(y0=aperture[0] - 0.5, y1=aperture[1] - 0.5, line_color="#dfac77", fillcolor="#dfac77", opacity=0.2)
            else:
                plot.add_vrect(x0=aperture[0] - 0.5, x1=aperture[1] - 0.5, line_color="#dfac77", fillcolor="#dfac77", opacity=0.2)
            plot.update_layout(xaxis_title="画像のXピクセル", yaxis_title="画像のYピクセル")
            chart_style(plot, grid=False, dragmode="zoom" if drag == "ズーム" else "pan", revision=revision)
            plot.update_yaxes(scaleanchor="x", scaleratio=1)
            st.plotly_chart(plot, width="stretch", theme=None, config=plot_config)
            st.caption("表示は5〜99パーセンタイル。大きい画像は表示時のみ間引きます。")
    else:
        st.subheader("解析結果を保存")
        st.caption("現在のドキュメントと測定値を書き出します。")
        left, right = st.columns(2)
        left.download_button("スペクトル CSV", export_csv(spectrum, continuum), "spectrum.csv", "text/csv", width="stretch")
        right.download_button("スペクトル FITS", export_fits(spectrum, continuum), "spectrum.fits", "application/octet-stream", width="stretch")
        if result is not None:
            report = dict(source=spectrum.source, axis_unit=spectrum.x_unit, flux_unit=spectrum.flux_unit,
                          fit_window=[low, high], line_kind=kind, continuum_degree=degree,
                          continuum_method="polynomial; 3-sigma MAD clipping; fit window excluded", measurements=result.to_dict())
            st.download_button("線の測定 JSON", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), "line_measurement.json", "application/json")
            with st.expander("測定結果のプレビュー", expanded=True):
                st.json(report)
        st.caption("FITSは抽出後の1次元テーブルを新規作成します。元ファイルの全HDUと装置固有ヘッダーは複製しません。")

status_bar(spectrum.x.size, axis_label, workspace, spectrum.error is not None)
