from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_demo_ui_and_absorption_interaction():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=45).run()
    assert not app.exception
    assert app.title[0].value == "FITS Spectral Lab"
    assert any(metric.label == "中心" for metric in app.metric)
    app.selectbox[0].select("吸収線").run()
    assert not app.exception
    assert app.selectbox[0].value == "吸収線"
    app.selectbox[0].select("輝線").run()
    assert not app.exception
    assert any(metric.label == "中心" for metric in app.metric)


def test_invalid_line_window_shows_error_without_crash():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=45).run()
    app.number_input[0].set_value(6590.0).run()
    assert not app.exception
    assert any("開始" in error.value for error in app.error)


def test_workbench_tools_preserve_line_settings_and_offer_exports():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=45).run()
    app.number_input(key="low_demo_Angstrom").set_value(6554.0).run()
    app.button(key="tool_線の解析").click().run()
    assert not app.exception
    assert app.session_state["workspace"] == "線の解析"
    assert app.number_input(key="low_demo_Angstrom").value == 6554.0
    assert len(app.get("plotly_chart")) == 1
    app.button(key="tool_保存・ファイル情報").click().run()
    assert not app.exception
    assert app.session_state["workspace"] == "保存・ファイル情報"
    assert len(app.get("download_button")) == 4
    app.button(key="tool_2D画像").click().run()
    assert not app.exception
    assert any("2次元FITS" in info.value for info in app.info)


def test_display_layers_and_zoom_reset_do_not_change_measurements():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=45).run()
    center = next(metric.value for metric in app.metric if metric.label == "中心")
    app.checkbox(key="layer_fit").uncheck().run()
    app.checkbox(key="layer_continuum").uncheck().run()
    assert not app.exception
    assert next(metric.value for metric in app.metric if metric.label == "中心") == center
    app.button(key="tool_reset").click().run()
    assert not app.exception
    assert app.session_state["view_epoch"] == 1
    assert next(metric.value for metric in app.metric if metric.label == "中心") == center
