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
