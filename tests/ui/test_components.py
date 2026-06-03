import os
import tempfile
from unittest.mock import patch

from src.ui.components import (
    detect_runtime_env,
    handle_video_cleanup,
    render_header_and_instructions,
    render_sidebar_config,
)

# --- detect_runtime_env ---


def test_detect_runtime_env_local(monkeypatch):
    """Verifica que en entorno local devuelve True."""
    monkeypatch.delenv("STREAMLIT_SHARING_REPOSITORY", raising=False)
    monkeypatch.setattr(os.path, "exists", lambda p: False)

    assert detect_runtime_env() is True


def test_detect_runtime_env_cloud(monkeypatch):
    """Verifica que con la variable de entorno de Cloud presente devuelve False."""
    monkeypatch.setenv("STREAMLIT_SHARING_REPOSITORY", "1")

    assert detect_runtime_env() is False


# --- handle_video_cleanup ---


def test_handle_video_cleanup_removes_file():
    """Verifica que el archivo temporal se elimina correctamente si existe."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name

    assert os.path.exists(path)
    handle_video_cleanup(path)
    assert not os.path.exists(path)


def test_handle_video_cleanup_missing_file():
    """Verifica que no se lanza excepción si el archivo no existe."""
    handle_video_cleanup("/tmp/archivo_que_no_existe_xyz.mp4")


def test_handle_video_cleanup_ignores_non_string():
    """Verifica que no se lanza excepción si se recibe un entero (fuente de cámara)."""
    handle_video_cleanup(0)


def test_handle_video_cleanup_oserror(monkeypatch):
    """Verifica que un OSError al eliminar el archivo se captura silenciosamente."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name

    def raise_oserror(p):
        raise OSError("Permission denied")

    monkeypatch.setattr(os, "remove", raise_oserror)
    handle_video_cleanup(path)


# --- render_sidebar_config ---


@patch("src.ui.components.st")
def test_render_sidebar_config_local_options(mock_st):
    """Verifica que en entorno local se ofrece la opción de cámara en vivo."""
    mock_st.radio.return_value = "Cámara en vivo"
    mock_st.selectbox.return_value = "Alta Precisión (100% IA)"
    mock_st.slider.side_effect = [90, 150, 40]

    source_mode, skip_mode, d_thr, u_thr, t_thr = render_sidebar_config(is_local=True)

    assert source_mode == "Cámara en vivo"
    assert skip_mode == "Alta Precisión (100% IA)"
    assert d_thr == 90
    assert u_thr == 150
    assert t_thr == 40
    assert "Cámara en vivo" in mock_st.radio.call_args[0][1]


@patch("src.ui.components.st")
def test_render_sidebar_config_cloud_no_camera(mock_st):
    """Verifica que en entorno Cloud no se ofrece la cámara y se muestra el aviso."""
    mock_st.radio.return_value = "Archivo de vídeo (Debug)"
    mock_st.selectbox.return_value = "Equilibrado (66% IA)"
    mock_st.slider.side_effect = [90, 150, 40]

    source_mode, _, _, _, _ = render_sidebar_config(is_local=False)

    assert source_mode == "Archivo de vídeo (Debug)"
    assert "Cámara en vivo" not in mock_st.radio.call_args[0][1]
    mock_st.caption.assert_called_once()


@patch("src.ui.components.st")
def test_render_sidebar_config_slider_values(mock_st):
    """Verifica que los valores de los sliders se devuelven correctamente."""
    mock_st.radio.return_value = "Archivo de vídeo (Debug)"
    mock_st.selectbox.return_value = "Alta Precisión (100% IA)"
    mock_st.slider.side_effect = [75, 165, 35]

    _, _, d_thr, u_thr, t_thr = render_sidebar_config(is_local=True)

    assert d_thr == 75
    assert u_thr == 165
    assert t_thr == 35


# --- render_header_and_instructions ---


@patch("src.ui.components.st")
def test_render_header_local_no_errors(mock_st):
    """Verifica que en entorno local se renderiza el título sin errores ni stop."""
    render_header_and_instructions(is_local=True, source_mode="Cámara en vivo")

    mock_st.title.assert_called_once()
    mock_st.caption.assert_called_once()
    mock_st.error.assert_not_called()
    mock_st.stop.assert_not_called()


@patch("src.ui.components.st")
def test_render_header_cloud_video_mode_no_stop(mock_st):
    """Verifica que en Cloud con modo vídeo no se muestra error ni se detiene la app."""
    render_header_and_instructions(
        is_local=False, source_mode="Archivo de vídeo (Debug)"
    )

    mock_st.error.assert_not_called()
    mock_st.stop.assert_not_called()


@patch("src.ui.components.st")
def test_render_header_cloud_camera_shows_error(mock_st):
    """Verifica que en Cloud con cámara en vivo se muestra error y se detiene la app."""
    render_header_and_instructions(is_local=False, source_mode="Cámara en vivo")

    mock_st.error.assert_called_once()
    mock_st.stop.assert_called_once()
