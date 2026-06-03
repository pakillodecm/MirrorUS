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
    """En entorno local no debe existir la variable de Cloud ni /mount/src."""
    monkeypatch.delenv("STREAMLIT_SHARING_REPOSITORY", raising=False)
    monkeypatch.setattr(os.path, "exists", lambda p: False)
    assert detect_runtime_env() is True


def test_detect_runtime_env_cloud(monkeypatch):
    """Si la variable de entorno de Cloud está presente, devuelve False."""
    monkeypatch.setenv("STREAMLIT_SHARING_REPOSITORY", "1")
    assert detect_runtime_env() is False


# --- handle_video_cleanup ---


def test_handle_video_cleanup_removes_file():
    """Debe eliminar el archivo temporal si existe."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name

    assert os.path.exists(path)
    handle_video_cleanup(path)
    assert not os.path.exists(path)


def test_handle_video_cleanup_missing_file():
    """No debe lanzar excepción si el archivo no existe."""
    handle_video_cleanup("/tmp/archivo_que_no_existe_xyz.mp4")


def test_handle_video_cleanup_ignores_non_string():
    """No debe lanzar excepción si recibe un entero (fuente de cámara)."""
    handle_video_cleanup(0)


def test_handle_video_cleanup_oserror(monkeypatch):
    """No debe propagar OSError si el archivo no se puede eliminar."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name

    def raise_oserror(p):
        raise OSError("Permission denied")

    monkeypatch.setattr(os, "remove", raise_oserror)
    handle_video_cleanup(path)  # No debe lanzar excepción


# --- render_sidebar_config ---


@patch("src.ui.components.st")
def test_render_sidebar_config_local_options(mock_st):
    """En entorno local debe ofrecer cámara y archivo como opciones."""
    mock_st.radio.return_value = "Cámara en vivo"
    mock_st.selectbox.return_value = "Alta Precisión (100% IA)"
    mock_st.slider.side_effect = [90, 150, 40]

    source_mode, skip_mode, d_thr, u_thr, t_thr = render_sidebar_config(is_local=True)

    assert source_mode == "Cámara en vivo"
    assert skip_mode == "Alta Precisión (100% IA)"
    assert d_thr == 90
    assert u_thr == 150
    assert t_thr == 40

    # Verifica que se ofreció la opción de cámara en el radio
    options_passed = mock_st.radio.call_args[0][1]
    assert "Cámara en vivo" in options_passed


@patch("src.ui.components.st")
def test_render_sidebar_config_cloud_no_camera(mock_st):
    """En entorno Cloud no debe ofrecer la opción de cámara en vivo."""
    mock_st.radio.return_value = "Archivo de vídeo (Debug)"
    mock_st.selectbox.return_value = "Equilibrado (66% IA)"
    mock_st.slider.side_effect = [90, 150, 40]

    source_mode, skip_mode, d_thr, u_thr, t_thr = render_sidebar_config(is_local=False)

    assert source_mode == "Archivo de vídeo (Debug)"

    # Verifica que NO se ofreció la cámara en las opciones
    options_passed = mock_st.radio.call_args[0][1]
    assert "Cámara en vivo" not in options_passed

    # Verifica que se muestra el aviso de Cloud
    mock_st.caption.assert_called_once()


@patch("src.ui.components.st")
def test_render_sidebar_config_slider_values(mock_st):
    """Los valores devueltos deben coincidir exactamente con los sliders."""
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
    """En entorno local debe renderizar título sin mostrar errores ni detener."""
    render_header_and_instructions(is_local=True, source_mode="Cámara en vivo")

    mock_st.title.assert_called_once()
    mock_st.caption.assert_called_once()
    mock_st.error.assert_not_called()
    mock_st.stop.assert_not_called()


@patch("src.ui.components.st")
def test_render_header_cloud_video_mode_no_stop(mock_st):
    """En Cloud con modo vídeo no debe mostrar error ni detener la app."""
    render_header_and_instructions(
        is_local=False, source_mode="Archivo de vídeo (Debug)"
    )

    mock_st.error.assert_not_called()
    mock_st.stop.assert_not_called()


@patch("src.ui.components.st")
def test_render_header_cloud_camera_shows_error(mock_st):
    """En Cloud con cámara en vivo debe mostrar error y detener la app."""
    render_header_and_instructions(is_local=False, source_mode="Cámara en vivo")

    mock_st.error.assert_called_once()
    mock_st.stop.assert_called_once()
