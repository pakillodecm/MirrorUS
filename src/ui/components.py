import os

import streamlit as st


def detect_runtime_env() -> bool:
    """Detecta si la aplicación se está ejecutando en local o en Streamlit Cloud.

    Returns:
        bool: True si es entorno local (permite cámara), False si es Cloud.
    """
    if os.environ.get("STREAMLIT_SHARING_REPOSITORY") or os.path.exists("/mount/src"):
        return False
    return True


def render_sidebar_config(is_local: bool, session_id: str):
    """Renderiza el panel lateral de configuración sin acoplamiento de estado interno.

    Maneja los sliders biomecánicos retornando valores primitivos de configuración.

    Args:
        is_local (bool): Indicador de entorno de ejecución.
        session_id (str): Identificador único de la sesión actual.

    Returns:
        tuple: (source_mode, skip_mode, d_thr, u_thr)
    """
    st.header("⚙️ Configuración")

    if is_local:
        options = ["Cámara en vivo", "Archivo de vídeo (Debug)"]
        index = 0
    else:
        options = ["Archivo de vídeo (Debug)"]
        index = 0
        st.caption(
            "⚠️ *Cámara en vivo deshabilitada en servidores Cloud (Falta periférico).*"
        )

    source_mode = st.radio("Fuente de entrada:", options, index=index)

    st.divider()
    st.subheader("⚡ Optimización de Rendimiento")

    skip_mode = st.selectbox(
        "Modo de ejecución:",
        [
            "Alta Precisión (100% IA)",
            "Equilibrado (66% IA)",
            "Máximo Rendimiento (50% IA)",
        ],
        index=0 if is_local else 1,
    )

    st.divider()
    st.subheader("🎯 Umbrales Biomecánicos")
    d_thr = st.slider("Umbral Profundidad (Ángulo bajo)", 60, 110, 90)
    u_thr = st.slider("Umbral Erguido (Ángulo alto)", 130, 180, 150)

    return source_mode, skip_mode, d_thr, u_thr


def render_header_and_instructions(is_local: bool, source_mode: str):
    """Renderiza la cabecera de la aplicación y las alertas del sistema.

    Args:
        is_local (bool): Indicador de entorno de ejecución.
        source_mode (str): Modo de fuente de entrada."""
    st.title("🏋️‍♂️ MirrorUS")
    st.caption("Trabajo Fin de Grado - Sistema Inteligente de Análisis Biomecánico")

    if not is_local and source_mode == "Cámara en vivo":
        st.error(
            "La cámara en vivo no se puede inicializar en Streamlit Cloud. "
            "Esta funcionalidad está pendiente de implementación en depliegue. "
            "Por favor, selecciona 'Archivo de vídeo (Debug)' para evaluar el sistema."
        )
        st.stop()


def handle_video_cleanup(input_path: str):
    """Elimina de forma segura el archivo de vídeo temporal del espacio de trabajo."""
    if isinstance(input_path, str) and os.path.exists(input_path):
        try:
            os.remove(input_path)
        except OSError:
            pass
