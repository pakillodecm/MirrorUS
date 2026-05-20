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
    """Renderiza el panel lateral de configuración y gestiona el archivo de vídeo.
    Maneja el aislamiento de archivos por sesión para evitar colisiones.

    Args:
        is_local (bool): Indicador de entorno de ejecución.
        session_id (str): Identificador único de la sesión actual.

    Returns:
        tuple: (source_mode, input_path, do_flip, skip_mode)
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

    input_path = 0
    do_flip = True

    if source_mode == "Archivo de vídeo (Debug)":
        video_file = st.file_uploader(
            "Sube un vídeo de tu sentadilla", type=["mp4", "mov", "avi"]
        )
        if video_file:
            _, file_extension = os.path.splitext(video_file.name)
            input_path = f"./temp_demo_{session_id}{file_extension}"

            if not os.path.exists(input_path):
                with open(input_path, "wb") as f:
                    f.write(video_file.read())

            do_flip = False
        else:
            st.warning("Por favor, sube un archivo de vídeo para continuar.")
            st.stop()

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

    st.session_state.counter.thr_down = d_thr
    st.session_state.counter.thr_up = u_thr

    return source_mode, input_path, do_flip, skip_mode


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
