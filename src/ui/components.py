import os

import pandas as pd
import streamlit as st

SKIP_FULL = "Alta Precisión (100%)"
SKIP_BALANCED = "Equilibrado (66%)"
SKIP_PERFORMANCE = "Máx. Rendimiento (50%)"

SOURCE_CAMERA = "Cámara en vivo"
SOURCE_FILE = "Archivo de vídeo"

_KNEE_OPTIMAL = 85.0
_KNEE_PARALLEL = 90.0
_KNEE_DISPLAY_MIN = 60.0
_KNEE_DISPLAY_MAX = 180.0
_KNEE_RANGE = _KNEE_DISPLAY_MAX - _KNEE_DISPLAY_MIN

_TORSO_OPTIMAL = 30.0
_TORSO_LIMIT = 40.0
_TORSO_DISPLAY_MAX = 60.0

_VALGUS_GOOD = 0.90
_VALGUS_ALERT = 0.85
_VALGUS_DISPLAY_MIN = 0.60
_VALGUS_DISPLAY_MAX = 1.20
_VALGUS_RANGE = _VALGUS_DISPLAY_MAX - _VALGUS_DISPLAY_MIN

_VBT_DESCENT_MIN = 1.5
_VBT_DESCENT_MAX = 3.0
_VBT_ASCENT_MIN = 1.0
_VBT_ASCENT_MAX = 2.5

_FSM_LABELS = {
    0: "⚪ Reposo",
    1: "⬇ Bajando",
    2: "✓ Zona profunda",
    3: "⬆ Subiendo",
}

# Vocabulario cromático del estado FSM, compartido con el indicador HTML en app.py.
_FSM_LINE_COLORS = {
    0: "#9aa1ab",
    1: "#d97706",
    2: "#16a34a",
    3: "#0066cc",
}


def _clamp01(value: float) -> float:
    """Restringe value al intervalo [0.0, 1.0] requerido por st.progress."""
    return max(0.0, min(1.0, value))


def _knee_semaphore(angle: float) -> str:
    """Clasifica el ángulo de rodilla según los umbrales de profundidad."""
    if angle <= _KNEE_OPTIMAL:
        return "✓ Paralelo óptimo"
    if angle <= _KNEE_PARALLEL:
        return "✓ Paralelo roto"
    if angle <= 130:
        return "⬇ En descenso"
    return "— Posición inicial"


def _torso_semaphore(tilt: float) -> str:
    """Clasifica la inclinación del torso según los umbrales de tolerancia."""
    if tilt <= _TORSO_OPTIMAL:
        return "✓ Inclinación óptima"
    if tilt <= _TORSO_LIMIT:
        return "⚠ Cerca del límite"
    return "✗ Inclinación excesiva"


def _valgus_semaphore(ratio: float) -> str:
    """Clasifica el índice de valgo según los umbrales de alineación."""
    if ratio >= _VALGUS_GOOD:
        return "✓ Alineación correcta"
    if ratio >= _VALGUS_ALERT:
        return "⚠ Zona de alerta"
    return "✗ Valgo detectado"


def _vbt_caption(value: float, lo: float, hi: float) -> str:
    """Texto de referencia VBT: estado si hay dato, rango si no lo hay."""
    if value <= 0:
        return f"Ref: {lo}–{hi} s"
    return "✓ Óptimo" if lo <= value <= hi else "⚠ Fuera de rango"


def _build_history_df(history: list) -> pd.DataFrame:
    """Construye el DataFrame del historial analítico de la sesión."""
    rows = [
        {
            "Rep": r["rep"],
            "Estado": "Válida" if r["valid"] else "Fallo",
            "Errores": ", ".join(r["errors"]) if r["errors"] else "—",
            "Bajada (s)": round(r["descent_duration_sec"], 2),
            "Subida (s)": round(r["ascent_duration_sec"], 2),
        }
        for r in history
    ]
    return pd.DataFrame(rows)


@st.dialog("Historial analítico de la serie", width="large")
def show_history_modal(history: list) -> None:  # pragma: no cover
    """Muestra el historial analítico en un modal nativo de Streamlit.

    Args:
        history: Lista de RepRecord generada por SquatAnalyzer.
    """
    if not history:
        st.caption("Sin repeticiones registradas aún.")
        return
    st.dataframe(_build_history_df(history), hide_index=True, use_container_width=True)


def render_left_panel(
    placeholder,
    fsm_state: int,
    rep_valid: int,
    rep_invalid: int,
    descent_sec: float,
    ascent_sec: float,
) -> None:
    """Renderiza el panel izquierdo: estado FSM, contadores y telemetría VBT.

    Sustituye el contenido del placeholder en cada llamada para permitir
    actualización en vivo dentro del bucle de tracking sin acumular elementos.

    Args:
        placeholder: st.empty() de la columna izquierda.
        fsm_state: Estado actual de la FSM (0=reposo, 1=bajando,
            2=profundidad, 3=subiendo).
        rep_valid: Repeticiones válidas acumuladas en la sesión.
        rep_invalid: Repeticiones con fallo acumuladas en la sesión.
        descent_sec: Duración de la última bajada (0.0 si no hay dato).
        ascent_sec: Duración de la última subida (0.0 si no hay dato).
    """
    line_color = _FSM_LINE_COLORS.get(fsm_state, "#9aa1ab")

    with placeholder.container():
        st.metric("Estado del sistema", _FSM_LABELS.get(fsm_state, "—"))
        st.markdown(
            f'<div style="height:3px;background:{line_color};'
            'border-radius:2px;margin-bottom:6px;"></div>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        c1.metric("👍 Reps válidas", rep_valid)
        c2.metric("❌ Con fallo", rep_invalid)

        st.caption("TELEMETRÍA VBT · ÚLTIMA REPETICIÓN")
        v1, v2 = st.columns(2)
        with v1:
            d_val = f"{descent_sec:.1f} s" if descent_sec > 0 else "—"
            st.metric("⬇ Bajada", d_val)
            st.caption(_vbt_caption(descent_sec, _VBT_DESCENT_MIN, _VBT_DESCENT_MAX))
        with v2:
            a_val = f"{ascent_sec:.1f} s" if ascent_sec > 0 else "—"
            st.metric("⬆ Subida", a_val)
            st.caption(_vbt_caption(ascent_sec, _VBT_ASCENT_MIN, _VBT_ASCENT_MAX))


def render_bio_metrics(
    placeholder,
    knee_angle: float,
    torso_tilt: float,
    valgus_ratio: float,
) -> None:
    """Renderiza las tres métricas biomecánicas en tiempo real.

    Cada métrica combina valor numérico (st.metric), barra de posición sobre
    el rango de referencia (st.progress) y clasificación semáforo (st.caption).

    Args:
        placeholder: st.empty() de ancho completo bajo las columnas.
        knee_angle: Ángulo de rodilla en grados.
        torso_tilt: Inclinación del torso en grados.
        valgus_ratio: Cociente de valgo rodillas/caderas.
    """
    knee_pct = _clamp01((_KNEE_DISPLAY_MAX - knee_angle) / _KNEE_RANGE)
    torso_pct = _clamp01(1.0 - torso_tilt / _TORSO_DISPLAY_MAX)
    valgus_pct = _clamp01((valgus_ratio - _VALGUS_DISPLAY_MIN) / _VALGUS_RANGE)

    with placeholder.container():
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("📐 Ángulo de rodilla", f"{knee_angle:.1f}°")
            st.progress(knee_pct)
            st.caption(f"{_knee_semaphore(knee_angle)} · óptimo <85°")
        with m2:
            st.metric("📏 Inclinación torso", f"{torso_tilt:.1f}°")
            st.progress(torso_pct)
            st.caption(f"{_torso_semaphore(torso_tilt)} · óptimo <30°")
        with m3:
            st.metric("🦵 Índice de valgo", f"{valgus_ratio:.2f}")
            st.progress(valgus_pct)
            st.caption(f"{_valgus_semaphore(valgus_ratio)} · correcto >0.90")


def detect_runtime_env() -> bool:
    """Devuelve True si la app corre en local (cámara disponible).

    Detecta Streamlit Cloud via la variable STREAMLIT_SHARING_REPOSITORY
    o la ruta /mount/src presente en ese entorno.
    """
    if os.environ.get("STREAMLIT_SHARING_REPOSITORY") or os.path.exists("/mount/src"):
        return False
    return True


def render_sidebar_config(is_local: bool):
    """Renderiza el panel lateral de configuración.

    El slider de umbral superior se limita a 170° para evitar que valores
    próximos a 180° bloqueen la FSM (MediaPipe no devuelve exactamente 180°
    en rodilla extendida, lo que haría imposible salir del estado STAND).

    Args:
        is_local: True si el entorno tiene acceso a cámara.

    Returns:
        Tuple (source_mode, skip_mode, d_thr, u_thr, t_thr).
    """
    st.markdown("**⚙️ Configuración**")

    if is_local:
        options = [SOURCE_CAMERA, SOURCE_FILE]
    else:
        options = [SOURCE_FILE]
        st.caption("⚠️ Cámara no disponible en Cloud.")

    source_mode = st.radio("Fuente", options, index=0)
    skip_mode = st.selectbox(
        "Modo IA",
        [SKIP_FULL, SKIP_BALANCED, SKIP_PERFORMANCE],
        index=0 if is_local else 1,
    )
    st.caption("UMBRALES BIOMECÁNICOS")
    d_thr = st.slider("Profundidad", 70, 110, 90)
    u_thr = st.slider("Erguido", 130, 170, 150)
    t_thr = st.slider("Torso (°)", 20, 60, 40)

    return source_mode, skip_mode, d_thr, u_thr, t_thr


def render_header_and_instructions(is_local: bool, source_mode: str) -> None:
    """Renderiza la cabecera y detiene la ejecución si se detecta cámara en Cloud.

    Args:
        is_local: True si el entorno tiene acceso a cámara.
        source_mode: Modo de fuente de entrada seleccionado en el sidebar.
    """
    st.markdown(
        "#### 🏋️‍♂️ MirrorUS"
        '<span style="font-size:14px;color:#9aa1ab;'
        'font-weight:400;margin-left:10px;">'
        "Análisis Biomecánico · TFG</span>",
        unsafe_allow_html=True,
    )
    if not is_local and source_mode == SOURCE_CAMERA:
        st.error(
            "La cámara no está disponible en Streamlit Cloud. "
            "Selecciona 'Archivo de vídeo' para evaluar el sistema."
        )
        st.stop()


def handle_video_cleanup(input_path) -> None:
    """Elimina el archivo temporal de forma segura; no-op si input_path es int."""
    if isinstance(input_path, str) and os.path.exists(input_path):
        try:
            os.remove(input_path)
        except OSError:
            pass
