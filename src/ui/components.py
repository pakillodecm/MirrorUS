import os

import pandas as pd
import streamlit as st

FSM_LINE_COLOR_DEFAULT = "#9aa1ab"

SOURCE_CAMERA = "Cámara en vivo"
SOURCE_FILE = "Archivo de vídeo"

SKIP_FULL = "Alta Precisión (100%)"
SKIP_BALANCED = "Equilibrado (66%)"
SKIP_PERFORMANCE = "Máx. Rendimiento (50%)"

SKIP_OPTIONS = [SKIP_FULL, SKIP_BALANCED, SKIP_PERFORMANCE]

UI_DOWN_LIMIT_SLIDER_MIN = 80
UI_DOWN_LIMIT_SLIDER_MAX = 120
UI_DOWN_LIMIT_SLIDER_DEFAULT = 95

UI_UP_LIMIT_SLIDER_MIN = 130
UI_UP_LIMIT_SLIDER_MAX = 170
UI_UP_LIMIT_SLIDER_DEFAULT = 155

UI_TORSO_SLIDER_MIN = 25
UI_TORSO_SLIDER_MAX = 55
UI_TORSO_SLIDER_DEFAULT = 40

KNEE_OPT = 85.0
TORSO_OPT = 30.0
VALGUS_GOOD = 0.05
VALGUS_ALERT = 0.08

KNEE_DISPLAY_MIN = 60.0
KNEE_DISPLAY_MAX = 180.0
KNEE_RANGE = KNEE_DISPLAY_MAX - KNEE_DISPLAY_MIN

TORSO_DISPLAY_MAX = 60.0

VALGUS_DISPLAY_MIN = -0.05
VALGUS_DISPLAY_MAX = 0.20
VALGUS_RANGE = VALGUS_DISPLAY_MAX - VALGUS_DISPLAY_MIN

VBT_DESCENT_MIN = 1.5
VBT_DESCENT_MAX = 3.0
VBT_ASCENT_MIN = 1.0
VBT_ASCENT_MAX = 2.5

FSM_LABELS = {
    0: "○ Reposo",
    1: "⬇ Bajando",
    2: "◆ Zona profunda",
    3: "⬆ Subiendo",
}

FSM_LINE_COLORS = {
    0: FSM_LINE_COLOR_DEFAULT,
    1: "#d97706",
    2: "#16a34a",
    3: "#0066cc",
}

_ERROR_LABELS = {
    "NO_DEPTH": "Sin profundidad",
    "MID_ASCENT_COLLAPSE": "Colapso en ascenso",
    "KNEE_VALGUS": "Valgo de rodilla",
    "TORSO_TILT": "Inclinación de torso",
}


def _clamp01(value: float) -> float:
    """Restringe value al intervalo [0.0, 1.0] requerido por st.progress."""
    return max(0.0, min(1.0, value))


def _knee_semaphore(angle: float, d_thr: float, u_thr: float) -> str:
    """Clasifica el ángulo de rodilla según los umbrales de profundidad."""
    if d_thr > KNEE_OPT and angle <= KNEE_OPT:
        return "✓ Profundidad óptima"
    if angle <= d_thr:
        return "✓ Paralelo roto"
    if angle <= u_thr:
        return "⬍ En movimiento"
    return "— Posición inicial"


def _torso_semaphore(tilt: float, t_thr: float) -> str:
    """Clasifica la inclinación del torso según los umbrales de tolerancia."""
    if tilt <= TORSO_OPT and tilt <= t_thr:
        return "✓ Inclinación óptima"
    if tilt <= t_thr:
        return "⚠ Cerca del límite"
    return "✗ Inclinación excesiva"


def _valgus_semaphore(dev: float) -> str:
    """Clasifica la desviación de valgo según los umbrales de referencia."""
    if dev < VALGUS_GOOD:
        return "✓ Alineación correcta"
    if dev < VALGUS_ALERT:
        return "⚠ Zona de alerta"
    return "✗ Valgo detectado"


def _vbt_caption(value: float, lo: float, hi: float) -> str:
    """Texto de referencia VBT: estado si hay dato, rango si no lo hay."""
    if value <= 0:
        return f"Ref: {lo:.1f} - {hi:.1f} s"
    return "✓ Dentro de rango" if lo <= value <= hi else "⚠ Fuera de rango"


def _format_error(error_name: str, record: dict) -> str:
    """Formatea un error con su etiqueta humana y los valores alcanzados."""
    label = _ERROR_LABELS.get(error_name, error_name)
    if error_name == "NO_DEPTH":
        angle = record.get("min_knee_angle")
        thr = record.get("depth_threshold")
        if angle is not None and thr is not None:
            return f"{label} ({angle:.0f}° / obj. <{thr:.0f}°)"
    elif error_name == "KNEE_VALGUS":
        dev = record.get("max_valgus_dev")
        if dev is not None:
            return f"{label} (desv. {dev:.3f} / lím. <{VALGUS_GOOD:.3f})"
    elif error_name == "TORSO_TILT":
        tilt = record.get("max_torso_tilt")
        thr = record.get("torso_threshold")
        if tilt is not None and thr is not None:
            return f"{label} ({tilt:.0f}° / lím. {thr:.0f}°)"
    return label


def _build_history_df(history: list) -> pd.DataFrame:
    """Construye el DataFrame del historial analítico de la sesión."""
    rows = []
    for r in history:
        depth = r.get("min_knee_angle")
        depth_str = f"{depth:.0f}°" if depth is not None else "—"

        if r["errors"]:
            error_lines = [_format_error(e, r) for e in r["errors"]]
            errors_text = "  |  ".join(error_lines)
        else:
            errors_text = "—"

        rows.append(
            {
                "Rep": r["rep"],
                "Estado": "Válida ✔" if r["valid"] else "Fallida ✖",
                "Flexión de rodillas (°)": depth_str,
                "Errores": errors_text,
                "Bajada (s)": round(r["descent_duration_sec"], 2),
                "Subida (s)": round(r["ascent_duration_sec"], 2),
            }
        )
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
    st.dataframe(
        _build_history_df(history),
        hide_index=True,
        width="stretch",
        column_config={
            "Rep": st.column_config.NumberColumn(width="small"),
            "Estado": st.column_config.TextColumn(width="small"),
            "Flexión de rodillas (°)": st.column_config.TextColumn(width="small"),
            "Errores": st.column_config.TextColumn(width="large"),
            "Bajada (s)": st.column_config.NumberColumn(width="small"),
            "Subida (s)": st.column_config.NumberColumn(width="small"),
        },
    )


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
    line_color = FSM_LINE_COLORS.get(fsm_state, FSM_LINE_COLOR_DEFAULT)

    with placeholder.container():
        st.metric("Estado del sistema", FSM_LABELS.get(fsm_state, "—"))
        st.markdown(
            f'<div style="height:3px;background:{line_color};'
            'border-radius:2px;margin-bottom:6px;"></div>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        c1.metric(
            "✔ Reps válidas", rep_valid, help="Repeticiones válidas en la sesión."
        )
        c2.metric(
            "✖ Con fallo", rep_invalid, help="Repeticiones con fallo en la sesión."
        )

        st.caption("TELEMETRÍA VBT · ÚLTIMA REPETICIÓN")
        v1, v2 = st.columns(2)
        with v1:
            d_val = f"{descent_sec:.1f} s" if descent_sec > 0 else "—"
            st.metric("⬇ Bajada", d_val, help="Duración de la fase de bajada.")
            st.caption(_vbt_caption(descent_sec, VBT_DESCENT_MIN, VBT_DESCENT_MAX))
        with v2:
            a_val = f"{ascent_sec:.1f} s" if ascent_sec > 0 else "—"
            st.metric("⬆ Subida", a_val, help="Duración de la fase de subida.")
            st.caption(_vbt_caption(ascent_sec, VBT_ASCENT_MIN, VBT_ASCENT_MAX))


def render_bio_metrics(
    placeholder,
    knee_angle: float,
    torso_tilt: float,
    valgus_ratio: float,
    d_thr: float,
    u_thr: float,
    t_thr: float,
) -> None:
    """Renderiza las tres métricas biomecánicas en tiempo real.

    Cada métrica combina valor numérico (st.metric), barra de posición sobre
    el rango de referencia (st.progress) y clasificación semáforo (st.caption).

    Args:
        placeholder: st.empty() de ancho completo bajo las columnas.
        knee_angle: Ángulo de rodilla en grados.
        torso_tilt: Inclinación del torso en grados.
        valgus_ratio: Cociente de valgo rodillas/tobillos.
        d_thr: Umbral de profundidad para clasificación de rodilla.
        u_thr: Umbral de erguido para clasificación de rodilla.
        t_thr: Umbral de inclinación para clasificación de torso.
    """
    knee_pct = _clamp01((KNEE_DISPLAY_MAX - knee_angle) / KNEE_RANGE)
    torso_pct = _clamp01(1.0 - torso_tilt / TORSO_DISPLAY_MAX)
    valgus_pct = _clamp01(1.0 - (valgus_ratio - VALGUS_DISPLAY_MIN) / VALGUS_RANGE)

    with placeholder.container():
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(
                "∡ Ángulo de rodilla",
                f"{knee_angle:.1f}°",
                help="Ángulo entre el muslo y la pantorrilla.",
            )
            st.progress(knee_pct)
            st.caption(
                f"{_knee_semaphore(knee_angle, d_thr, u_thr)} · óptimo <{KNEE_OPT:.0f}°"
            )
        with m2:
            st.metric(
                "↗ Inclinación torso",
                f"{torso_tilt:.1f}°",
                help="Inclinación del torso respecto a la vertical.",
            )
            st.progress(torso_pct)
            st.caption(
                f"{_torso_semaphore(torso_tilt, t_thr)} · óptimo <{TORSO_OPT:.0f}°"
            )
        with m3:
            st.metric(
                "∥ Índice de valgo",
                f"{valgus_ratio:.3f}",
                help="Desviación medial de rodilla respecto al eje cadera-tobillo.",
            )
            st.progress(valgus_pct)
            st.caption(
                f"{_valgus_semaphore(valgus_ratio)} · correcto <{VALGUS_GOOD:.2f}"
            )


def detect_runtime_env() -> bool:
    """Devuelve True si la app corre en local (cámara disponible).

    Detecta Streamlit Cloud via la variable STREAMLIT_SHARING_REPOSITORY
    o la ruta /mount/src presente en ese entorno.
    """
    if os.environ.get("STREAMLIT_SHARING_REPOSITORY") or os.path.exists("/mount/src"):
        return False
    return True


def render_sidebar_config(is_local: bool, disabled: bool = False):
    """Renderiza el panel lateral de configuración.

    El slider de umbral superior se limita a 170° para evitar que valores
    próximos a 180° bloqueen la FSM (MediaPipe no devuelve exactamente 180°
    en rodilla extendida, lo que haría imposible salir del estado STAND).

    Args:
        is_local: True si el entorno tiene acceso a cámara.
        disabled: True si los controles deben deshabilitarse (durante tracking).

    Returns:
        Tuple (source_mode, skip_mode, d_thr, u_thr, t_thr).
    """
    st.markdown("**⚙ Configuración**")

    if is_local:
        options = [SOURCE_CAMERA, SOURCE_FILE]
    else:
        options = [SOURCE_FILE]
        st.caption("⚠ Cámara no disponible en Cloud.")

    source_mode = st.radio("Fuente", options, index=0, disabled=disabled)
    skip_mode = st.selectbox(
        "Modo IA",
        SKIP_OPTIONS,
        index=0 if is_local else 1,
        disabled=disabled,
    )
    st.caption("UMBRALES BIOMECÁNICOS")
    d_thr = st.slider(
        "Profundidad (°)",
        UI_DOWN_LIMIT_SLIDER_MIN,
        UI_DOWN_LIMIT_SLIDER_MAX,
        UI_DOWN_LIMIT_SLIDER_DEFAULT,
        disabled=disabled,
    )
    u_thr = st.slider(
        "Erguido (°)",
        UI_UP_LIMIT_SLIDER_MIN,
        UI_UP_LIMIT_SLIDER_MAX,
        UI_UP_LIMIT_SLIDER_DEFAULT,
        disabled=disabled,
    )
    t_thr = st.slider(
        "Torso (°)",
        UI_TORSO_SLIDER_MIN,
        UI_TORSO_SLIDER_MAX,
        UI_TORSO_SLIDER_DEFAULT,
        disabled=disabled,
    )

    return source_mode, skip_mode, d_thr, u_thr, t_thr


def render_header_and_instructions(is_local: bool, source_mode: str) -> None:
    """Renderiza la cabecera y detiene la ejecución si se detecta cámara en Cloud.

    Args:
        is_local: True si el entorno tiene acceso a cámara.
        source_mode: Modo de fuente de entrada seleccionado en el sidebar.
    """
    st.markdown(
        "#### ✦ MirrorUS"
        '<span style="font-size:14px;color:#9aa1ab;'
        'font-weight:400;margin-left:10px;">'
        "Análisis Biomecánico · TFG</span>",
        unsafe_allow_html=True,
    )
    if not is_local and source_mode == SOURCE_CAMERA:
        st.error(
            "La cámara no está disponible en Streamlit Cloud. "
            f"Selecciona '{SOURCE_FILE}' para evaluar el sistema."
        )
        st.stop()


def handle_video_cleanup(input_path) -> None:
    """Elimina el archivo temporal de forma segura; no-op si input_path es int."""
    if isinstance(input_path, str) and os.path.exists(input_path):
        try:
            os.remove(input_path)
        except OSError:
            pass
