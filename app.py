import glob
import hashlib
import os
import time
import uuid

import cv2
import numpy as np
import streamlit as st

from src.logic.depth_detector import DepthDetector
from src.logic.pose_detector import PoseDetector
from src.logic.squat_analyzer import SquatAnalyzer
from src.logic.torso_detector import TorsoTiltDetector
from src.logic.valgus_detector import KneeValgusDetector
from src.ui.components import (
    SKIP_BALANCED,
    SKIP_PERFORMANCE,
    SOURCE_CAMERA,
    SOURCE_FILE,
    detect_runtime_env,
    handle_video_cleanup,
    render_bio_metrics,
    render_header_and_instructions,
    render_left_panel,
    render_sidebar_config,
    show_history_modal,
)

# ---------------------------------------------------------------------------
# Paleta de colores BGR para el esqueleto
# ---------------------------------------------------------------------------
_COLOR_BLUE = (204, 102, 0)  # azul vivo: movimiento sin error (#0066cc)
_COLOR_BLUE_DIM = (155, 145, 130)  # azul muy desaturado: movimiento con error de frame
_COLOR_GREEN = (74, 163, 22)  # verde: estado 0 tras rep válida
_COLOR_ORANGE = (0, 140, 255)  # naranja: estado 0 tras rep inválida
_COLOR_RED = (0, 0, 220)  # rojo: articulaciones destacadas

# ---------------------------------------------------------------------------
# Configuración visual por tipo de error
#
# ring:       joints con marcador rojo + anillo blanco
# plain:      joints con marcador rojo sin anillo
# conn_plain: si True, los joints plain también participan en la generación
#             de conexiones destacadas (además de los ring). Necesario para
#             TORSO_TILT, donde se quiere resaltar el trapecio completo
#             hombro-hombro / hombro-cadera / cadera-cadera.
# ---------------------------------------------------------------------------
_ERROR_VISUAL_CONFIG = {
    "NO_DEPTH": {
        "ring": ["LEFT_HIP", "RIGHT_HIP"],
        "plain": ["LEFT_KNEE", "RIGHT_KNEE"],
        "conn_plain": False,
    },
    "MID_ASCENT_COLLAPSE": {
        "ring": ["LEFT_HIP", "RIGHT_HIP"],
        "plain": ["LEFT_KNEE", "RIGHT_KNEE"],
        "conn_plain": False,
    },
    "KNEE_VALGUS": {
        "ring": ["LEFT_KNEE", "RIGHT_KNEE"],
        "plain": ["LEFT_ANKLE", "RIGHT_ANKLE"],
        "conn_plain": False,
    },
    "TORSO_TILT": {
        "ring": ["LEFT_SHOULDER", "RIGHT_SHOULDER"],
        "plain": ["LEFT_HIP", "RIGHT_HIP"],
        "conn_plain": True,
    },
}

_DEPTH_STATE_CONFIG = {
    0: ("#9ba3af", "○ Reposo"),
    1: ("#d97706", "⬇ Bajando"),
    2: ("#16a34a", "◆ Zona profunda"),
    3: ("#0066cc", "⬆ Subiendo"),
}


def _resolve_error_visuals(error_names, mp_pose):
    """Resuelve los conjuntos de articulaciones y conexiones a destacar.

    Combina la configuración de múltiples errores por unión. El anillo tiene
    prioridad sobre plain cuando un joint aparece en ambos.

    Una conexión se destaca si:
    - Al menos uno de sus extremos pertenece a conn_joints, Y
    - Ambos extremos pertenecen al universo (ring ∪ plain).

    conn_joints == ring por defecto. Si algún error activa conn_plain=True,
    conn_joints se amplía a ring ∪ plain para ese error.

    Args:
        error_names: Iterable de nombres de error activos.
        mp_pose: mp.solutions.pose para resolver índices de landmark.

    Returns:
        Tupla (ring_indices, plain_indices, conn_joints) como sets de enteros.
    """
    ring = set()
    plain = set()
    conn_plain = False

    for name in error_names:
        cfg = _ERROR_VISUAL_CONFIG.get(name, {})
        ring.update(mp_pose.PoseLandmark[j].value for j in cfg.get("ring", []))
        plain.update(mp_pose.PoseLandmark[j].value for j in cfg.get("plain", []))
        if cfg.get("conn_plain", False):
            conn_plain = True

    plain -= ring  # el anillo tiene prioridad sobre plain
    conn_joints = ring | (plain if conn_plain else set())

    return ring, plain, conn_joints


def _draw_skeleton(
    frame,
    landmarks,
    mp_pose,
    base_color,
    ring_indices,
    plain_indices,
    conn_joints,
):
    """Dibuja el esqueleto con codificación de color unificada.

    Las conexiones se destacan en rojo cuando al menos un extremo pertenece
    a conn_joints y ambos extremos están en el universo de joints destacados.
    Joints con ring: rojo con anillo blanco. Joints plain: rojo sin anillo.
    El resto del esqueleto se dibuja en base_color.

    Args:
        frame: Frame BGR de OpenCV.
        landmarks: pose_landmarks del resultado raw de MediaPipe.
        mp_pose: mp.solutions.pose accesible desde el detector persistente.
        base_color: Tupla BGR para conexiones y joints no destacados.
        ring_indices: Set de índices MediaPipe con anillo blanco.
        plain_indices: Set de índices MediaPipe sin anillo.
        conn_joints: Set de índices que generan conexiones destacadas.
    """
    h, w = frame.shape[:2]
    lm_list = landmarks.landmark
    universe = ring_indices | plain_indices

    for conn in mp_pose.POSE_CONNECTIONS:
        s, e = tuple(conn)
        lm_s, lm_e = lm_list[s], lm_list[e]
        if lm_s.visibility < 0.3 or lm_e.visibility < 0.3:
            continue
        pt_s = (int(lm_s.x * w), int(lm_s.y * h))
        pt_e = (int(lm_e.x * w), int(lm_e.y * h))
        is_highlighted = (
            (s in conn_joints or e in conn_joints) and s in universe and e in universe
        )
        if is_highlighted:
            cv2.line(frame, pt_s, pt_e, _COLOR_RED, 3, cv2.LINE_AA)
        else:
            cv2.line(frame, pt_s, pt_e, base_color, 2, cv2.LINE_AA)

    for idx, lm in enumerate(lm_list):
        if lm.visibility < 0.3:
            continue
        cx, cy = int(lm.x * w), int(lm.y * h)
        if idx in ring_indices:
            cv2.circle(frame, (cx, cy), 8, _COLOR_RED, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 11, (255, 255, 255), 2, cv2.LINE_AA)
        elif idx in plain_indices:
            cv2.circle(frame, (cx, cy), 6, _COLOR_RED, -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (cx, cy), 5, base_color, -1, cv2.LINE_AA)


def _depth_indicator_html(
    progress: float,
    fsm_state: int,
    min_progress: float = 0.0,
    show_max_hint: bool = False,
    min_angle_deg: float = 180.0,
) -> str:
    """Genera la barra de profundidad HTML coloreada por fase FSM con marca de mínimo.

    st.progress no permite cambiar de color, de ahí el HTML directo. La marca
    vertical indica la profundidad máxima alcanzada esta repetición.

    Args:
        progress: Progreso actual de profundidad en [0.0, 1.0].
        fsm_state: Estado actual de la FSM (0-3).
        min_progress: Profundidad máxima alcanzada esta rep, en [0.0, 1.0].
            Se usa para posicionar la marca vertical del tick.
        show_max_hint: Si True muestra el máximo en el label (activo tras NO_DEPTH).
        min_angle_deg: Ángulo mínimo de rodilla alcanzado esta rep, en grados.

    Returns:
        HTML listo para st.markdown con unsafe_allow_html=True.
    """
    color, label = _DEPTH_STATE_CONFIG.get(fsm_state, ("#9ba3af", "—"))
    pct = int(progress * 100)
    min_pct = int(min_progress * 100)

    if show_max_hint and min_pct > 0:
        right_label = (
            f'<span style="color:#d97706;font-weight:500;">'
            f"&nbsp;Sin profundidad suficiente&nbsp({min_angle_deg:.0f}°)</span>"
        )
    else:
        right_label = (
            f'<span style="color:{color};font-weight:500;">'
            f"{pct}%&nbsp;·&nbsp;{label}</span>"
        )

    min_tick = (
        '<div style="height:3px;position:relative;margin-top:1px;">'
        f'<div style="position:absolute;left:{min_pct}%;'
        "transform:translateX(-50%);width:2px;height:3px;"
        'background:#3a4453;border-radius:1px;"></div></div>'
        if min_pct > 2
        else ""
    )

    return (
        '<div style="margin-top:6px;">'
        '<div style="display:flex;justify-content:space-between;'
        f'font-size:11px;color:#6b7785;margin-bottom:4px;">'
        '<span style="font-weight:600;letter-spacing:0.04em;">'
        f"Profundidad</span>{right_label}</div>"
        '<div style="height:6px;background:#e3e6ec;border-radius:4px;overflow:hidden;">'
        f'<div style="width:{pct}%;height:6px;background:{color};'
        'border-radius:4px;"></div>'
        f"</div>{min_tick}</div>"
    )


def _video_placeholder_html() -> str:
    """Card oscura con esquinas de visor e instrucciones de posicionamiento.

    Returns:
        HTML listo para st.markdown con unsafe_allow_html=True.
    """
    corner = (
        "position:absolute;width:20px;height:20px;"
        "border-color:#202a3d;border-style:solid;"
    )
    return (
        '<div style="background:#0b0e14;border-radius:10px;'
        "min-height:340px;display:flex;flex-direction:column;"
        "align-items:center;justify-content:center;"
        'border:0.5px solid #1b2233;position:relative;padding:28px;">'
        f'<div style="{corner}top:14px;left:14px;border-width:2px 0 0 2px;"></div>'
        f'<div style="{corner}top:14px;right:14px;border-width:2px 2px 0 0;"></div>'
        f'<div style="{corner}bottom:14px;left:14px;border-width:0 0 2px 2px;"></div>'
        f'<div style="{corner}bottom:14px;right:14px;border-width:0 2px 2px 0;"></div>'
        '<div style="font-size:10px;color:#303c50;font-weight:600;'
        "letter-spacing:0.18em;text-transform:uppercase;"
        'margin-bottom:26px;font-family:monospace;">Sin señal de vídeo</div>'
        '<div style="display:flex;flex-direction:column;gap:10px;">'
        '<div style="font-size:11px;color:#31415a;">'
        "&mdash;&nbsp; Posicionate frente a la cámara</div>"
        '<div style="font-size:11px;color:#31415a;">'
        "&mdash;&nbsp; Distancia recomendada: 2 a 3 metros</div>"
        '<div style="font-size:11px;color:#31415a;">'
        "&mdash;&nbsp; Pies completamente visibles en el encuadre</div>"
        '<div style="font-size:11px;color:#31415a;">'
        "&mdash;&nbsp; Iluminación frontal uniforme, sin contraluz</div>"
        "</div></div>"
    )


# ---------------------------------------------------------------------------
# 1. CONFIGURACIÓN GLOBAL
# ---------------------------------------------------------------------------
st.set_page_config(page_title="MirrorUS", layout="wide")

st.markdown(
    "<style>.block-container{padding-top:2rem!important;}</style>",
    unsafe_allow_html=True,
)

if "startup_purged" not in st.session_state:
    for residual in glob.glob("./temp_*"):
        try:
            os.remove(residual)
        except OSError:
            pass
    st.session_state.startup_purged = True

# ---------------------------------------------------------------------------
# 2. INICIALIZACIÓN PERSISTENTE DE DETECTORES
# ---------------------------------------------------------------------------
if "detector" not in st.session_state:
    st.session_state.detector = PoseDetector()
if "depth_detector" not in st.session_state:
    st.session_state.depth_detector = DepthDetector()
if "valgus_detector" not in st.session_state:
    st.session_state.valgus_detector = KneeValgusDetector(threshold=0.08)
if "torso_detector" not in st.session_state:
    st.session_state.torso_detector = TorsoTiltDetector(max_tilt_deg=40.0)

if "analyzer" not in st.session_state:
    st.session_state.analyzer = SquatAnalyzer(
        depth_detector=st.session_state.depth_detector,
        detectors={
            "KNEE_VALGUS": st.session_state.valgus_detector,
            "TORSO_TILT": st.session_state.torso_detector,
        },
    )
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "last_valid_results" not in st.session_state:
    st.session_state.last_valid_results = None
if "min_angle_this_rep" not in st.session_state:
    st.session_state.min_angle_this_rep = 180.0
if "prev_fsm_state" not in st.session_state:
    st.session_state.prev_fsm_state = 0
# Resetea el checkbox de tracking si una mutación lo solicitó en el rerun anterior
if st.session_state.pop("_reset_run_btn", False):
    st.session_state.pop("run_btn", None)

# ---------------------------------------------------------------------------
# 3. SIDEBAR: CONFIGURACIÓN, FUENTE DE VÍDEO E INICIO DE TRACKING
# ---------------------------------------------------------------------------
is_local = detect_runtime_env()
video_file = None  # sobreescrito en el sidebar si source_mode == SOURCE_FILE

with st.sidebar:
    source_mode, skip_mode, d_thr, u_thr, t_thr = render_sidebar_config(
        is_local, disabled=st.session_state.get("run_btn", False)
    )

    if source_mode == SOURCE_FILE:
        video_file = st.file_uploader(
            "Sube un vídeo de tu sentadilla",
            type=["mp4", "mov", "avi"],
            label_visibility="collapsed",
            disabled=st.session_state.get("run_btn", False),
        )

    file_missing = source_mode == SOURCE_FILE and not video_file

    run = st.checkbox("▶  Iniciar Seguimiento", key="run_btn", disabled=file_missing)

st.session_state.depth_detector.down_threshold = d_thr
st.session_state.depth_detector.up_threshold = u_thr
st.session_state.torso_detector.max_tilt_deg = t_thr

# ---------------------------------------------------------------------------
# 4. CABECERA
# ---------------------------------------------------------------------------
render_header_and_instructions(is_local, source_mode)

# ---------------------------------------------------------------------------
# 5. PROCESAMIENTO DE LA FUENTE MULTIMEDIA
# ---------------------------------------------------------------------------
input_path = None
do_flip = True

if source_mode == SOURCE_FILE:
    if video_file:
        f_ext = os.path.splitext(video_file.name)[1]
        f_sign = hashlib.md5(video_file.name.encode()).hexdigest()[:6]
        input_path = f"./temp_{st.session_state.session_id}_{f_sign}{f_ext}"
        if not os.path.exists(input_path):
            with open(input_path, "wb") as f:
                f.write(video_file.read())
        do_flip = False
else:
    input_path = 0
    do_flip = True

# ---------------------------------------------------------------------------
# 6. SENSOR DE MUTACIÓN
# ---------------------------------------------------------------------------
if "prev_source" not in st.session_state:
    st.session_state.prev_source = source_mode
if "prev_path" not in st.session_state:
    st.session_state.prev_path = input_path

if (
    source_mode != st.session_state.prev_source
    or input_path != st.session_state.prev_path
):
    st.session_state["_reset_run_btn"] = True
    if "cap" in st.session_state:
        st.session_state.cap.release()
        del st.session_state.cap
    st.session_state.last_valid_results = None
    st.session_state.analyzer.reset_counters()
    st.session_state.detector.reset_filters()
    st.session_state.min_angle_this_rep = 180.0
    st.session_state.prev_fsm_state = 0
    if st.session_state.prev_path and isinstance(st.session_state.prev_path, str):
        handle_video_cleanup(st.session_state.prev_path)
    st.session_state.prev_source = source_mode
    st.session_state.prev_path = input_path
    st.rerun()

# ---------------------------------------------------------------------------
# 7. LAYOUT PRINCIPAL
# ---------------------------------------------------------------------------
col_panel, col_video = st.columns([0.38, 0.62])
with col_panel:
    left_placeholder = st.empty()
with col_video:
    frame_placeholder = st.empty()
    depth_indicator_ph = st.empty()

bio_placeholder = st.empty()

if st.button("☰ Ver historial analítico de la serie", disabled=run):
    show_history_modal(st.session_state.analyzer.history)

render_left_panel(
    left_placeholder,
    fsm_state=0,
    rep_valid=st.session_state.analyzer.count_valid,
    rep_invalid=st.session_state.analyzer.count_invalid,
    descent_sec=0.0,
    ascent_sec=0.0,
)
render_bio_metrics(bio_placeholder, 180.0, 0.0, 0.0, d_thr, u_thr, t_thr)
depth_indicator_ph.markdown(_depth_indicator_html(0.0, 0), unsafe_allow_html=True)

if file_missing:
    with frame_placeholder.container():
        st.warning("Sube un vídeo en el panel lateral para continuar.")
elif not run:
    frame_placeholder.markdown(_video_placeholder_html(), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 8. BUCLE DE TRACKING
# ---------------------------------------------------------------------------
if run and not file_missing:
    backend = cv2.CAP_DSHOW if source_mode == SOURCE_CAMERA else cv2.CAP_ANY

    if "cap" not in st.session_state:
        st.session_state.cap = cv2.VideoCapture(input_path, backend)
        if source_mode == SOURCE_CAMERA:
            st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            st.session_state.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cap = st.session_state.cap
    if source_mode == SOURCE_CAMERA:
        loop_delay = 0.01
    else:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0 or np.isnan(video_fps):
            video_fps = 30.0
        loop_delay = 1.0 / video_fps

    _mp_pose = st.session_state.detector.mp_pose
    frame_idx = 0
    video_start_time = None

    try:
        while run:
            start_time = time.time()
            if frame_idx == 0 and source_mode != SOURCE_CAMERA:
                video_start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                if source_mode == SOURCE_FILE:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = 0
                    video_start_time = time.time()
                    continue
                break

            if do_flip:
                frame = cv2.flip(frame, 1)

            should_skip = False
            if skip_mode == SKIP_BALANCED:
                should_skip = frame_idx % 3 == 2
            elif skip_mode == SKIP_PERFORMANCE:
                should_skip = frame_idx % 2 == 1

            if not should_skip:
                results = st.session_state.detector.extract_landmarks(frame)
                st.session_state.last_valid_results = results
            else:
                results = st.session_state.last_valid_results
                if results is None:
                    results = st.session_state.detector.extract_landmarks(frame)
                    st.session_state.last_valid_results = results

            payload = st.session_state.analyzer.process_frame(results.world)
            current_state = payload["fsm_state"]
            history = payload["session_history"]
            metrics = payload["metrics"]
            frame_errors = payload["current_frame_errors"]
            angle = metrics.get("knee_angle", 180.0)

            if st.session_state.prev_fsm_state == 0 and current_state == 1:
                st.session_state.min_angle_this_rep = angle
            if current_state in (1, 2, 3):
                st.session_state.min_angle_this_rep = min(
                    angle, st.session_state.min_angle_this_rep
                )
            # Reset del mínimo solo tras rep válida; en NO_DEPTH persiste
            # para mostrar el ángulo alcanzado en el indicador de profundidad.
            if (
                st.session_state.prev_fsm_state in (1, 2, 3)
                and current_state == 0
                and history
                and history[-1]["valid"]
            ):
                st.session_state.min_angle_this_rep = 180.0
            st.session_state.prev_fsm_state = current_state

            show_max_hint = (
                current_state == 0
                and bool(history)
                and not history[-1]["valid"]
                and "NO_DEPTH" in history[-1]["errors"]
            )

            render_left_panel(
                left_placeholder,
                fsm_state=current_state,
                rep_valid=payload["rep_valid_count"],
                rep_invalid=payload["rep_invalid_count"],
                descent_sec=metrics.get("descent_duration_sec", 0.0),
                ascent_sec=metrics.get("ascent_duration_sec", 0.0),
            )
            render_bio_metrics(
                bio_placeholder,
                angle,
                metrics.get("torso_tilt_deg", 0.0),
                metrics.get("valgus_ratio", 0.0),
                d_thr,
                u_thr,
                t_thr,
            )

            h_orig, w_orig = frame.shape[:2]
            display_h = 480
            display_w = int(display_h * (w_orig / h_orig))
            frame_display = cv2.resize(
                frame, (display_w, display_h), interpolation=cv2.INTER_AREA
            )

            # Determinar modo visual del esqueleto según estado FSM e historial
            if current_state in (1, 2, 3):
                active_frame_errors = {k for k, v in frame_errors.items() if v}
                if active_frame_errors:
                    skel_base = _COLOR_BLUE_DIM
                    ring_i, plain_i, conn_j = _resolve_error_visuals(
                        active_frame_errors, _mp_pose
                    )
                else:
                    skel_base = _COLOR_BLUE
                    ring_i, plain_i, conn_j = set(), set(), set()
            else:  # current_state == 0
                if not history:
                    skel_base = _COLOR_BLUE
                    ring_i, plain_i, conn_j = set(), set(), set()
                elif history[-1]["valid"]:
                    skel_base = _COLOR_GREEN
                    ring_i, plain_i, conn_j = set(), set(), set()
                else:
                    skel_base = _COLOR_ORANGE
                    ring_i, plain_i, conn_j = _resolve_error_visuals(
                        history[-1]["errors"], _mp_pose
                    )

            if results.world:
                _draw_skeleton(
                    frame_display,
                    results.raw.pose_landmarks,
                    _mp_pose,
                    skel_base,
                    ring_i,
                    plain_i,
                    conn_j,
                )

            frame_placeholder.image(frame_display, channels="BGR", width="stretch")

            range_angle = max(u_thr - d_thr, 1)
            progress = float(np.clip((u_thr - angle) / range_angle, 0.0, 1.0))
            min_progress = float(
                np.clip(
                    (u_thr - st.session_state.min_angle_this_rep) / range_angle,
                    0.0,
                    1.0,
                )
            )
            depth_indicator_ph.markdown(
                _depth_indicator_html(
                    progress,
                    current_state,
                    min_progress,
                    show_max_hint,
                    min_angle_deg=st.session_state.min_angle_this_rep,
                ),
                unsafe_allow_html=True,
            )

            frame_idx += 1

            # Compensación dinámica de timing
            if source_mode == SOURCE_CAMERA:
                elapsed = time.time() - start_time
                time.sleep(max(0.001, loop_delay - elapsed))
            else:
                elapsed = time.time() - video_start_time
                expected = int(elapsed * video_fps)
                current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                behind = expected - current_pos
                if behind > 0:
                    for _ in range(behind):
                        cap.grab()
                    frame_idx += behind
                else:
                    sleep_t = (current_pos / video_fps) - elapsed
                    if sleep_t > 0.005:
                        time.sleep(sleep_t)

    finally:
        cap.release()
        if "cap" in st.session_state:
            del st.session_state.cap
        handle_video_cleanup(input_path)

else:
    if "cap" in st.session_state:
        st.session_state.cap.release()
        del st.session_state.cap
    if not file_missing:
        handle_video_cleanup(input_path)
