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

_ERROR_LANDMARK_MAP = {
    "KNEE_VALGUS": ["LEFT_KNEE", "RIGHT_KNEE"],
    "TORSO_TILT": ["LEFT_SHOULDER", "RIGHT_SHOULDER"],
}

_REP_ERROR_NAMES = {"NO_DEPTH", "MID_ASCENT_COLLAPSE"}
_REP_ERROR_KEY_JOINTS = ["LEFT_KNEE", "RIGHT_KNEE"]

_DEPTH_STATE_CONFIG = {
    0: ("#9aa1ab", "Reposo"),
    1: ("#d97706", "⬇ Bajando"),
    2: ("#16a34a", "✓ Zona profunda"),
    3: ("#0066cc", "⬆ Subiendo"),
}


def _draw_skeleton_frame(frame, landmarks, mp_pose, frame_errors):
    """Dibuja el esqueleto con codificación de color por error de fotograma.

    Verde sin errores. Articulaciones con error en rojo con anillo blanco.
    Conexiones que tocan articulaciones con error en rojo. El resto del
    esqueleto se atenúa a gris para dirigir la atención al foco del problema.

    Args:
        frame: Frame BGR de OpenCV.
        landmarks: pose_landmarks del resultado raw de MediaPipe.
        mp_pose: mp.solutions.pose accesible desde el detector persistente.
        frame_errors: dict {nombre_error: bool} del payload del analizador.
    """
    h, w = frame.shape[:2]
    error_indices = set()
    for err_name, is_active in frame_errors.items():
        if is_active and err_name in _ERROR_LANDMARK_MAP:
            for lm_name in _ERROR_LANDMARK_MAP[err_name]:
                error_indices.add(mp_pose.PoseLandmark[lm_name].value)

    has_errors = bool(error_indices)
    ok_color = (74, 163, 22)
    err_color = (0, 0, 220)
    dim_color = (140, 140, 140)
    lm_list = landmarks.landmark

    for conn in mp_pose.POSE_CONNECTIONS:
        s, e = tuple(conn)
        lm_s, lm_e = lm_list[s], lm_list[e]
        if lm_s.visibility < 0.3 or lm_e.visibility < 0.3:
            continue
        pt_s = (int(lm_s.x * w), int(lm_s.y * h))
        pt_e = (int(lm_e.x * w), int(lm_e.y * h))
        if s in error_indices or e in error_indices:
            cv2.line(frame, pt_s, pt_e, err_color, 3, cv2.LINE_AA)
        else:
            color = dim_color if has_errors else ok_color
            cv2.line(frame, pt_s, pt_e, color, 2, cv2.LINE_AA)

    for idx, lm in enumerate(lm_list):
        if lm.visibility < 0.3:
            continue
        cx, cy = int(lm.x * w), int(lm.y * h)
        if idx in error_indices:
            cv2.circle(frame, (cx, cy), 8, err_color, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 11, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            color = dim_color if has_errors else ok_color
            cv2.circle(frame, (cx, cy), 5, color, -1, cv2.LINE_AA)


def _draw_skeleton_rep_error(frame, landmarks, mp_pose):
    """Dibuja esqueleto naranja con rodillas en rojo: señal de error de repetición.

    Activo en FSM=0 cuando la última rep registró NO_DEPTH o MID_ASCENT_COLLAPSE.
    El naranja distingue este modo del error de fotograma (rojo puro).

    Args:
        frame: Frame BGR de OpenCV.
        landmarks: pose_landmarks del resultado raw de MediaPipe.
        mp_pose: mp.solutions.pose accesible desde el detector persistente.
    """
    h, w = frame.shape[:2]
    orange_bgr = (0, 140, 255)
    red_bgr = (0, 0, 220)
    key_indices = {
        mp_pose.PoseLandmark[lm_name].value for lm_name in _REP_ERROR_KEY_JOINTS
    }
    lm_list = landmarks.landmark

    for conn in mp_pose.POSE_CONNECTIONS:
        s, e = tuple(conn)
        lm_s, lm_e = lm_list[s], lm_list[e]
        if lm_s.visibility < 0.3 or lm_e.visibility < 0.3:
            continue
        pt_s = (int(lm_s.x * w), int(lm_s.y * h))
        pt_e = (int(lm_e.x * w), int(lm_e.y * h))
        cv2.line(frame, pt_s, pt_e, orange_bgr, 2, cv2.LINE_AA)

    for idx, lm in enumerate(lm_list):
        if lm.visibility < 0.3:
            continue
        cx, cy = int(lm.x * w), int(lm.y * h)
        if idx in key_indices:
            cv2.circle(frame, (cx, cy), 8, red_bgr, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 11, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.circle(frame, (cx, cy), 5, orange_bgr, -1, cv2.LINE_AA)


def _depth_indicator_html(
    progress: float,
    fsm_state: int,
    min_progress: float = 0.0,
    show_max_hint: bool = False,
) -> str:
    """Genera la barra de profundidad HTML coloreada por fase FSM con marca de mínimo.

    st.progress no permite cambiar de color, de ahí el HTML directo. La marca
    vertical indica la profundidad máxima alcanzada esta repetición.

    Args:
        progress: Progreso actual de profundidad en [0.0, 1.0].
        fsm_state: Estado actual de la FSM (0-3).
        min_progress: Profundidad máxima alcanzada esta rep, en [0.0, 1.0].
        show_max_hint: Si True muestra el máximo en el label (activo tras NO_DEPTH).

    Returns:
        HTML listo para st.markdown con unsafe_allow_html=True.
    """
    color, label = _DEPTH_STATE_CONFIG.get(fsm_state, ("#9aa1ab", "—"))
    pct = int(progress * 100)
    min_pct = int(min_progress * 100)

    if show_max_hint and min_pct > 0:
        right_label = (
            f'<span style="color:#d97706;font-weight:500;">'
            f"máx.&nbsp;{min_pct}%&nbsp;·&nbsp;sin profundidad</span>"
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
        'background:#374151;border-radius:1px;"></div></div>'
        if min_pct > 2
        else ""
    )

    return (
        '<div style="margin-top:6px;">'
        '<div style="display:flex;justify-content:space-between;'
        f'font-size:11px;color:#6b7280;margin-bottom:4px;">'
        f"<span>Profundidad</span>{right_label}</div>"
        '<div style="height:6px;background:#e1e4e9;border-radius:3px;overflow:hidden;">'
        f'<div style="width:{pct}%;height:6px;background:{color};'
        'border-radius:3px;"></div>'
        f"</div>{min_tick}</div>"
    )


def _video_placeholder_html() -> str:
    """Card oscura con esquinas de visor e instrucciones de posicionamiento.

    Returns:
        HTML listo para st.markdown con unsafe_allow_html=True.
    """
    corner = (
        "position:absolute;width:20px;height:20px;"
        "border-color:#1e2738;border-style:solid;"
    )
    return (
        '<div style="background:#0a0d12;border-radius:9px;'
        "min-height:340px;display:flex;flex-direction:column;"
        "align-items:center;justify-content:center;"
        'border:0.5px solid #1a2030;position:relative;padding:28px;">'
        f'<div style="{corner}top:14px;left:14px;border-width:2px 0 0 2px;"></div>'
        f'<div style="{corner}top:14px;right:14px;border-width:2px 2px 0 0;"></div>'
        f'<div style="{corner}bottom:14px;left:14px;border-width:0 0 2px 2px;"></div>'
        f'<div style="{corner}bottom:14px;right:14px;border-width:0 2px 2px 0;"></div>'
        '<div style="font-size:10px;color:#2e3848;font-weight:600;'
        "letter-spacing:0.18em;text-transform:uppercase;"
        'margin-bottom:26px;font-family:monospace;">Sin señal de vídeo</div>'
        '<div style="display:flex;flex-direction:column;gap:10px;">'
        '<div style="font-size:11px;color:#2e3d52;">'
        "&mdash;&nbsp; Posicionate frente a la cámara</div>"
        '<div style="font-size:11px;color:#2e3d52;">'
        "&mdash;&nbsp; Distancia recomendada: 2 a 3 metros</div>"
        '<div style="font-size:11px;color:#2e3d52;">'
        "&mdash;&nbsp; Pies completamente visibles en el encuadre</div>"
        '<div style="font-size:11px;color:#2e3d52;">'
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
    st.session_state.valgus_detector = KneeValgusDetector(threshold=0.90)
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

# ---------------------------------------------------------------------------
# 3. SIDEBAR Y SINCRONIZACIÓN
# ---------------------------------------------------------------------------
is_local = detect_runtime_env()

with st.sidebar:
    source_mode, skip_mode, d_thr, u_thr, t_thr = render_sidebar_config(
        is_local, disabled=st.session_state.get("run_btn", False)
    )

st.session_state.depth_detector.down_threshold = d_thr
st.session_state.depth_detector.up_threshold = u_thr
st.session_state.torso_detector.max_tilt_deg = t_thr

# ---------------------------------------------------------------------------
# 4. CABECERA
# ---------------------------------------------------------------------------
render_header_and_instructions(is_local, source_mode)

# ---------------------------------------------------------------------------
# 5. GESTIÓN DE ENTRADA MULTIMEDIA
# ---------------------------------------------------------------------------
video_file = None
input_path = None
do_flip = True

if source_mode == SOURCE_FILE:
    video_file = st.file_uploader(
        "Sube un vídeo de tu sentadilla",
        type=["mp4", "mov", "avi"],
        label_visibility="collapsed",
    )
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

file_missing = source_mode == SOURCE_FILE and not video_file

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
    st.session_state.run_btn = False
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
run = st.checkbox("🔥 Iniciar Seguimiento", key="run_btn", disabled=file_missing)

col_panel, col_video = st.columns([0.38, 0.62])
with col_panel:
    left_placeholder = st.empty()
with col_video:
    frame_placeholder = st.empty()
    depth_indicator_ph = st.empty()

bio_placeholder = st.empty()

if st.button("📊 Ver historial analítico de la serie", disabled=run):
    show_history_modal(st.session_state.analyzer.history)

render_left_panel(
    left_placeholder,
    fsm_state=0,
    rep_valid=st.session_state.analyzer.count_valid,
    rep_invalid=st.session_state.analyzer.count_invalid,
    descent_sec=0.0,
    ascent_sec=0.0,
)
render_bio_metrics(bio_placeholder, 180.0, 0.0, 1.0)
depth_indicator_ph.markdown(_depth_indicator_html(0.0, 0), unsafe_allow_html=True)

if file_missing:
    with frame_placeholder.container():
        st.warning("Sube un vídeo usando el selector de arriba para continuar.")
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
            if (
                st.session_state.prev_fsm_state in (1, 2, 3)
                and current_state == 0
                and history
                and history[-1]["valid"]
            ):
                st.session_state.min_angle_this_rep = 180.0
            st.session_state.prev_fsm_state = current_state
            rep_error_active = (
                current_state == 0
                and bool(history)
                and not history[-1]["valid"]
                and bool(_REP_ERROR_NAMES.intersection(history[-1]["errors"]))
            )
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
                metrics.get("valgus_ratio", 1.0),
            )

            h_orig, w_orig = frame.shape[:2]
            display_h = 480
            display_w = int(display_h * (w_orig / h_orig))
            frame_display = cv2.resize(
                frame, (display_w, display_h), interpolation=cv2.INTER_AREA
            )

            if results.world:
                if rep_error_active:
                    _draw_skeleton_rep_error(
                        frame_display, results.raw.pose_landmarks, _mp_pose
                    )
                else:
                    _draw_skeleton_frame(
                        frame_display,
                        results.raw.pose_landmarks,
                        _mp_pose,
                        frame_errors,
                    )

            frame_placeholder.image(
                frame_display, channels="BGR", use_container_width=True
            )

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
                    progress, current_state, min_progress, show_max_hint
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
