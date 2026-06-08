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
# MAPA DE ARTICULACIONES POR ERROR
# ---------------------------------------------------------------------------
_ERROR_LANDMARK_MAP = {
    "KNEE_VALGUS": ["LEFT_KNEE", "RIGHT_KNEE"],
    "TORSO_TILT": [
        "LEFT_SHOULDER",
        "RIGHT_SHOULDER",
        "LEFT_HIP",
        "RIGHT_HIP",
    ],
}

_DEPTH_STATE_CONFIG = {
    0: ("#9aa1ab", "Reposo"),
    1: ("#d97706", "⬇ Bajando"),
    2: ("#16a34a", "✓ Zona profunda"),
    3: ("#0066cc", "⬆ Subiendo"),
}


# ---------------------------------------------------------------------------
# HELPERS DE RENDERIZADO
# ---------------------------------------------------------------------------


def _draw_skeleton_refined(frame, landmarks, mp_pose, frame_errors):
    """Dibuja el esqueleto con codificación de color por error."""
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
            c = dim_color if has_errors else ok_color
            cv2.line(frame, pt_s, pt_e, c, 2, cv2.LINE_AA)

    for idx, lm in enumerate(lm_list):
        if lm.visibility < 0.3:
            continue
        cx, cy = int(lm.x * w), int(lm.y * h)
        if idx in error_indices:
            cv2.circle(frame, (cx, cy), 8, err_color, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 11, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            c = dim_color if has_errors else ok_color
            cv2.circle(frame, (cx, cy), 5, c, -1, cv2.LINE_AA)


def _depth_indicator_html(progress: float, fsm_state: int) -> str:
    """Genera el HTML del indicador de profundidad coloreado por fase FSM."""
    color, label = _DEPTH_STATE_CONFIG.get(fsm_state, ("#9aa1ab", "—"))
    pct = int(progress * 100)
    return (
        '<div style="margin-top:6px;">'
        '<div style="display:flex;justify-content:space-between;'
        f'font-size:11px;color:#6b7280;margin-bottom:4px;">'
        f"<span>Profundidad</span>"
        f'<span style="color:{color};font-weight:500;">'
        f"{pct}%&nbsp;·&nbsp;{label}</span>"
        "</div>"
        '<div style="height:6px;background:#e1e4e9;'
        'border-radius:3px;overflow:hidden;">'
        f'<div style="width:{pct}%;height:6px;'
        f'background:{color};border-radius:3px;"></div>'
        "</div>"
        "</div>"
    )


def _video_placeholder_html() -> str:
    """Card oscura con esquinas de visor e instrucciones de posicionamiento."""
    corner = (
        "position:absolute;width:20px;height:20px;"
        "border-color:#1e2738;border-style:solid;"
    )
    return (
        '<div style="background:#0a0d12;border-radius:9px;'
        "min-height:340px;display:flex;flex-direction:column;"
        "align-items:center;justify-content:center;"
        'border:0.5px solid #1a2030;position:relative;padding:28px;">'
        f'<div style="{corner}top:14px;left:14px;'
        'border-width:2px 0 0 2px;"></div>'
        f'<div style="{corner}top:14px;right:14px;'
        'border-width:2px 2px 0 0;"></div>'
        f'<div style="{corner}bottom:14px;left:14px;'
        'border-width:0 0 2px 2px;"></div>'
        f'<div style="{corner}bottom:14px;right:14px;'
        'border-width:0 2px 2px 0;"></div>'
        '<div style="font-size:10px;color:#2e3848;font-weight:600;'
        "letter-spacing:0.18em;text-transform:uppercase;"
        'margin-bottom:26px;font-family:monospace;">Sin señal de vídeo</div>'
        '<div style="display:flex;flex-direction:column;gap:10px;">'
        '<div style="font-size:11px;color:#2e3d52;letter-spacing:0.02em;">'
        "&mdash;&nbsp; Posicionate frente a la cámara</div>"
        '<div style="font-size:11px;color:#2e3d52;letter-spacing:0.02em;">'
        "&mdash;&nbsp; Distancia recomendada: 2 a 3 metros</div>"
        '<div style="font-size:11px;color:#2e3d52;letter-spacing:0.02em;">'
        "&mdash;&nbsp; Pies completamente visibles en el encuadre</div>"
        '<div style="font-size:11px;color:#2e3d52;letter-spacing:0.02em;">'
        "&mdash;&nbsp; Iluminación frontal uniforme, sin contraluz</div>"
        "</div>"
        "</div>"
    )


# ---------------------------------------------------------------------------
# 1. CONFIGURACIÓN GLOBAL
# ---------------------------------------------------------------------------
st.set_page_config(page_title="MirrorUS", layout="wide")

# Reducción del padding superior por defecto de Streamlit
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

# ---------------------------------------------------------------------------
# 3. SIDEBAR Y SINCRONIZACIÓN
# ---------------------------------------------------------------------------
is_local = detect_runtime_env()

with st.sidebar:
    source_mode, skip_mode, d_thr, u_thr, t_thr = render_sidebar_config(is_local)

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
    if st.session_state.prev_path and isinstance(st.session_state.prev_path, str):
        handle_video_cleanup(st.session_state.prev_path)
    st.session_state.prev_source = source_mode
    st.session_state.prev_path = input_path
    st.rerun()

# ---------------------------------------------------------------------------
# 7. LAYOUT PRINCIPAL
# ---------------------------------------------------------------------------
run = st.checkbox(
    "🔥 Iniciar Seguimiento",
    key="run_btn",
    disabled=file_missing,
)

col_panel, col_video = st.columns([0.38, 0.62])

with col_panel:
    left_placeholder = st.empty()

with col_video:
    frame_placeholder = st.empty()
    depth_indicator_ph = st.empty()

bio_placeholder = st.empty()

if st.button(
    "📊 Ver historial analítico de la serie",
):
    show_history_modal(st.session_state.analyzer.history)

# Render del estado inicial
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
else:
    if not run:
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
                else:
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
            metrics = payload["metrics"]
            frame_errors = payload["current_frame_errors"]

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
                metrics.get("knee_angle", 180.0),
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
                _draw_skeleton_refined(
                    frame_display,
                    results.raw.pose_landmarks,
                    _mp_pose,
                    frame_errors,
                )

            frame_placeholder.image(
                frame_display, channels="BGR", use_container_width=True
            )

            angle = metrics.get("knee_angle", 180.0)
            range_angle = max(u_thr - d_thr, 1)
            progress = float(np.clip((u_thr - angle) / range_angle, 0.0, 1.0))
            depth_indicator_ph.markdown(
                _depth_indicator_html(progress, current_state),
                unsafe_allow_html=True,
            )

            frame_idx += 1

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
