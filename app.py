import glob
import hashlib
import os
import time
import uuid

import cv2
import numpy as np
import streamlit as st

# --- COMPONENTES ARQUITECTÓNICOS ---
from src.logic.depth_detector import DepthDetector
from src.logic.pose_detector import PoseDetector
from src.logic.squat_analyzer import SquatAnalyzer
from src.logic.valgus_detector import KneeValgusDetector
from src.ui.components import (
    detect_runtime_env,
    handle_video_cleanup,
    render_header_and_instructions,
    render_sidebar_config,
)

# 1. CONFIGURACIÓN E INICIALIZACIÓN PERSISTENTE
st.set_page_config(page_title="MirrorUS", layout="centered")

# Protocolo preventivo de limpieza en el arranque de la aplicación
if "startup_purged" not in st.session_state:
    for residual_file in glob.glob("./temp_*"):
        try:
            os.remove(residual_file)
        except OSError:
            pass
    st.session_state.startup_purged = True

if "detector" not in st.session_state:
    st.session_state.detector = PoseDetector()
if "depth_detector" not in st.session_state:
    st.session_state.depth_detector = DepthDetector()
if "valgus_detector" not in st.session_state:
    st.session_state.valgus_detector = KneeValgusDetector(threshold=0.90)
if "analyzer" not in st.session_state:
    st.session_state.analyzer = SquatAnalyzer(
        depth_detector=st.session_state.depth_detector,
        detectors={"KNEE_VALGUS": st.session_state.valgus_detector},
    )
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "last_valid_results" not in st.session_state:
    st.session_state.last_valid_results = None

# Sincronización del estado visual y feedback
if "last_rep_feedback" not in st.session_state:
    st.session_state.last_rep_feedback = {
        "text": "Esperando repeticiones...",
        "type": "info",
    }

is_local = detect_runtime_env()

with st.sidebar:
    source_mode, skip_mode, d_thr, u_thr = render_sidebar_config(
        is_local, st.session_state.session_id
    )

# Sincronización dinámica de los sliders del panel lateral con el detector geométrico
st.session_state.depth_detector.down_threshold = d_thr
st.session_state.depth_detector.up_threshold = u_thr

render_header_and_instructions(is_local, source_mode)

# --- GESTIÓN CENTRALIZADA DE ENTRADAS MULTIMEDIA ---
input_path = None
do_flip = True
video_file = None

if source_mode == "Archivo de vídeo (Debug)":
    video_file = st.file_uploader(
        "Sube un vídeo de tu sentadilla", type=["mp4", "mov", "avi"]
    )
    if video_file:
        f_extension = os.path.splitext(video_file.name)[1]
        f_sign = hashlib.md5(video_file.name.encode()).hexdigest()[:6]
        input_path = f"./temp_{st.session_state.session_id}_{f_sign}{f_extension}"

        if not os.path.exists(input_path):
            with open(input_path, "wb") as f:
                f.write(video_file.read())
        do_flip = False
else:
    input_path = 0
    do_flip = True

# --- SENSOR DE MUTACIÓN ADAPTATIVO ---
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
    st.session_state.last_rep_feedback = {
        "text": "Sistema listo. Esperando inicio...",
        "type": "info",
    }

    if st.session_state.prev_path and isinstance(st.session_state.prev_path, str):
        handle_video_cleanup(st.session_state.prev_path)

    st.session_state.prev_source = source_mode
    st.session_state.prev_path = input_path
    st.rerun()

# --- CONTROLADOR DE VISIBILIDAD REACTIVO POR UX ---
if source_mode == "Archivo de vídeo (Debug)" and not video_file:
    st.warning("Por favor, sube un archivo de vídeo para continuar.")
else:
    run = st.checkbox("🔥 Iniciar Seguimiento", key="run_btn")

    # Contenedores dinámicos del Front
    feedback_placeholder = st.empty()
    metrics_cols = st.columns(2)
    rep_valid_metric = metrics_cols[0].empty()
    rep_invalid_metric = metrics_cols[1].empty()

    frame_placeholder = st.empty()

    # Marcadores estáticos iniciales para evitar parpadeos
    rep_valid_metric.metric(
        "👍 REPETICIONES VÁLIDAS", st.session_state.analyzer.count_valid
    )
    rep_invalid_metric.metric(
        "❌ REPETICIONES CON FALLO", st.session_state.analyzer.count_invalid
    )

    if st.session_state.last_rep_feedback["type"] == "error":
        feedback_placeholder.error(st.session_state.last_rep_feedback["text"])
    elif st.session_state.last_rep_feedback["type"] == "success":
        feedback_placeholder.success(st.session_state.last_rep_feedback["text"])
    else:
        feedback_placeholder.info(st.session_state.last_rep_feedback["text"])

    # 2. CONTROLADOR DE FLUJO (ORQUESTADOR ESTABLE)
    if run:
        backend = cv2.CAP_DSHOW if source_mode == "Cámara en vivo" else cv2.CAP_ANY

        if "cap" not in st.session_state:
            st.session_state.cap = cv2.VideoCapture(input_path, backend)
            if source_mode == "Cámara en vivo":
                st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                st.session_state.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        cap = st.session_state.cap

        if source_mode == "Cámara en vivo":
            loop_delay = 0.01
        else:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps <= 0 or np.isnan(video_fps):
                video_fps = 30.0
            loop_delay = 1.0 / video_fps

        frame_idx = 0
        video_start_time = None
        last_history_len = len(st.session_state.analyzer.history)

        while run:
            start_time = time.time()

            if frame_idx == 0 and source_mode != "Cámara en vivo":
                video_start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                if source_mode == "Archivo de vídeo (Debug)":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = 0
                    video_start_time = time.time()
                    continue
                else:
                    break

            if do_flip:
                frame = cv2.flip(frame, 1)

            should_skip = False
            if skip_mode == "Equilibrado (66% IA)":
                if frame_idx % 3 == 2:  # Patrón 1 1 0 1 1 0
                    should_skip = True
            elif skip_mode == "Máximo Rendimiento (50% IA)":
                if frame_idx % 2 == 1:  # Patrón 1 0 1 0 1 0
                    should_skip = True

            if not should_skip:
                results = st.session_state.detector.extract_landmarks(frame)
                st.session_state.last_valid_results = results
            else:
                results = st.session_state.last_valid_results
                if results is None:
                    results = st.session_state.detector.extract_landmarks(frame)
                    st.session_state.last_valid_results = results

            # CONSUMO DEL BACKEND A TRAVÉS DE UN ÚNICO PAYLOAD GENÉRICO
            payload = st.session_state.analyzer.process_frame(results.world)
            current_state = payload["fsm_state"]
            current_history = payload["session_history"]

            # Actualización en tiempo real de marcadores de rendimiento en el Front
            rep_valid_metric.metric(
                "👍 REPETICIONES VÁLIDAS", payload["rep_valid_count"]
            )
            rep_invalid_metric.metric(
                "❌ REPETICIONES CON FALLO", payload["rep_invalid_count"]
            )

            if len(current_history) > last_history_len:
                last_rep = current_history[-1]
                if last_rep["valid"]:
                    st.session_state.last_rep_feedback = {
                        "text": f"✅ Repetición {last_rep['rep']} excelente.",
                        "type": "success",
                    }
                    feedback_placeholder.success(
                        st.session_state.last_rep_feedback["text"]
                    )
                else:
                    errors = ", ".join(last_rep["errors"])
                    st.session_state.last_rep_feedback = {
                        "text": f"⚠️ Repetición {last_rep['rep']} fallida: {errors}.",
                        "type": "error",
                    }
                    feedback_placeholder.error(
                        st.session_state.last_rep_feedback["text"]
                    )
                last_history_len = len(current_history)
            else:
                # Guías dinámicas de ejecución continua
                if current_state == 1:
                    feedback_placeholder.warning(
                        "⬇️ Descendiendo... Mantén las rodillas hacia fuera."
                    )
                elif current_state == 2:
                    feedback_placeholder.warning(
                        "🏋️‍♂️ Zona Profunda alcanzada. ¡Fuerza hacia arriba!"
                    )
                elif current_state == 3:
                    feedback_placeholder.warning(
                        "⬆️ Ascendiendo... Controla el plano frontal."
                    )

            # Preparación y pintado de la capa gráfica
            h_orig, w_orig = frame.shape[:2]
            display_h = 480
            display_w = int(display_h * (w_orig / h_orig))
            frame_display = cv2.resize(
                frame, (display_w, display_h), interpolation=cv2.INTER_AREA
            )

            if results.world:
                # El esqueleto cambia a ROJO si hay un fallo de valgo en el frame exacto
                if payload["current_frame_errors"].get("KNEE_VALGUS", False):
                    # dibujo alternativo/manipulación visual en esqueleto -> TODO
                    # De momento pintamos landmarks estándar.
                    st.session_state.detector.draw_landmarks(frame_display, results.raw)
                else:
                    st.session_state.detector.draw_landmarks(frame_display, results.raw)

                # Capa de la barra lateral sagital
                angle = payload["metrics"]["knee_angle"]
                range_angle = u_thr - d_thr
                progress = np.clip((u_thr - angle) / range_angle, 0.0, 1.0)

                bar_w, bar_x = int(display_w * 0.08), int(display_w * 0.05)
                bar_y_top, bar_y_bottom = int(display_h * 0.25), int(display_h * 0.85)
                bar_height = bar_y_bottom - bar_y_top

                if current_state == 2:
                    color = (0, 255, 0)  # Verde: Profundidad conseguida
                elif current_state in [1, 3]:
                    color = (0, 255, 255)  # Amarillo: En transición
                else:
                    color = (0, 0, 255)  # Rojo: Bloqueo/Reposo

                cv2.rectangle(
                    frame_display,
                    (bar_x, bar_y_top),
                    (bar_x + bar_w, bar_y_bottom),
                    (40, 40, 40),
                    -1,
                )
                fill_level = int(bar_y_bottom - (progress * bar_height))
                cv2.rectangle(
                    frame_display,
                    (bar_x, fill_level),
                    (bar_x + bar_w, bar_y_bottom),
                    color,
                    -1,
                )
                cv2.putText(
                    frame_display,
                    f"{int(progress * 100)}%",
                    (bar_x, bar_y_top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            frame_placeholder.image(frame_display, channels="BGR", width="stretch")
            frame_idx += 1

            # Reloj de Compensación Dinámica
            if source_mode == "Cámara en vivo":
                elapsed = time.time() - start_time
                time.sleep(max(0.001, loop_delay - elapsed))
            elif source_mode == "Archivo de vídeo (Debug)":
                expected_timeline = frame_idx * loop_delay
                actual_timeline = time.time() - video_start_time

                if actual_timeline > expected_timeline:
                    frames_to_skip = int(
                        (actual_timeline - expected_timeline) / loop_delay
                    )
                    if frames_to_skip > 0:
                        for _ in range(frames_to_skip):
                            cap.grab()
                        frame_idx += frames_to_skip
                else:
                    time.sleep(max(0.001, expected_timeline - actual_timeline))

        cap.release()
        if "cap" in st.session_state:
            del st.session_state.cap
        handle_video_cleanup(input_path)
    else:
        if "cap" in st.session_state:
            st.session_state.cap.release()
            del st.session_state.cap
        handle_video_cleanup(input_path)

    # --- HISTORIAL DESPLEGABLE DE RENDIMIENTO DE SESIÓN ---
    if st.session_state.analyzer.history:
        st.divider()
        st.subheader("📊 Historial Analítico de la Serie")
        st.table(st.session_state.analyzer.history)
