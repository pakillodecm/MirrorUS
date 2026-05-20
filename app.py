import time
import uuid

import cv2
import numpy as np
import streamlit as st

# --- COMPONENTES ARQUITECTÓNICOS ---
from src.logic.pose_detector import PoseDetector
from src.logic.squat_counter import SquatCounter
from src.ui.components import (
    detect_runtime_env,
    handle_video_cleanup,
    render_header_and_instructions,
    render_sidebar_config,
)

# 1. CONFIGURACIÓN E INICIALIZACIÓN PERSISTENTE
st.set_page_config(page_title="MirrorUS", layout="centered")

if "detector" not in st.session_state:
    st.session_state.detector = PoseDetector()
if "counter" not in st.session_state:
    st.session_state.counter = SquatCounter()
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "last_valid_results" not in st.session_state:
    st.session_state.last_valid_results = None

is_local = detect_runtime_env()

with st.sidebar:
    source_mode, input_path, do_flip, skip_mode = render_sidebar_config(
        is_local, st.session_state.session_id
    )

render_header_and_instructions(is_local, source_mode)

run = st.checkbox("🔥 Iniciar Seguimiento")
frame_placeholder = st.empty()

# 2. CONTROLADOR DE FLUJO (ORQUESTADOR ADAPTATIVO)
if run:
    backend = cv2.CAP_DSHOW if source_mode == "Cámara en vivo" else cv2.CAP_ANY
    cap = cv2.VideoCapture(input_path, backend)

    if source_mode == "Cámara en vivo":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        loop_delay = 0.01
    else:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0 or np.isnan(video_fps):
            video_fps = 30.0
        loop_delay = 1.0 / video_fps

    frame_idx = 0  # Inicializador de fotogramas secuenciales

    while run:
        start_time = time.time()  # Ancla de tiempo para compensación dinámica

        ret, frame = cap.read()
        if not ret:
            if source_mode == "Archivo de vídeo (Debug)":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        if do_flip:
            # 1 -> flip horizontal (efecto espejo), 0 -> flip vertical, -1 -> ambos ejes
            frame = cv2.flip(frame, 1)

        # 2.1 Algoritmo Adaptativo de Salto de Frames (Variable Frame Scheduler)
        should_skip = False
        if skip_mode == "Equilibrado (66% IA)":
            if frame_idx % 3 == 2:  # Patrón 1 1 0 1 1 0
                should_skip = True
        elif skip_mode == "Máximo Rendimiento (50% IA)":
            if frame_idx % 2 == 1:  # Patrón 1 0 1 0 1 0
                should_skip = True

        # 2.2 Gestión de Inferencia vs Retención de Orden Cero (Zero-Order Hold)
        if not should_skip:
            results = st.session_state.detector.extract_landmarks(frame)
            st.session_state.last_valid_results = results
        else:
            results = st.session_state.last_valid_results
            if results is None:
                # Inferencia de rescate obligatoria si el primer frame requiere skip
                results = st.session_state.detector.extract_landmarks(frame)
                st.session_state.last_valid_results = results

        # La FSM se alimenta de la postura persistida para no perder continuidad
        count = st.session_state.counter.update(results.world)

        # 2.3 Preparación del frame de visualización
        h_orig, w_orig = frame.shape[:2]
        display_h = 480
        display_w = int(display_h * (w_orig / h_orig))
        frame_display = cv2.resize(
            frame, (display_w, display_h), interpolation=cv2.INTER_AREA
        )

        h, w = display_h, display_w
        rect_w, rect_h = int(w * 0.4), 60
        rect_x = (w // 2) - (rect_w // 2)

        d_thr = st.session_state.counter.thr_down
        u_thr = st.session_state.counter.thr_up

        # 2.4 Pintar Capas Gráficas Recuperadas
        if results.world:
            angle = st.session_state.counter.last_angle
            st.session_state.detector.draw_landmarks(frame_display, results.raw)

            # --- UI: BARRA DE PROFUNDIDAD ---
            range_angle = u_thr - d_thr
            progress = np.clip((u_thr - angle) / range_angle, 0.0, 1.0)

            bar_w, bar_x = int(w * 0.08), int(w * 0.05)
            bar_y_top, bar_y_bottom = int(h * 0.25), int(h * 0.85)
            bar_height = bar_y_bottom - bar_y_top

            actual_state = st.session_state.counter.state
            if actual_state == 2:
                color = (0, 255, 0)  # Verde: Profundidad conseguida
            elif actual_state in [1, 3]:
                color = (0, 255, 255)  # Amarillo: En transición
            else:
                color = (0, 0, 255)  # Rojo: Bloqueo/Reposo

            # Fondo gris de la barra
            cv2.rectangle(
                frame_display,
                (bar_x, bar_y_top),
                (bar_x + bar_w, bar_y_bottom),
                (40, 40, 40),
                -1,
            )
            # Relleno dinámico proporcional
            fill_level = int(bar_y_bottom - (progress * bar_height))
            cv2.rectangle(
                frame_display,
                (bar_x, fill_level),
                (bar_x + bar_w, bar_y_bottom),
                color,
                -1,
            )
            # Texto porcentual
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

        # --- UI: CONTADOR DE REPETICIONES PERSISTENTE ---
        cv2.rectangle(
            frame_display,
            (rect_x, 10),
            (rect_x + rect_w, 10 + rect_h),
            (0, 0, 0),
            -1,
        )
        rep_color = (
            (0, 255, 0) if st.session_state.counter.state == 2 else (0, 255, 255)
        )
        cv2.putText(
            frame_display,
            f"REPS: {count}",
            (rect_x + 20, 10 + 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            rep_color,
            3,
            cv2.LINE_AA,
        )

        frame_placeholder.image(frame_display, channels="BGR", width="stretch")

        frame_idx += 1  # Incremento de control indexado

        # 2.5 Reloj de Compensación Dinámica (Evita la cámara lenta)
        elapsed = time.time() - start_time
        time.sleep(max(0.001, loop_delay - elapsed))

    cap.release()
    handle_video_cleanup(input_path)
else:
    handle_video_cleanup(input_path)
    st.info("Activa el checkbox superior para poner en marcha el detector de pose.")
