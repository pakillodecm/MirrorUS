import glob
import hashlib
import os
import time
import uuid

import cv2
import numpy as np
import streamlit as st

# --- COMPONENTES ARQUITECTÓNICOS ---
from src.logic.pose_detector import PoseDetector
from src.logic.squat_counter import SquatCounter
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
if "counter" not in st.session_state:
    st.session_state.counter = SquatCounter()
if "valgus_detector" not in st.session_state:
    st.session_state.valgus_detector = KneeValgusDetector(threshold=0.90)
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "last_valid_results" not in st.session_state:
    st.session_state.last_valid_results = None

# Variables de estado para el ciclo de vida de la repetición actual
if "prev_count" not in st.session_state:
    st.session_state.prev_count = 0
if "current_rep_had_valgus" not in st.session_state:
    st.session_state.current_rep_had_valgus = False
if "feedback_message" not in st.session_state:
    st.session_state.feedback_message = "Sistema listo. Esperando inicio..."

is_local = detect_runtime_env()

with st.sidebar:
    source_mode, skip_mode = render_sidebar_config(
        is_local, st.session_state.session_id
    )

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

# Si el usuario cambia la radio o sube otro vídeo, se activa el protocolo de apagado
if (
    source_mode != st.session_state.prev_source
    or input_path != st.session_state.prev_path
):
    st.session_state.run_btn = False  # Forzamos el apagado del checkbox en la memoria

    if "cap" in st.session_state:
        st.session_state.cap.release()
        del st.session_state.cap
    st.session_state.last_valid_results = None
    st.session_state.prev_count = 0
    st.session_state.current_rep_had_valgus = False
    st.session_state.feedback_message = "Sistema listo. Esperando inicio..."

    # Destrucción física inmediata del archivo binario saliente
    if st.session_state.prev_path and isinstance(st.session_state.prev_path, str):
        handle_video_cleanup(st.session_state.prev_path)

    # Sincronizamos el histórico para el próximo ciclo
    st.session_state.prev_source = source_mode
    st.session_state.prev_path = input_path
    st.rerun()  # Reinicio inmediato para reflejar el apagado visual

# --- CONTROLADOR DE VISIBILIDAD REACTIVO POR UX ---
if source_mode == "Archivo de vídeo (Debug)" and not video_file:
    st.warning("Por favor, sube un archivo de vídeo para continuar.")
else:
    # Vinculamos el checkbox a la clave controlada por nuestro sensor de mutación
    run = st.checkbox("🔥 Iniciar Seguimiento", key="run_btn")

    # Cuadro de texto dinámico para feedback biomecánico en la pantalla principal
    feedback_placeholder = st.empty()
    feedback_placeholder.info(st.session_state.feedback_message)

    frame_placeholder = st.empty()

    # 2. CONTROLADOR DE FLUJO (ORQUESTADOR ESTABLE CON AUTO-RESET)
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

        frame_idx = 0  # Inicializador de fotogramas secuenciales
        video_start_time = None  # Ancla de tiempo absoluto de reproducción

        while run:
            start_time = time.time()  # Ancla de tiempo para compensación dinámica

            # Inicialización del reloj absoluto en el primer fotograma real
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
                # 1 -> flip horizontal (efecto espejo), 0 -> flip vertical, -1 -> ambos
                frame = cv2.flip(frame, 1)

            # 2.1 Algoritmo Adaptativo de Salto de Frames
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
            current_state = st.session_state.counter.state

            # --- EVALUACIÓN DE VALGO EN PARALELO ---
            if results.world:
                is_valgus, ratio = st.session_state.valgus_detector.analyze(
                    results.world
                )

                # Monitoreamos el colapso solo en las fases críticas: DEEP o ASCENDING
                if is_valgus and current_state in [2, 3]:
                    st.session_state.current_rep_had_valgus = True

            # Lógica de cierre de la repetición: detección del incremento del contador
            if count > st.session_state.prev_count:
                m = f"Repetición {count}"
                if st.session_state.current_rep_had_valgus:
                    m = f"⚠️ {m} completada con ¡FALLO DE VALGO! Corrige las rodillas."
                    st.session_state.feedback_message = m
                else:
                    m = f"✅ {m} excelente. Buen control y alineación."
                    st.session_state.feedback_message = m

                # Actualizamos interfaz y reseteamos variables para siguiente repetición
                if st.session_state.current_rep_had_valgus:
                    feedback_placeholder.error(st.session_state.feedback_message)
                else:
                    feedback_placeholder.success(st.session_state.feedback_message)
                st.session_state.prev_count = count
                st.session_state.current_rep_had_valgus = False
            else:
                # Actualización de feedback continuo en tiempo de ejecución
                if current_state == 0:
                    feedback_placeholder.info(st.session_state.feedback_message)
                elif current_state == 1:
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

            # 2.4 Pintar Capas Gráficas
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
