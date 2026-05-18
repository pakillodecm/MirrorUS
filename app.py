import os
import tempfile
import time
import warnings

import cv2
import numpy as np
import streamlit as st

from src.logic.pose_detector import PoseDetector
from src.logic.squat_counter import SquatCounter

# --- SILENCIAR LOGS Y AVISOS DE CONSOLA ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_LOGGING_LEVEL"] = "3"
warnings.filterwarnings(
    "ignore", category=UserWarning, module="google.protobuf.symbol_database"
)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(page_title="MirrorUS", layout="centered")

if "detector" not in st.session_state:
    st.session_state.detector = PoseDetector()
if "counter" not in st.session_state:
    st.session_state.counter = SquatCounter()

st.title("🏋️‍♂️ MirrorUS")

# --- SIDEBAR: CONFIGURACIÓN Y DEBUG ---
with st.sidebar:
    st.header("⚙️ Configuración")
    source_mode = st.radio(
        "Fuente de entrada:", ["Cámara en vivo", "Archivo de vídeo (Debug)"]
    )

    input_path = 0
    do_flip = True

    if source_mode == "Archivo de vídeo (Debug)":
        video_file = st.file_uploader("Sube un vídeo", type=["mp4", "mov", "avi"])
        if video_file:
            _, file_extension = os.path.splitext(video_file.name)
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=file_extension
            ) as tfile:
                tfile.write(video_file.read())
                input_path = tfile.name
            do_flip = False
        else:
            st.warning("Sube un vídeo para continuar.")
            st.stop()

    st.divider()
    st.subheader("Umbrales")
    d_thr = st.slider("Umbral Profundidad", 60, 110, 90)
    u_thr = st.slider("Umbral Erguido", 140, 180, 160)

    st.session_state.counter.thr_down = d_thr
    st.session_state.counter.thr_up = u_thr

run = st.checkbox("🔥 Iniciar Seguimiento")
frame_placeholder = st.empty()

# --- LÓGICA DE EJECUCIÓN ---
if run:
    backend = cv2.CAP_DSHOW if source_mode == "Cámara en vivo" else cv2.CAP_ANY
    cap = cv2.VideoCapture(input_path, backend)

    if source_mode == "Cámara en vivo":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        loop_delay = 1.0 / 30.0
    else:
        # Calcular delay real basado en los FPS del vídeo
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0 or np.isnan(video_fps):
            video_fps = 30.0
        loop_delay = 1.0 / video_fps

    while run:
        ret, frame = cap.read()
        if not ret:
            if source_mode == "Archivo de vídeo (Debug)":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    st.error("Error crítico: No se puede leer el archivo de vídeo.")
                    break
            else:
                break

        if do_flip:
            # 1 -> flip horizontal (efecto espejo), 0 -> flip vertical, -1 -> ambos ejes
            frame = cv2.flip(frame, 1)

        # 1. PROCESAMIENTO GENERAL
        results = st.session_state.detector.extract_landmarks(frame)
        count = st.session_state.counter.update(results.world)

        # 2. PREPARAR FRAME DE VISUALIZACIÓN
        h_orig, w_orig = frame.shape[:2]
        display_h = 480
        display_w = int(display_h * (w_orig / h_orig))
        frame_display = cv2.resize(
            frame, (display_w, display_h), interpolation=cv2.INTER_AREA
        )

        h, w = display_h, display_w
        rect_w, rect_h = int(w * 0.4), 60
        rect_x = (w // 2) - (rect_w // 2)

        # 3. DIBUJAR CAPA DE CONTROLES BIOMÉTRICOS SI HAY DETECCIÓN
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

        # 4. UI: CONTADOR DE REPETICIONES PERSISTENTE
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

        # Enviar el frame a Streamlit
        frame_placeholder.image(frame_display, channels="BGR", width="stretch")

        # Delay imperativo de flujo para estabilización en la nube
        time.sleep(loop_delay)

        if not run:
            break

    cap.release()
    if source_mode == "Archivo de vídeo (Debug)" and os.path.exists(input_path):
        try:
            os.remove(input_path)
        except Exception:
            pass
else:
    st.info(
        "Configura la fuente en el panel izquierdo y activa el checkbox para empezar."
    )
