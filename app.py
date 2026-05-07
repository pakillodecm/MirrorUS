import cv2
import streamlit as st

from src.logic.pose_detector import PoseDetector
from src.logic.squat_counter import SquatCounter

# Configuración inicial
st.set_page_config(page_title="MirrorUS", layout="centered")

if "detector" not in st.session_state:
    st.session_state.detector = PoseDetector()
if "counter" not in st.session_state:
    st.session_state.counter = SquatCounter()

st.title("🏋️‍♂️ MirrorUS")

# Sidebar para ajustes en tiempo real
with st.sidebar:
    st.header("Ajustes")
    d_thr = st.slider("Umbral Profundidad", 60, 110, 90)
    u_thr = st.slider("Umbral Erguido", 140, 180, 160)
    st.session_state.counter.thr_down = d_thr
    st.session_state.counter.thr_up = u_thr

run = st.checkbox("🔥 Iniciar Cámara")
frame_placeholder = st.empty()

if run:
    # 0 es la cámara por defecto en Windows
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("No se puede acceder a la cámara")
            break

        # Procesamiento
        results = st.session_state.detector.extract_landmarks(frame)

        if results.world:
            count = st.session_state.counter.update(results.world)
            st.session_state.detector.draw_landmarks(frame, results.raw)

            # UI sobre el vídeo
            cv2.putText(
                frame,
                f"REPS: {count}",
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                3,
            )

        # Mostrar en Streamlit
        frame_placeholder.image(frame, channels="BGR", use_container_width=True)

        if not run:
            break

    cap.release()
else:
    st.info("Activa el checkbox para empezar.")
