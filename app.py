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

# Sidebar para ajustes
with st.sidebar:
    st.header("Ajustes")
    d_thr = st.slider("Umbral Profundidad", 60, 110, 90)
    u_thr = st.slider("Umbral Erguido", 140, 180, 160)

    # Actualizamos los umbrales en el objeto del session_state
    st.session_state.counter.thr_down = d_thr
    st.session_state.counter.thr_up = u_thr

run = st.checkbox("🔥 Iniciar Cámara")
frame_placeholder = st.empty()

if run:
    # CAP_DSHOW es vital en Windows para evitar latencia de inicio
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # 1. RESOLUCIÓN Y FPS
    # Por defecto suele ser 640x480 (4:3). 640x360 (16:9) mejora el aspect ratio.
    # 1280x720 tiene mucho más detalle pero puede sobrecargar la CPU.
    # Forzamos 30 FPS. Si la luz es mala, la cámara bajará los FPS para exponer más.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("No se puede acceder a la cámara")
            break

        # 1 -> flip horizontal (efecto espejo), 0 -> flip vertical, -1 -> ambos ejes.
        frame = cv2.flip(frame, 1)

        # Procesamiento
        results = st.session_state.detector.extract_landmarks(frame)

        if results.world:
            count = st.session_state.counter.update(results.world)
            st.session_state.detector.draw_landmarks(frame, results.raw)

            # UI sobre el vídeo (Cálculos en el frame original)
            cv2.rectangle(frame, (30, 15), (380, 105), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"REPS: {count}",
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 255),
                6,
                cv2.LINE_AA,
            )

        # --- EL TRUCO DE INGENIERÍA ---
        # Redimensionamos a 480p para que Streamlit lo renderice fluido
        # pero la IA ha trabajado con la calidad de 720p.
        frame_display = cv2.resize(frame, (640, 360))

        # Mostrar en Streamlit
        frame_placeholder.image(frame_display, channels="BGR", use_container_width=True)

        # Para poder frenar el bucle al desmarcar el checkbox
        # Streamlit no refresca los widgets dentro de un while de forma nativa,
        # pero este es el flujo que tenías definido.
        if not run:
            break

    cap.release()
else:
    st.info("Activa el checkbox para empezar.")
