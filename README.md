# 🏋️‍♂️ MirrorUS

> Sistema inteligente de análisis biomecánico de sentadillas en tiempo real.
> Trabajo Fin de Grado — Universidad de Sevilla

MirrorUS utiliza visión por computador y estimación de pose (MediaPipe) para
analizar la técnica de sentadilla en tiempo real, detectando fallos posturales
como el valgo de rodilla o la inclinación excesiva del torso.

## Características

- Detección de profundidad de sentadilla (rotura del paralelo)
- Detección de valgo de rodilla en coordenadas métricas reales
- Detección de inclinación del torso
- Telemetría de velocidad por fases (VBT): duración de bajada y subida
- Soporte para cámara en vivo y análisis de vídeo pregrabado
- Historial analítico de la sesión exportable

## Arquitectura

```
src/
├── logic/          # Lógica pura, sin dependencias de UI
│   ├── angles.py           # Cálculo de ángulos biomecánicos
│   ├── depth_detector.py   # Sensor de profundidad de sentadilla
│   ├── valgus_detector.py  # Detector de valgo de rodilla
│   ├── torso_detector.py   # Detector de inclinación del torso
│   ├── squat_analyzer.py   # Orquestador FSM + telemetría VBT
│   ├── pose_detector.py    # Wrapper de MediaPipe + filtrado
│   └── filters.py          # Filtro One Euro para suavizado
└── ui/
    └── components.py       # Componentes Streamlit desacoplados
```

## Instalación

Requiere Python 3.12 y [Poetry](https://python-poetry.org/).

```bash
git clone https://github.com/tu-usuario/mirrorus.git
cd mirrorus
poetry install
```

## Uso

```bash
poetry run streamlit run app.py
```

## Tests

```bash
poetry run pytest
```

## Tecnologías

- [MediaPipe](https://mediapipe.dev/) — Estimación de pose
- [OpenCV](https://opencv.org/) — Captura y procesado de vídeo
- [Streamlit](https://streamlit.io/) — Interfaz web
- [NumPy](https://numpy.org/) — Cálculo vectorial
- [Poetry](https://python-poetry.org/) — Gestión de dependencias

## Autor

Francisco de Castro Mañas — frademann@alum.us.es
