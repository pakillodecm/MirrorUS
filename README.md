# ✦ MirrorUS — Análisis Biomecánico de Sentadilla

**Trabajo Fin de Grado · Ingeniería del Software · ETSII Universidad de Sevilla**
Autor: Francisco de Castro Mañas · Tutora: Diana Borrego

🔗 **Aplicación desplegada:** [mirrorus.streamlit.app](https://mirrorus.streamlit.app/)

---

## Descripción

MirrorUS es un sistema de análisis biomecánico en tiempo real para la sentadilla (squat). A partir de la señal de vídeo de una cámara convencional, detecta la pose del atleta fotograma a fotograma mediante MediaPipe, aplica una máquina de estados finitos (FSM) para segmentar cada repetición y evalúa tres indicadores posturales:

- **Profundidad de rodilla** — ángulo ponderado por visibilidad entre muslo y pantorrilla.
- **Inclinación de torso** — desviación del vector torso respecto a la vertical.
- **Colapso medial de rodilla (valgo)** — desviación perpendicular de la rodilla respecto al eje cadera-tobillo, invariante ante sentadillas sumo o con stance ancho.

Los resultados se presentan en una interfaz Streamlit con retroalimentación visual en tiempo real: esqueleto coloreado por estado de error, barra de profundidad, métricas biomecánicas e historial analítico de la serie.

---

## Características principales

- Detección de pose con **MediaPipe Pose** (model complexity 1, coordenadas world métricas)
- Filtrado de ruido con **filtro One Euro** por articulación
- FSM de 4 estados: Reposo → Bajando → Zona profunda → Subiendo
- Timeout automático de repetición para movimientos abortados
- Detección de errores por fotograma: valgo de rodilla, inclinación de torso
- Detección de errores por repetición: sin profundidad, colapso en ascenso
- Telemetría VBT (Velocity Based Training): duración de bajada y subida
- Historial analítico con ángulo de flexión, errores y valores de referencia
- Soporte para cámara en vivo y vídeo subido
- Interfaz completamente en español
- Cobertura de tests del 100 %

---

## Requisitos del sistema

- **Python 3.12**
- **Poetry** (gestor de dependencias) Instrucciones de instalación en https://python-poetry.org/docs/#installation
> **Nota:** tras instalar Poetry, abre una terminal nueva para que el comando `poetry` esté disponible en el PATH. En Windows, si `poetry` no se reconoce, consulta la sección de instalación de la documentación oficial.
- Dependencias del sistema (necesarias para OpenCV y MediaPipe):
  - Linux: `libgl1`, `ffmpeg`
  - Windows/macOS: instaladas automáticamente con los paquetes

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/pakillodecm/TFG-Pose-Tracking.git
cd TFG-Pose-Tracking

# 2. Instalar dependencias con Poetry
poetry install
```

---

## Ejecución

```bash
poetry run streamlit run app.py
```

La aplicación se abre automáticamente en `http://localhost:8501`.

### Configuración del sidebar

| Parámetro | Rango | Defecto | Descripción |
|-----------|-------|---------|-------------|
| Fuente | Cámara / Archivo | Cámara | Origen del vídeo |
| Modo IA | Alta precisión / Equilibrado / Máx. rendimiento | Alta precisión | Fracción de fotogramas procesados |
| Profundidad (°) | 80–120° | 95° | Ángulo de rodilla para considerar paralelo roto |
| Erguido (°) | 130–170° | 155° | Ángulo de rodilla para considerar posición erguida |
| Torso (°) | 25–55° | 40° | Ángulo máximo de inclinación de torso |

---

## Tests

```bash
# Ejecutar todos los tests con cobertura
poetry run pytest

# Solo tests sin cobertura (más rápido)
poetry run pytest --no-cov
```

La suite completa cubre el 100 % de las líneas de `src/`. Los tests de lógica son puramente unitarios y no requieren cámara ni GPU.

---

## Arquitectura

```
tfg/
├── app.py                  # Orquestador principal (Streamlit)
├── src/
│   ├── logic/              # Núcleo de análisis (sin dependencias de UI)
│   │   ├── angles.py       # Cálculo de ángulos 3D
│   │   ├── depth_detector.py   # Detector de profundidad de rodilla
│   │   ├── filters.py      # Filtro One Euro
│   │   ├── pose_detector.py    # Wrapper MediaPipe + filtrado
│   │   ├── squat_analyzer.py   # FSM, telemetría y registro de repeticiones
│   │   ├── torso_detector.py   # Detector de inclinación de torso
│   │   └── valgus_detector.py  # Detector de colapso medial de rodilla
│   └── ui/
│       └── components.py   # Componentes y constantes de la interfaz
└── tests/                  # Tests unitarios (espejo de src/)
```

**Principio de separación:** `src/logic/` no importa nada de Streamlit ni OpenCV. El dibujo del esqueleto y toda la lógica de UI residen en `app.py` y `src/ui/`. Esto permite testear el núcleo analítico de forma aislada y rápida.

**Flujo de datos por fotograma:**

```
Frame BGR → PoseDetector → world landmarks (filtrados)
         → SquatAnalyzer.process_frame() → FramePayload
         → render_left_panel / render_bio_metrics / _draw_skeleton
```

---

## Stack tecnológico

| Componente | Tecnología |
|-----------|-----------|
| Interfaz | Streamlit ≥ 1.35 |
| Detección de pose | MediaPipe 0.10.14 |
| Visión artificial | OpenCV (headless) |
| Cómputo numérico | NumPy < 2.0 |
| Filtrado de señal | One Euro Filter (implementación propia) |
| Tests | pytest + pytest-cov |
| Linting | black, flake8, isort |
| CI | GitHub Actions |

---

## Limitaciones conocidas

- Requiere que el atleta sea completamente visible en el encuadre (de pies a cabeza).
- El análisis de valgo asume que el plano frontal del atleta es aproximadamente paralelo al plano de la cámara.
- El rendimiento depende del hardware: en equipos sin GPU se recomienda el modo "Equilibrado" o "Máx. rendimiento".
- Desplegado en [Streamlit Community Cloud](https://mirrorus.streamlit.app/) con funcionalidad analítica completa. La visualización del vídeo en vivo presenta limitaciones en el entorno cloud: la cámara no es accesible (el servidor carece de dispositivo de captura) y la previsualización del vídeo subido no se renderiza, si bien el análisis, las métricas y el historial operan con normalidad. El uso con cámara en vivo requiere ejecución local.
