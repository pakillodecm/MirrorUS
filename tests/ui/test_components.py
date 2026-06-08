import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.ui.components import (
    SOURCE_CAMERA,
    SOURCE_FILE,
    _build_history_df,
    _clamp01,
    _knee_semaphore,
    _torso_semaphore,
    _valgus_semaphore,
    _vbt_caption,
    detect_runtime_env,
    handle_video_cleanup,
    render_bio_metrics,
    render_header_and_instructions,
    render_left_panel,
    render_sidebar_config,
)


class TestClamp01:
    def test_below_zero(self):
        assert _clamp01(-5.0) == 0.0

    def test_above_one(self):
        assert _clamp01(2.5) == 1.0

    def test_zero_boundary(self):
        assert _clamp01(0.0) == 0.0

    def test_one_boundary(self):
        assert _clamp01(1.0) == 1.0

    def test_midpoint(self):
        assert _clamp01(0.5) == 0.5


class TestKneeSemaphore:
    def test_optimal(self):
        assert _knee_semaphore(80.0) == "✓ Paralelo óptimo"

    def test_boundary_optimal(self):
        assert _knee_semaphore(85.0) == "✓ Paralelo óptimo"

    def test_broken(self):
        assert _knee_semaphore(87.0) == "✓ Paralelo roto"

    def test_boundary_parallel(self):
        assert _knee_semaphore(90.0) == "✓ Paralelo roto"

    def test_descending(self):
        assert _knee_semaphore(110.0) == "⬇ En descenso"

    def test_initial(self):
        assert _knee_semaphore(160.0) == "— Posición inicial"


class TestTorsoSemaphore:
    def test_optimal(self):
        assert _torso_semaphore(20.0) == "✓ Inclinación óptima"

    def test_boundary_optimal(self):
        assert _torso_semaphore(30.0) == "✓ Inclinación óptima"

    def test_near_limit(self):
        assert _torso_semaphore(35.0) == "⚠ Cerca del límite"

    def test_boundary_limit(self):
        assert _torso_semaphore(40.0) == "⚠ Cerca del límite"

    def test_excessive(self):
        assert _torso_semaphore(50.0) == "✗ Inclinación excesiva"


class TestValgusSemaphore:
    def test_correct(self):
        assert _valgus_semaphore(1.0) == "✓ Alineación correcta"

    def test_boundary_good(self):
        assert _valgus_semaphore(0.90) == "✓ Alineación correcta"

    def test_alert(self):
        assert _valgus_semaphore(0.87) == "⚠ Zona de alerta"

    def test_boundary_alert(self):
        assert _valgus_semaphore(0.85) == "⚠ Zona de alerta"

    def test_valgus(self):
        assert _valgus_semaphore(0.80) == "✗ Valgo detectado"


class TestVbtCaption:
    def test_no_data(self):
        r = _vbt_caption(0.0, 1.5, 3.0)
        assert "1.5" in r and "3.0" in r

    def test_optimal(self):
        assert _vbt_caption(2.0, 1.5, 3.0) == "✓ Óptimo"

    def test_boundary_low(self):
        assert _vbt_caption(1.5, 1.5, 3.0) == "✓ Óptimo"

    def test_boundary_high(self):
        assert _vbt_caption(3.0, 1.5, 3.0) == "✓ Óptimo"

    def test_too_slow(self):
        assert _vbt_caption(4.0, 1.5, 3.0) == "⚠ Fuera de rango"

    def test_too_fast(self):
        assert _vbt_caption(0.5, 1.5, 3.0) == "⚠ Fuera de rango"


class TestBuildHistoryDf:
    def test_empty_returns_empty_df(self):
        df = _build_history_df([])
        assert isinstance(df, pd.DataFrame) and len(df) == 0

    def test_columns(self):
        df = _build_history_df(
            [
                {
                    "rep": 1,
                    "valid": True,
                    "errors": [],
                    "descent_duration_sec": 2.0,
                    "ascent_duration_sec": 1.5,
                }
            ]
        )
        assert list(df.columns) == [
            "Rep",
            "Estado",
            "Errores",
            "Bajada (s)",
            "Subida (s)",
        ]

    def test_valid_rep_labels(self):
        df = _build_history_df(
            [
                {
                    "rep": 1,
                    "valid": True,
                    "errors": [],
                    "descent_duration_sec": 2.1,
                    "ascent_duration_sec": 1.8,
                }
            ]
        )
        assert df.iloc[0]["Estado"] == "Válida"
        assert df.iloc[0]["Errores"] == "—"

    def test_invalid_rep_with_errors(self):
        df = _build_history_df(
            [
                {
                    "rep": 2,
                    "valid": False,
                    "errors": ["KNEE_VALGUS", "NO_DEPTH"],
                    "descent_duration_sec": 1.0,
                    "ascent_duration_sec": 0.0,
                }
            ]
        )
        assert df.iloc[0]["Estado"] == "Fallo"
        assert "KNEE_VALGUS" in df.iloc[0]["Errores"]

    def test_durations_rounded(self):
        df = _build_history_df(
            [
                {
                    "rep": 1,
                    "valid": True,
                    "errors": [],
                    "descent_duration_sec": 2.1234,
                    "ascent_duration_sec": 1.8765,
                }
            ]
        )
        assert df.iloc[0]["Bajada (s)"] == 2.12
        assert df.iloc[0]["Subida (s)"] == 1.88

    def test_multiple_reps_count(self):
        history = [
            {
                "rep": i,
                "valid": True,
                "errors": [],
                "descent_duration_sec": 2.0,
                "ascent_duration_sec": 1.5,
            }
            for i in range(1, 6)
        ]
        assert len(_build_history_df(history)) == 5


class TestDetectRuntimeEnv:
    def test_local_returns_true(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.path.exists", return_value=False):
                assert detect_runtime_env() is True

    def test_cloud_env_var(self):
        with patch.dict(os.environ, {"STREAMLIT_SHARING_REPOSITORY": "repo"}):
            assert detect_runtime_env() is False

    def test_cloud_mount_path(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.path.exists", return_value=True):
                assert detect_runtime_env() is False


class TestHandleVideoCleanup:
    def test_removes_existing_file(self, tmp_path):
        f = tmp_path / "temp.mp4"
        f.write_bytes(b"data")
        handle_video_cleanup(str(f))
        assert not f.exists()

    def test_nonexistent_path_no_raise(self):
        handle_video_cleanup("/nonexistent/file.mp4")

    def test_integer_no_raise(self):
        handle_video_cleanup(0)

    def test_none_no_raise(self):
        handle_video_cleanup(None)

    def test_oserror_on_remove_is_silenced(self, tmp_path):
        f = tmp_path / "temp.mp4"
        f.write_bytes(b"data")
        with patch("src.ui.components.os.remove", side_effect=OSError):
            handle_video_cleanup(str(f))


def _make_placeholder():
    """Placeholder mock con soporte de context manager."""
    ph = MagicMock()
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=None)
    cm.__exit__ = MagicMock(return_value=False)
    ph.container.return_value = cm
    return ph


class TestRenderSidebarConfig:
    @patch("src.ui.components.st")
    def test_local_includes_camera(self, mock_st):
        mock_st.radio.return_value = SOURCE_CAMERA
        mock_st.selectbox.return_value = "Alta Precisión (100%)"
        mock_st.slider.side_effect = [90, 150, 40]
        render_sidebar_config(is_local=True)
        assert SOURCE_CAMERA in mock_st.radio.call_args[0][1]

    @patch("src.ui.components.st")
    def test_cloud_excludes_camera(self, mock_st):
        mock_st.radio.return_value = SOURCE_FILE
        mock_st.selectbox.return_value = "Alta Precisión (100%)"
        mock_st.slider.side_effect = [90, 150, 40]
        render_sidebar_config(is_local=False)
        assert SOURCE_CAMERA not in mock_st.radio.call_args[0][1]

    @patch("src.ui.components.st")
    def test_returns_five_values(self, mock_st):
        mock_st.radio.return_value = SOURCE_CAMERA
        mock_st.selectbox.return_value = "Alta Precisión (100%)"
        mock_st.slider.side_effect = [90, 150, 40]
        assert len(render_sidebar_config(is_local=True)) == 5

    @patch("src.ui.components.st")
    def test_slider_values_propagated(self, mock_st):
        mock_st.radio.return_value = SOURCE_CAMERA
        mock_st.selectbox.return_value = "Alta Precisión (100%)"
        mock_st.slider.side_effect = [85, 160, 35]
        _, _, d, u, t = render_sidebar_config(is_local=True)
        assert d == 85 and u == 160 and t == 35


class TestRenderHeaderAndInstructions:
    @patch("src.ui.components.st")
    def test_renders_title(self, mock_st):
        render_header_and_instructions(is_local=True, source_mode=SOURCE_CAMERA)
        assert "MirrorUS" in mock_st.markdown.call_args[0][0]

    @patch("src.ui.components.st")
    def test_cloud_camera_stops(self, mock_st):
        mock_st.stop.side_effect = SystemExit
        with pytest.raises(SystemExit):
            render_header_and_instructions(is_local=False, source_mode=SOURCE_CAMERA)
        mock_st.stop.assert_called_once()

    @patch("src.ui.components.st")
    def test_cloud_file_no_stop(self, mock_st):
        render_header_and_instructions(is_local=False, source_mode=SOURCE_FILE)
        mock_st.stop.assert_not_called()

    @patch("src.ui.components.st")
    def test_local_no_stop(self, mock_st):
        render_header_and_instructions(is_local=True, source_mode=SOURCE_CAMERA)
        mock_st.stop.assert_not_called()


class TestRenderLeftPanel:
    @patch("src.ui.components.st")
    def test_calls_container(self, mock_st):
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        render_left_panel(_make_placeholder(), 0, 0, 0, 0.0, 0.0)
        _make_placeholder().container.call_count  # just verifying call pattern

    @patch("src.ui.components.st")
    def test_all_states_render_metric(self, mock_st):
        for state in range(4):
            mock_st.reset_mock()
            mock_st.columns.return_value = [MagicMock(), MagicMock()]
            render_left_panel(_make_placeholder(), state, 0, 0, 0.0, 0.0)
            mock_st.metric.assert_called()

    @patch("src.ui.components.st")
    def test_rep_counters_on_columns(self, mock_st):
        cols = [MagicMock(), MagicMock()]
        mock_st.columns.return_value = cols
        render_left_panel(_make_placeholder(), 0, 5, 3, 0.0, 0.0)
        cols[0].metric.assert_any_call("👍 Reps válidas", 5)
        cols[1].metric.assert_any_call("❌ Con fallo", 3)

    @patch("src.ui.components.st")
    def test_descent_shown_when_positive(self, mock_st):
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        render_left_panel(_make_placeholder(), 1, 0, 0, 2.5, 0.0)
        mock_st.metric.assert_any_call("⬇ Bajada", "2.5 s")

    @patch("src.ui.components.st")
    def test_dash_shown_when_no_data(self, mock_st):
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        render_left_panel(_make_placeholder(), 0, 0, 0, 0.0, 0.0)
        mock_st.metric.assert_any_call("⬇ Bajada", "—")

    @patch("src.ui.components.st")
    def test_markdown_called_for_color_line(self, mock_st):
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        render_left_panel(_make_placeholder(), 2, 0, 0, 0.0, 0.0)
        mock_st.markdown.assert_called()


class TestRenderBioMetrics:
    @patch("src.ui.components.st")
    def test_three_metrics_rendered(self, mock_st):
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
        render_bio_metrics(_make_placeholder(), 90.0, 25.0, 0.95)
        assert mock_st.metric.call_count == 3

    @patch("src.ui.components.st")
    def test_three_progress_bars(self, mock_st):
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
        render_bio_metrics(_make_placeholder(), 90.0, 25.0, 0.95)
        assert mock_st.progress.call_count == 3

    @patch("src.ui.components.st")
    def test_knee_angle_in_output(self, mock_st):
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
        render_bio_metrics(_make_placeholder(), 87.0, 25.0, 0.95)
        assert "87.0" in str(mock_st.metric.call_args_list)

    @patch("src.ui.components.st")
    def test_progress_values_clamped(self, mock_st):
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
        render_bio_metrics(_make_placeholder(), 200.0, 90.0, 0.0)
        for call in mock_st.progress.call_args_list:
            assert 0.0 <= call[0][0] <= 1.0
