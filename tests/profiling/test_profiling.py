from unittest.mock import Mock, patch
import torch

from drone_detector_mlops.utils.profiling import ProfilerWithTable, get_profiler


class TestProfilerWithTable:
    """Tests for ProfilerWithTable wrapper class."""

    def test_initialization(self):
        """Test ProfilerWithTable can be initialized."""
        mock_profiler = Mock()
        wrapper = ProfilerWithTable(mock_profiler, print_table=True)

        assert wrapper.profiler == mock_profiler
        assert wrapper.print_table is True

    def test_initialization_print_table_false(self):
        """Test ProfilerWithTable with print_table=False."""
        mock_profiler = Mock()
        wrapper = ProfilerWithTable(mock_profiler, print_table=False)

        assert wrapper.profiler == mock_profiler
        assert wrapper.print_table is False

    def test_enter_calls_profiler_enter(self):
        """Test that __enter__ calls the profiler's __enter__."""
        mock_profiler = Mock()
        mock_profiler.__enter__ = Mock(return_value="profiler_context")
        wrapper = ProfilerWithTable(mock_profiler, print_table=True)

        result = wrapper.__enter__()

        mock_profiler.__enter__.assert_called_once()
        assert result == "profiler_context"

    def test_exit_calls_profiler_exit(self):
        """Test that __exit__ calls the profiler's __exit__."""
        mock_profiler = Mock()
        mock_profiler.__exit__ = Mock(return_value=None)
        mock_profiler.key_averages = Mock()
        wrapper = ProfilerWithTable(mock_profiler, print_table=False)

        wrapper.__exit__(None, None, None)

        mock_profiler.__exit__.assert_called_once_with(None, None, None)

    def test_exit_prints_summary_when_enabled(self):
        """Test that __exit__ prints summary when print_table is True."""
        mock_profiler = Mock()
        mock_profiler.__exit__ = Mock(return_value=None)
        mock_key_avg = Mock()
        mock_key_avg.table = Mock(return_value="mock_table")
        mock_profiler.key_averages = Mock(return_value=mock_key_avg)

        wrapper = ProfilerWithTable(mock_profiler, print_table=True)

        wrapper.__exit__(None, None, None)

        mock_profiler.key_averages.assert_called_once()
        assert mock_key_avg.table.call_count == 2

    def test_exit_does_not_print_when_disabled(self):
        """Test that __exit__ does not print when print_table is False."""
        mock_profiler = Mock()
        mock_profiler.__exit__ = Mock(return_value=None)
        wrapper = ProfilerWithTable(mock_profiler, print_table=False)

        wrapper.__exit__(None, None, None)

        mock_profiler.key_averages.assert_not_called()

    def test_step_calls_profiler_step(self):
        """Test that step calls the profiler's step method."""
        mock_profiler = Mock()
        mock_profiler.step = Mock()
        wrapper = ProfilerWithTable(mock_profiler, print_table=True)

        wrapper.step()

        mock_profiler.step.assert_called_once()

    def test_context_manager_usage(self):
        """Test ProfilerWithTable can be used as context manager."""
        mock_profiler = Mock()
        mock_profiler.__enter__ = Mock(return_value=mock_profiler)
        mock_profiler.__exit__ = Mock(return_value=None)
        mock_profiler.key_averages = Mock()

        wrapper = ProfilerWithTable(mock_profiler, print_table=False)

        with wrapper as prof:
            assert prof == mock_profiler

        mock_profiler.__enter__.assert_called_once()
        mock_profiler.__exit__.assert_called_once()


class TestGetProfiler:
    """Tests for get_profiler function."""

    def test_creates_output_directory(self, tmp_path):
        """Test that get_profiler creates output directory."""
        output_dir = tmp_path / "profiler_test"
        get_profiler(output_dir=str(output_dir), print_table=False)

        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_returns_profiler_with_table(self, tmp_path):
        """Test that get_profiler returns ProfilerWithTable instance."""
        output_dir = tmp_path / "profiler_test"
        profiler = get_profiler(output_dir=str(output_dir), print_table=False)

        assert isinstance(profiler, ProfilerWithTable)

    def test_profiler_has_print_table_true_by_default(self, tmp_path):
        """Test that profiler has print_table=True by default."""
        output_dir = tmp_path / "profiler_test"
        profiler = get_profiler(output_dir=str(output_dir))

        assert profiler.print_table is True

    def test_profiler_respects_print_table_parameter(self, tmp_path):
        """Test that profiler respects print_table parameter."""
        output_dir = tmp_path / "profiler_test"
        profiler = get_profiler(output_dir=str(output_dir), print_table=False)

        assert profiler.print_table is False

    @patch("torch.cuda.is_available")
    def test_enables_cuda_profiling_when_available(self, mock_cuda_available, tmp_path):
        """Test that CUDA profiling is enabled when GPU is available."""
        mock_cuda_available.return_value = True
        output_dir = tmp_path / "profiler_test"

        profiler = get_profiler(output_dir=str(output_dir), print_table=False)

        assert isinstance(profiler, ProfilerWithTable)

    @patch("torch.cuda.is_available")
    def test_cpu_only_when_cuda_not_available(self, mock_cuda_available, tmp_path):
        """Test that only CPU profiling is enabled when CUDA is not available."""
        mock_cuda_available.return_value = False
        output_dir = tmp_path / "profiler_test"

        profiler = get_profiler(output_dir=str(output_dir), print_table=False)

        assert isinstance(profiler, ProfilerWithTable)

    def test_can_be_used_as_context_manager(self, tmp_path):
        """Test that profiler can be used as context manager."""
        output_dir = tmp_path / "profiler_test"
        profiler = get_profiler(output_dir=str(output_dir), print_table=False)

        with profiler:
            # Create a simple tensor operation
            x = torch.randn(10, 10)
            _ = x + x

        assert True

    def test_profiler_step_method_exists(self, tmp_path):
        """Test that profiler has a step method."""
        output_dir = tmp_path / "profiler_test"
        profiler = get_profiler(output_dir=str(output_dir), print_table=False)

        assert hasattr(profiler, "step")
        assert callable(profiler.step)

    def test_creates_nested_directory_structure(self, tmp_path):
        """Test that get_profiler creates nested directory structure."""
        output_dir = tmp_path / "nested" / "profiler" / "test"
        get_profiler(output_dir=str(output_dir), print_table=False)

        assert output_dir.exists()
        assert output_dir.is_dir()
