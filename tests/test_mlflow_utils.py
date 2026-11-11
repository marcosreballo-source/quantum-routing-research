import pytest
from unittest.mock import patch, MagicMock

from DRL_Lei.utils.mlflow_run import flatten_dict, log_config_to_mlflow, SafeMLflowRun


class TestFlattenDict:
    def test_flat_dict(self):
        assert flatten_dict({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_nested(self):
        result = flatten_dict({"x": {"y": 1, "z": 2}})
        assert result == {"x.y": 1, "x.z": 2}

    def test_deeply_nested(self):
        result = flatten_dict({"a": {"b": {"c": 3}}})
        assert result == {"a.b.c": 3}

    def test_empty(self):
        assert flatten_dict({}) == {}

    def test_mixed(self):
        result = flatten_dict({"a": 1, "b": {"c": 2}})
        assert result == {"a": 1, "b.c": 2}


class TestLogConfigToMlflow:
    @patch("DRL_Lei.utils.mlflow_run.mlflow")
    def test_calls_log_params(self, mock_mlflow):
        cfg = {"lr": 0.001, "model": {"dim": 64}}
        log_config_to_mlflow(cfg)
        mock_mlflow.log_params.assert_called_once_with({"lr": 0.001, "model.dim": 64})


class TestSafeMLflowRun:
    @patch("DRL_Lei.utils.mlflow_run.mlflow")
    def test_context_manager_calls_start_end(self, mock_mlflow):
        mock_mlflow.start_run.return_value = MagicMock()
        with SafeMLflowRun(run_name="test_run"):
            pass
        mock_mlflow.start_run.assert_called_once_with(run_name="test_run")
        mock_mlflow.end_run.assert_called_once_with(status="FINISHED")

    @patch("DRL_Lei.utils.mlflow_run.mlflow")
    def test_sets_failed_on_exception(self, mock_mlflow):
        mock_mlflow.start_run.return_value = MagicMock()
        with pytest.raises(ValueError):
            with SafeMLflowRun(run_name="fail_run"):
                raise ValueError("boom")
        mock_mlflow.end_run.assert_called_once_with(status="FAILED")
