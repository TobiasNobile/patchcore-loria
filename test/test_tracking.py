import pytest

from patchcore.tracking import _flatten_params, patchcore_run


def test_flatten_params_flat():
    assert _flatten_params({"a": 1, "b": "x"}) == {"a": "1", "b": "x"}


def test_flatten_params_nested():
    result = _flatten_params({"model": {"backbone": "resnet18", "patchsize": 3}})
    assert result == {"model.backbone": "resnet18", "model.patchsize": "3"}


def test_flatten_params_list():
    result = _flatten_params({"layers": ["layer1", "layer2"]})
    assert result == {"layers": "layer1,layer2"}


def test_patchcore_run_logs_params_and_metrics(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path}/mlruns.db"
    params = {"backbone": "resnet18", "patchsize": 3, "seed": 0}

    with patchcore_run(
        experiment="test_exp",
        run_name="bottle",
        params=params,
        tracking_uri=tracking_uri,
    ) as run:
        run.log_metrics({"instance_auroc": 0.98, "full_pixel_auroc": 0.95})

    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("test_exp")
    runs = client.search_runs(experiment.experiment_id)
    assert len(runs) == 1

    logged = runs[0]
    assert logged.data.params["backbone"] == "resnet18"
    assert logged.data.params["patchsize"] == "3"
    assert logged.data.metrics["instance_auroc"] == pytest.approx(0.98)
    assert logged.data.metrics["full_pixel_auroc"] == pytest.approx(0.95)


def test_patchcore_run_tags_failed_on_exception(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path}/mlruns.db"

    with pytest.raises(ValueError):
        with patchcore_run(
            experiment="test_exp",
            run_name="failing_run",
            params={},
            tracking_uri=tracking_uri,
        ):
            raise ValueError("training crashed")

    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("test_exp")
    runs = client.search_runs(experiment.experiment_id)
    assert runs[0].data.tags["run_status"] == "FAILED"


def test_patchcore_run_skips_missing_artifact(tmp_path, caplog):
    import logging
    tracking_uri = f"sqlite:///{tmp_path}/mlruns.db"

    with patchcore_run(
        experiment="test_exp",
        run_name="no_artifact",
        params={},
        tracking_uri=tracking_uri,
    ) as run:
        with caplog.at_level(logging.WARNING, logger="patchcore.tracking"):
            run.log_artifacts("/nonexistent/path")

    assert "Artifact path not found" in caplog.text
