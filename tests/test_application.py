"""Input, real model and HTTP regression tests for baseline-only inference."""

import io
from pathlib import Path
import pytest
from persistency.schema import FEATURES, validate_records
from persistency.service import PredictionService
from persistency.web import create_app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(name="service", scope="module")
def fixture_service():
    """Load the reviewed baseline-only artifact."""
    return PredictionService(ROOT / "models")


@pytest.fixture(name="client")
def fixture_client():
    """Create isolated HTTP client."""
    return create_app({"TESTING": True, "MODEL_DIR": ROOT / "models"}).test_client()


@pytest.mark.parametrize("records", [[], {}, None, [None], [{}], [{}] * 501])
def test_invalid_records(records):
    """Require a bounded batch of complete objects."""
    with pytest.raises(ValueError):
        validate_records(records)


@pytest.mark.parametrize("value", [42, True, [], "not-a-category", float("nan")])
def test_invalid_categories(service, value):
    """Unknown/invalid source categories fail clearly."""
    with pytest.raises(ValueError):
        validate_records([dict(service.sample, Age_Bucket=value)])


def test_missing_and_extra_fields(service):
    """Explicit missing values are encoded; extra sensitive fields are ignored."""
    assert service.predict([dict(service.sample, Age_Bucket=None)])
    assert service.predict([dict(service.sample, Ptid="ignored")]) == service.predict(
        [service.sample]
    )
    with pytest.raises(ValueError):
        validate_records([{field: None for field in FEATURES}])


def test_batch_parity(service):
    """Single and batched predictions agree."""
    result = service.predict([service.sample])[0]
    assert service.predict([service.sample] * 3) == [result] * 3
    assert abs(sum(result["scores"].values()) - 1) < 0.00001


def test_no_during_treatment_features():
    """The deployed feature contract excludes obvious future/outcome measurements."""
    assert len(FEATURES) == 6
    assert not any("During" in name or "Adherent" in name for name in FEATURES)


def test_api_html_csv(client, service):
    """Exercise the three user entry points with the same model."""
    assert client.get("/").status_code == 200
    result = client.post("/api/predict", json={"records": [service.sample]})
    assert result.status_code == 200
    assert result.json["predictions"] == service.predict([service.sample])
    html = client.post("/predict", data=service.sample)
    assert html.status_code == 200 and b"MODEL RESULT" in html.data
    sample = client.get("/sample.csv")
    result = client.post(
        "/predict-csv", data={"file": (io.BytesIO(sample.data), "sample.csv")}
    )
    assert result.status_code == 200
    assert service.predict([service.sample])[0]["category"].encode() in result.data


def test_client_errors(client):
    """Malformed payloads, missing uploads and unknown routes fail safely."""
    assert client.post("/api/predict", json={}).status_code == 400
    assert client.post("/predict-csv").status_code == 400
    assert client.get("/missing").status_code == 404
    assert (
        client.post(
            "/api/predict",
            data=b"x" * (1024 * 1024 + 1),
            content_type="application/json",
        ).status_code
        == 413
    )


def test_report_and_health(client):
    """Readiness and metric totals describe the current artifact."""
    assert client.get("/health/ready").json["status"] == "ready"
    report = client.get("/api/metrics").json
    assert sum(v["rows"] for v in report["split"].values()) == 3424
    assert sum(map(sum, report["test"]["confusion_matrix"])) == report["test"]["rows"]


def test_integrity_rejection(tmp_path):
    """Do not deserialize mismatched artifacts."""
    (tmp_path / "metrics.json").write_bytes((ROOT / "models/metrics.json").read_bytes())
    (tmp_path / "pipeline.joblib").write_bytes(b"bad")
    with pytest.raises(ValueError, match="integrity"):
        PredictionService(tmp_path)
