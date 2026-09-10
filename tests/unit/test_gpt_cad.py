from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image
from typer.testing import CliRunner

from da3_cad.cad.program import extract_parameters
from da3_cad.cli import app
from da3_cad.gpt_cad import (
    CADResponse,
    GPTConfig,
    Parameter,
    collect_images,
    parameterize,
    prepare_images,
    provider_credentials,
    run_gpt_cad,
    validate_generated_program,
)


def candidate(code: str | None = None) -> CADResponse:
    return CADResponse(
        name="Block",
        code=code or "import cadquery as cq\nlength=20\nr=cq.Workplane('XY').box(length,10,5)\n",
        parameters=[Parameter(name="length", value=20, unit="mm", source="specified")],
        assumptions=[],
    )


class FakeClient:
    def __init__(self, answers: list[CADResponse | None], status: str = "completed"):
        self.answers = iter(answers)
        self.calls: list[dict[str, Any]] = []
        self.responses = self
        self.status = status

    def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="test-response",
            model="test-model",
            status=self.status,
            usage=None,
            output_parsed=next(self.answers),
        )


def test_parameterization_preserves_editable_values() -> None:
    source = parameterize(candidate())
    assert extract_parameters(source) == {"length": 20}
    assert "length = PARAMETERS['length']" in source


@pytest.mark.parametrize(
    "expression",
    [
        "cq.exporters.export(r,'/tmp/unwanted')",
        "cq.importers.importStep('/tmp/private')",
        "cq.Workplane.__class__",
        "cq.__dict__",
        "open('/tmp/private')",
    ],
)
def test_generated_program_cannot_use_io_or_introspection(expression: str) -> None:
    with pytest.raises(ValueError):
        validate_generated_program(candidate(candidate().code + expression))


def test_images_are_resized_and_all_views_preserved(tmp_path: Path) -> None:
    paths = []
    for i in range(2):
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (2000, 1000)).save(p)
        paths.append(p)
    content, manifest = prepare_images(collect_images(tmp_path))
    assert sum(c["type"] == "input_image" for c in content) == 2
    assert all(m["submitted_size"] == [1536, 768] for m in manifest)
    assert all("sha256" in m and "submitted_sha256" in m for m in manifest)


def test_proxy_credentials_env_then_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    p = tmp_path / ".config/llm-proxy/api_key"
    p.parent.mkdir(parents=True)
    p.write_text("file-secret\n")
    monkeypatch.delenv("LLMPROXY_API_KEY", raising=False)
    assert provider_credentials("llm-proxy")[0] == "file-secret"
    monkeypatch.setenv("LLMPROXY_API_KEY", "env-secret")
    assert provider_credentials("llm-proxy")[0] == "env-secret"


def test_missing_key_does_not_create_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("LLMPROXY_API_KEY", raising=False)
    with pytest.raises(ValueError, match="LLMPROXY_API_KEY"):
        run_gpt_cad("block", tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_generation_exports_real_cad_and_viewer(tmp_path: Path) -> None:
    client = FakeClient([candidate()])
    out = tmp_path / "run"
    result = run_gpt_cad("block length 20mm", out, client=client)
    assert result["status"] == "GENERATED"
    for name in [
        "model.step",
        "model.stl",
        "model.py",
        "viewer.html",
        "parameters.json",
        "report.json",
    ]:
        assert (out / name).stat().st_size > 0
    assert client.calls[0]["store"] is False
    assert client.calls[0]["model"] == "gpt-5.6-sol"
    assert json.loads((out / "quality.json").read_text())["geometric_accuracy_verified"] is False
    with pytest.raises(FileExistsError):
        run_gpt_cad("block", out, client=client)


def test_kernel_error_is_repaired_with_original_images(tmp_path: Path) -> None:
    bad = candidate("import cadquery as cq\nlength=20\nr=cq.Workplane('XY')\n")
    client = FakeClient([bad, candidate()])
    image = tmp_path / "view.png"
    Image.new("RGB", (32, 32)).save(image)
    result = run_gpt_cad(
        "block", tmp_path / "run", images=[image], client=client, create_viewer=False
    )
    assert len(client.calls) == 2
    assert result["attempts"][0]["status"] == "invalid"
    assert result["attempts"][1]["status"] == "valid"
    for call in client.calls:
        assert any(c["type"] == "input_image" for c in call["input"][0]["content"])
    assert "Validation error" in client.calls[1]["input"][0]["content"][-1]["text"]


@pytest.mark.parametrize("status,answer", [("incomplete", candidate()), ("completed", None)])
def test_incomplete_or_refused_response_never_exports(
    tmp_path: Path, status: str, answer: Any
) -> None:
    out = tmp_path / "run"
    with pytest.raises(RuntimeError):
        run_gpt_cad("block", out, client=FakeClient([answer], status))
    assert not (out / "model.step").exists()
    assert json.loads((out / "report.json").read_text())["status"] == "FAILED"


def test_repair_budget_is_bounded(tmp_path: Path) -> None:
    bad = candidate("import cadquery as cq\nlength=20\nr=cq.Workplane('XY')\n")
    client = FakeClient([bad, bad])
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        run_gpt_cad("block", tmp_path / "run", client=client, config=GPTConfig(max_repairs=1))
    assert len(client.calls) == 2
    assert not (tmp_path / "run/model.step").exists()


def test_product_cli_dry_run_has_no_api_or_files(tmp_path: Path) -> None:
    for command in ["generate", "reconstruct", "photo-cad"]:
        output = tmp_path / command
        result = CliRunner().invoke(
            app,
            [
                command,
                "--prompt",
                "plate",
                "--dimension",
                "width=20mm",
                "--output",
                str(output),
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "gpt-5.6-sol" in result.output and "width=20mm" in result.output
        assert not output.exists()
