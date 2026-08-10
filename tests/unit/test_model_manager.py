from __future__ import annotations

from types import SimpleNamespace

from da3_cad import model_manager
from da3_cad.model_manager import StagedModelManager


class _Device:
    type = "cpu"

    def __str__(self) -> str:
        return "cpu"


class _Parameter:
    def numel(self) -> int:
        return 7

    def element_size(self) -> int:
        return 4


class _Model:
    def __init__(self) -> None:
        self.moves: list[str] = []

    def parameters(self) -> list[_Parameter]:
        return [_Parameter(), _Parameter()]

    def buffers(self) -> list[_Parameter]:
        return []

    def to(self, device: _Device) -> _Model:
        self.moves.append(str(device))
        return self


def test_cpu_stage_has_same_load_infer_unload_lifecycle(monkeypatch: object) -> None:
    fake_torch = SimpleNamespace(
        __version__="test",
        version=SimpleNamespace(cuda=None),
        cuda=SimpleNamespace(is_available=lambda: False),
        device=lambda _name: _Device(),
    )
    monkeypatch.setattr(model_manager, "_torch_module", lambda: fake_torch)  # type: ignore[attr-defined]
    model = _Model()

    result, report = StagedModelManager("cpu").execute(lambda: model, lambda _: "result")

    assert result == "result"
    assert model.moves == ["cpu", "cpu"]
    assert report.model_parameters == 14
    assert report.model_parameter_bytes == 56
    assert report.peak_allocated_bytes is None
    assert report.unload_returned_to_baseline is None
