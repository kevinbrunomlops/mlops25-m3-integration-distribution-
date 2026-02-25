from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Any 
import yaml

ModelFormat = Literal["torchscript", "onnx"]
InputEncoding = Literal["flat", "nested"]

@dataclass(frozen=True)
class ModelConfig:
    format: ModelFormat
    path: str
@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "mps"
    num_threads: int = 1

@dataclass(frozen=True)
class InputConfig:
    channels: int
    height: int
    width: int
    encoding: InputEncoding = "flat"
    
@dataclass(frozen=True)
class PreprocessConfig:
    scale_0_255_to_0_1: bool = False
    normalize: bool = True
    mean: list[float] = None
    std: list[float] = None

@dataclass(frozen=True)
class ServiceConfig:
    model: ModelConfig
    runtime: RuntimeConfig
    input: InputConfig
    preprocess: PreprocessConfig
    labels: list[str]

@dataclass(frozen=True)
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000

@dataclass(frozen=True)
class AppConfig:
    service: ServiceConfig
    api: APIConfig
    
def _require(d: dict[str, Any], key:str) -> Any:
    if key not in d:
        raise KeyError(f"Missing required key: {key}")
    return d[key]

def load_config(params_path: str = "params.yaml") -> AppConfig:
    p = Path(params_path)
    if not p.exists():
        raise FileNotFoundError(f"Missing config file: {params_path}")
    
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    service = _require(raw, "service")
    api = raw.get("api") or {}

    model_raw = _require(service, "model")
    runtime_raw = service.get("runtime", {})
    input_raw = _require(service, "input")
    pre_raw = service.get("preprocess",{})
    labels = service.get("labels", [])

    model = ModelConfig(
        format=model_raw.get("format","torchscript"),
        path=_require(model_raw, "path"),
    )

    runtime = RuntimeConfig(
        device=runtime_raw.get("device", "mps"),
        num_threads=int(runtime_raw.get("num_threads", 1)),
    )

    inp = InputConfig(
        channels=int(_require(input_raw, "channels")),
        height=int(_require(input_raw, "height")),
        width=int(_require(input_raw, "width")),
        encoding=input_raw.get("encoding", "flat"),
    )

    preprocess = PreprocessConfig(
        scale_0_255_to_0_1=bool(pre_raw.get("scale_0_255_to_0_1", False)),
        normalize=bool(pre_raw.get("normalize", True)),
        mean=list(pre_raw.get("mean", [0.0, 0.0, 0.0])),
        std=list(pre_raw.get("std", [1.0, 1.0, 1.0])),
    )

    svc = ServiceConfig(
        model=model,
        runtime=runtime,
        input=inp,
        preprocess=preprocess,
        labels=list(labels),
    )
    api_cfg = APIConfig(
        host=api.get("host", "0.0.0.0"),
        port=int(api.get("port", 8000)),
    )

    return AppConfig(service=svc, api=api_cfg)
