from dataclasses import dataclass
from pathlib import Path
import yaml 

@dataclass
class Config:
    seed: int
    data_dir: str
    num_workers: int
    epochs: int
    batch_size: int
    lr: float
    run_name: str
    out_dir: str

def load_config(path: str = "params.yaml") -> Config:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    
    raw = yaml.safe_load(p.read_text())

    return Config(
        seed=int(raw.get("seed", 42)),
        data_dir=str(raw["data"]["dir"]),
        num_workers=int(raw["data"].get("num_workers", 2)),
        epochs=int(raw["train"]["epochs"]),
        batch_size=int(raw["train"]["batch_size"]),
        lr=float(raw["train"]["lr"]),
        run_name=str(raw["run"]["name"]), 
        out_dir=str(raw["run"].get("out_dir", "runs")),
    ) 