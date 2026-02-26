from __future__ import annotations

import torch
from src.ml.config import load_config


class Predictor:
    def __init__(self, params_path: str = "params.yaml"):
        cfg = load_config(params_path)
        self.cfg = cfg

        torch.set_num_threads(cfg.service.runtime.num_threads)

        model_path = cfg.service.model.path
        self.model = torch.jit.load(model_path, map_location="mps")
        self.model.eval()

        self.c = cfg.service.input.channels
        self.h = cfg.service.input.height
        self.w = cfg.service.input.width

        self.mean = torch.tensor(cfg.service.preprocess.mean, dtype=torch.float32).view(1, self.c, 1, 1)
        self.std = torch.tensor(cfg.service.preprocess.std, dtype=torch.float32).view(1, self.c, 1, 1)

        self.labels = cfg.service.labels
    
    @torch.inference_mode() 
    def predict(self, x: list[float]) -> dict:
        expected = self.c * self.h * self.w
        if len(x) != expected:
            raise ValueError(f"Expected {expected} floats (C*H*W = {self.c}*{self.h}*{self.w}). Got {len(x)}. ")
        
        t = torch.tensor(x, dtype=torch.float32).view(1, self.c, self.h, self.w)

        if self.cfg.service.preprocess.scale_0_255_to_0_1:
            t = t / 255.0

        if self.cfg.service.preprocess.normalize:
            t = (t - self.mean) / self.std
        
        logits = self.model(t)
        class_id = int(torch.argmax(logits, dim=1).item())

        out = {"class_id": class_id}
        if self.labels and 0 <= class_id < len(self.labels):
            out["label"] = self.labels[class_id]
        return out

