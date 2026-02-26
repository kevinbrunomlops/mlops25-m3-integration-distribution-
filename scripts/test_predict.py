from src.ml.predict import Predictor
import random

p = Predictor("params.yaml")
x = [random.random() for _ in range(3072)]
print(p.predict(x))