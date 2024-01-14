import os
from pathlib import Path
from transformers.convert_graph_to_onnx import convert


if __name__ == "__main__":
    output_dir = "models"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=False)
 
    output_fpath = os.path.join(output_dir, "kf-deberta-multitask.onnx")
    convert(framework="pt", model="upskyy/kf-deberta-multitask", output=Path(output_fpath), opset=15)
