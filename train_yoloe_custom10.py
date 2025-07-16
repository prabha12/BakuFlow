#!/usr/bin/env python3
"""
Train YOLOE for Custom 10 Classes (BakuFlow)
"""
import os
from ultralytics import YOLOE
from ultralytics.utils import yaml_load, LOGGER
import torch

# Set your dataset config (YOLO format)
data_yaml = "C:\\Users\\rlakn\\OneDrive\\Desktop\\AIML_Projects\\AutoLabeling\\TSR_data\\data.yaml"  # Update this path to your dataset config

# Read class names from classes.txt
with open("classes.txt", "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f if line.strip()]
assert len(class_names) == 10, f"Expected 10 classes, got {len(class_names)}"

# Model config (choose scale: v8l, v8m, v8s, etc.)
model_cfg = YOLOE("yoloe-v8l.yaml") # or the correct path to your config file")  # Update if you want a different scale
# Output directory
output_dir = "runs/train_yoloe_custom10"
os.makedirs(output_dir, exist_ok=True)

# Training parameters
epochs = 80
batch_size = 32
imgsz = 640
lr0 = 1e-3
optimizer = "AdamW"

# Initialize model
model = YOLOE(model_cfg)

# Set classes and text prompt embeddings
text_pe = model.get_text_pe(class_names)
model.set_classes(class_names, text_pe)

# Save text prompt embeddings for reproducibility
pe_path = os.path.join(output_dir, "custom10-pe.pt")
torch.save({"names": class_names, "pe": text_pe}, pe_path)

# Train
model.train(
    data=data_yaml,
    epochs=epochs,
    batch=batch_size,
    imgsz=imgsz,
    optimizer=optimizer,
    lr0=lr0,
    device="cuda" if torch.cuda.is_available() else "cpu",
    project=output_dir,
    name="yoloe_custom10",
    train_pe_path=pe_path
)

print(f"Training complete! Check weights and logs in {output_dir}")
