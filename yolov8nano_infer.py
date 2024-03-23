from ultralytics import YOLO
import argparse
import os
import platform
import sys
from pathlib import Path

def text_detection():
# Load a model
  model = YOLO('./pretrained_model/YOLOv8nanoTextDe/best.pt')

  img_path = "./cut/test.jpg"

#change iou and conf following to your best value
  result = model(img_path, batch=1, iou = 0.7, conf = 0.25)
  t=result[0].boxes
  return t.xyxy
# convert result to json
# print(result)