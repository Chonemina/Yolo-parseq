from PIL import Image
import os
import torch
import json
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint
import argparse

@torch.inference_mode()
def init_model(checkpoint, device):
    model = load_from_checkpoint(checkpoint).eval().to(device)
    return model
def recognize(model, img, device):
    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)
    img = img_transform(img).unsqueeze(0).to(device)
    p = model(img).softmax(-1)
    pred, p = model.tokenizer.decode(p)
    return pred[0]

def try_recognize():
    device = "cpu"
    checkpoint = "G://Lab//auto_quiz_scoring//yolo_experiment//first_try//code_reg//using_parseq//parseq//pretrained//parseq-tiny-epoch=9-step=244-val_accuracy=99.0909-val_NED=99.0909.ckpt"
    model = load_from_checkpoint(checkpoint).eval().to(device)
    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)
    img_dir_path = r"G:\Lab\auto_quiz_scoring\yolo_experiment\first_try\end-to-end-tech\demo1-yolo-parseq\test_set"
    predict = r"G:\Lab\auto_quiz_scoring\yolo_experiment\first_try\end-to-end-tech\demo1-yolo-parseq\parseq_infer\result.json"
    imgs = os.listdir(img_dir_path)
    preds = []
    for img_name in imgs:
        image_path = os.path.join(img_dir_path, img_name)
        img = Image.open(image_path).convert('RGB')
        img = img_transform(img).unsqueeze(0).to(device)
        p = model(img).softmax(-1)
        pred, p = model.tokenizer.decode(p)
        results = {"image": img_name, "pred": pred[0]}
        preds.append(results)
    print("Number of prediction: ", len(preds))
    with open(predict, "w", encoding="utf-8") as f:
        for pred in preds:
            json.dump(pred,f)
            f.write("\n")
