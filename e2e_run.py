from ultralytics import YOLO
from utils.plots import Annotator, colors
import os
from parseq_infer import init_model, recognize
import torch
from PIL import Image, ImageDraw
import cv2
import json
from autoCorrect import lostquest_id
# Load a model yolov8
model_object_detect = YOLO(r'/kaggle/working/Yolo-parseq/pretrained_model/medium/best.pt')
model_text_detect = YOLO(r"/kaggle/working/Yolo-parseq/pretrained_model/YOLOv8nanoTextDe/best.pt")
# test_set is a folder that just contains 30 test images
test_set = r"/kaggle/working/Yolo-parseq/test_set"
imgs = os.listdir(test_set)
imgs_path = []
for img in imgs:
    imgs_path.append(os.path.join(test_set,img))
#change iou and conf following to your best value
results = model_object_detect(test_set, batch=1, iou = 0.7, conf = 0.25)
print(len(results))

#path to the parseq model
parseq_model = r"/kaggle/working/Yolo-parseq/pretrained_model/parseq-tiny-epoch=7-step=298-val_accuracy=99.0909-val_NED=99.0909.pt"

# Next, we start recognize using parseq
parseq = init_model(checkpoint=parseq_model, device= "cpu")

for i,result in enumerate(results):
    sum_res = []
    img = Image.open(imgs_path[i]).convert('RGB')
    im0 = img.copy()
    annotator = Annotator(im0, line_width=3, pil=True)
    result = json.loads(result.tojson())
    # print(type(result))
    for res in result:
        # print(type(res))
        xy1xy2 = [res["box"]["x1"],res["box"]["y1"],res["box"]["x2"],res["box"]["y2"]]
        # print(xy1xy2)
        c = res["class"]  # integer class
        #just for autocorrect part 
        questidquantity=0
        fullquestquantity=0
        # Now we check if the object is selected_ans to infer it in yolov8 text detection
        if c == 3:  # change this to "c == 1" if run Yolov9 detect 3 class
            im1 = img.crop(tuple(xy1xy2))
            text_res = model_text_detect(im1)
            text_res = json.loads(text_res[0].tojson())
            best_text_res = None
            # Maybe the text detection res contain more than one box, we have to find best box
            if len(text_res) > 1:
                max_conf = 0
                for i in range(len(text_res)):
                    if text_res[i]["confidence"] > max_conf:
                        best_text_res = text_res[i]
                        max_conf = text_res[i]["confidence"]
            elif len(text_res) == 1:
                best_text_res = text_res[0]
                print(best_text_res)
                print("Got text detection!")
            if best_text_res != None:
                xy1xy2_text = [best_text_res["box"]["x1"],best_text_res["box"]["y1"],best_text_res["box"]["x2"],best_text_res["box"]["y2"]]
                im2 = im1.crop(tuple(xy1xy2_text))
            else:
                im2 = im1 #if its got no change, just take it
            with torch.no_grad():
                label = recognize(parseq,img=im2,device="cpu")
                print(label)
            annotator.box_label(xy1xy2, label, color=colors(c, True))
            x1 = int(xy1xy2[0])
            y1 = int(xy1xy2[1])
            x2 = int(xy1xy2[2])
            y2 = int(xy1xy2[3])
            res = str([x1,y1,x2,y1,x2,y2,x1,y2])[1:-1] + "," + label + "\n"
            res = res.replace(" ", "")
            sum_res.append(res)
        # if it is the ques_id object, just crop it and recognize it
        elif c == 2:
            #tim so luong box quest_id:
            questidquantity+=1
            #mainpart
            im1 = img.crop(tuple(xy1xy2))
            with torch.no_grad():
                label = recognize(parseq,img=im1,device="cpu")
                print(label)
            annotator.box_label(xy1xy2, label, color=colors(c, True))
            x1 = int(xy1xy2[0])
            y1 = int(xy1xy2[1])
            x2 = int(xy1xy2[2])
            y2 = int(xy1xy2[3])
            res = str([x1,y1,x2,y1,x2,y2,x1,y2])[1:-1] + "," + label + "\n"
            res = res.replace(" ", "")
            sum_res.append(res)
        #tim so luong box full_quest
        if c==0: 
            fullquestquantity+=1
            # 
    if fullquestquantity!=questidquantity: lostquest_id(sum_res)
    else: print("Không co cau hoi nao bi detect thieu")
    #          
    img = annotator.result()
    img_save_path = r"/kaggle/working/Yolo-parseq/test_set" + imgs[i]
    txt_save_path = r"/kaggle/working/Yolo-parseq/test_set" + imgs[i][:-3]+"txt"
    cv2.imwrite(img_save_path, img)
    with open(txt_save_path, "w") as f:
        f.writelines(sum_res)
        