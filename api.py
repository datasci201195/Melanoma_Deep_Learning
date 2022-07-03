# way to upload image: endpoint
# way to save the image
# function to make prediction on image
# show the results
import os
from flask import Flask
from flask import request
from flask import render_template

from pickletools import optimize
from sched import scheduler
from turtle import forward
from sklearn import metrics
import torch

import albumentations
import pretrainedmodels

import numpy as np
import pandas as pd

from sklearn import metrics
import torch.nn as nn
from torch.nn import functional as F

from torch.nn import functional
from wtfml.data_loaders.image import ClassificationLoader
from wtfml.utils import EarlyStopping
from wtfml.engine import Engine
from xgboost import DeviceQuantileDMatrix

app = Flask(__name__)
UPLOAD_FOLDER = "/home/rupesh/projects/Melonoma_Deep_Learning/static"
DEVICE = "cuda"
MODEL = None

class SEResnext50_32x4d(nn.Module):
    def __init__(self,pretrained = "imagenet"):
        super(SEResnext50_32x4d, self).__init__()

        self.base_model = pretrainedmodels.__dict__[
            "se_resnext50_32x4d"
        ](pretrained=None)
        if pretrained is not None:
            self.base_model.load_state_dict(
                torch.load(
                    "/home/rupesh/projects/Melonoma_Deep_Learning/input/se_resnext50_32x4d-a260b3a4.pth"
                )
            )

        self.l0 = nn.Linear(2048, 1)

    def forward(self, image, targets):
        batch_size, _, _, _ = image.shape
        
        x = self.base_model.features(image)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        
        
        out = torch.sigmoid(self.l0(x))
        loss = 0

        return out, loss

def predict(image_path, model):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
        ]
    )

    
    images = [image_path]
    targets = [0]

    test_dataset = ClassificationLoader(
        image_paths=images,
        targets=targets,
        resize=None,
        augmentations=aug,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4
    )


    predictions = Engine.predict(test_loader, model, device=DEVICE)
    predictions = np.vstack((predictions)).ravel()

    return predictions

@app.route("/",methods = ["GET","POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            pred = predict(image_location,MODEL)[0]
            return render_template("index.html", prediction=pred*10, image_loc = image_file.filename)
    return render_template("index.html", prediction=0, image_file= None)

if __name__ == "__main__":
    MODEL = SEResnext50_32x4d(pretrained="imagenet")
    MODEL.load_state_dict(torch.load("/home/rupesh/projects/Melonoma_Deep_Learning/model_fold_0.bin"))
    MODEL.to(DEVICE)
    app.run(port = 12000, debug = True)

