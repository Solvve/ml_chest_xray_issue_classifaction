#imports
import sys
sys.path.append('src')

import numpy
import torch
import torch.utils.data
import torchvision.transforms.functional
import torch
import torch._six
import torch.distributed as dist
import torch.utils.data
import torchvision
import cv2
from flask import render_template, redirect
from flask import Flask,flash, url_for
from flask import request

import os
import utils
import omegaconf
import json
import albumentations
import timm




with open('config.json') as f:
    cfg = json.load(f)

cfg = omegaconf.OmegaConf.create(cfg)

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = 'static/images/original'
path_to_file = ''

class ModelInference:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_cl = timm.create_model(cfg['MODEL_CLASSIFIER'], pretrained = False)
        n_features = self.model_cl.fc.in_features
        self.model_cl.fc = torch.nn.Linear(n_features, cfg['CLASSIFIER_N_CLASS'])
        self.model_cl.load_state_dict(torch.load(os.path.join(self.cfg.PATH_TO_MODEL, self.cfg.WEIGHT_FILE2), map_location = torch.device(cfg.DEVICE)))
        self.model_cl.eval()
        self.model = utils.get_model(self.cfg)
        self.model.load_state_dict(torch.load(os.path.join(self.cfg.PATH_TO_MODEL, self.cfg.WEIGHT_FILE), map_location = torch.device(cfg.DEVICE)))
        self.model.eval()
        valid_augs_list = [utils.load_obj(i['class_name'])(**i['params']) for i in self.cfg['augmentation']['valid']['augs']]
        #cl_augs_list = [utils.load_obj([i['class_name']])(**i['params']) for i in self.cfg['augmentation_classifier']['augs']]
        self.transforms_cl = albumentations.Compose([albumentations.Resize(256, 256, p = 1),
                                                    albumentations.pytorch.ToTensorV2(p=1)])
        self.transforms = albumentations.Compose(valid_augs_list)
        self.class_names = self.cfg.class_names
        self.detection_threshold = self.cfg.DETECTION_THRESHOLD
        self.filter_threshold = self.cfg.DETECTION_THRESHOLD_CL
        self.img_size = self.cfg.img_size

    def predict(self, image_path):
        img = cv2.imread(image_path)
        filename = image_path.split('/')[-1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(numpy.float32)
        h = img.shape[0]
        w = img.shape[1]
        img_result = img.copy()
        # Normalization
        img = img / 255.0

        #prediction step 1
        img_cl = self.transforms_cl(image = img)['image']
        filter_result = self.model_cl(img_cl.unsqueeze_(0)).detach().numpy()[0]
        filter_result = utils.softmax(filter_result)[1]
        filter_result = True if filter_result > self.filter_threshold else False
        #filter_result = True
        if filter_result:
            img = self.transforms(image=img)['image']

            prediction_results = self.model(img.unsqueeze_(0))
        # img = img.data.cpu().numpy()
        # print(prediction_results)
            boxes = prediction_results[0]['boxes'].data.cpu().numpy()
            scores = prediction_results[0]['scores'].data.cpu().numpy()
            labels = prediction_results[0]['labels'].data.cpu().numpy()
        # print(labels)
            boxes = boxes[scores >= self.detection_threshold]
            labels = labels[scores >= self.detection_threshold]
            label2color = {class_id: [numpy.random.randint(0, 255) for i in range(3)] for class_id in
                       numpy.unique(labels)}
            thickness = 3
            for label_id, box in zip(labels, boxes):
                color = label2color[label_id]
                img_result = cv2.rectangle(
                    img_result,
                    (int(box[0] / self.img_size * w), int(box[1] / self.img_size * h)),
                    (int(box[2] / self.img_size * w), int(box[3] / self.img_size * h)),
                    color, thickness
                )

                img_result = cv2.putText(img_result, self.class_names[str(label_id)], (int(box[0] / self.img_size * w), int(box[1] / self.img_size * h)),
                                         cv2.FONT_HERSHEY_COMPLEX, 0.5,color,1,  cv2.LINE_AA)
        else:
            labels = []

        saved_file = f'static/images/predicted/{filename}'

        cv2.imwrite(saved_file, cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))

        if len(labels) > 0:
            names = [self.class_names[str(i)] for i in labels]
        else:
            names = []
        return saved_file, names

model_inf = ModelInference(cfg)

@app.route('/', methods = ['GET', 'POST'])
@app.route('/index', methods = ['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        path_to_save = os.path.join(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        global path_to_file
        path_to_file = path_to_save
        file.save(path_to_save)
        return redirect(url_for('results'))

@app.route('/results', methods = ['GET', 'POST'])
def results():
    if request.method == 'GET':
        result_file, names = model_inf.predict(path_to_file)
        if len(names) == 0:
            msg = 'No abnormalities were detected'
            
        else:
            msg = 'Next abnormalities detected : ' + ', '.join(list(numpy.unique(names)))

        return render_template('results.html', img_original = path_to_file, img_detected = result_file, msg = msg)
    else:
        return redirect(url_for('index'))

    if __name__ == '__main__':
        app.jinja_env.auto_reload = True
        app.config['TEMPLATES_AUTO_RELOAD'] = True

        app.run(port=500, use_reloader=True)