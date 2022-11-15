import time
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import torch
# import transforms as T
from movinets import MoViNet
from movinets.config import _C

from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import find_classes, make_dataset

import os
import glob
import json

import numpy as np
import cv2
import mediapy as media
import pandas as pd
import matplotlib.pyplot as plt

import onnxruntime
import json
import time

from tqdm import tqdm

def load_video(file_path, image_size=(224, 224), original_fps=30, new_fps=5, start_time=None, end_time=None, gray=False):
    """Loads a video file into a TF tensor."""
    cap = cv2.VideoCapture(file_path)
    
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fc = 0
    ret = True

    if start_time and end_time:
        start_frame = original_fps * start_time
        end_frame = original_fps * end_time
    else:
        start_frame = 0
        end_frame = frameCount
    
    fps_factor = original_fps / new_fps
    now_frame = 0
    if gray:
        buf = np.zeros((int((end_frame - start_frame) / fps_factor), image_size[1], image_size[0]), np.dtype('uint8'))
    else:
        buf = np.zeros((int((end_frame - start_frame) / fps_factor), image_size[1], image_size[0], 3), np.dtype('uint8'))
    
    while (fc < frameCount  and ret):
        ret, tmp = cap.read()
        now_frame += 1
        if start_frame > now_frame:
            continue
        if end_frame < now_frame:
            break
        if now_frame % fps_factor == 0:
            tmp = cv2.resize(tmp, dsize=image_size)
            if gray:
                buf[fc] = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
            else:
                buf[fc] = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
            fc += 1
    cap.release()
    
    return buf


def preprocess_video(video, image_size=(224,224), frame_num=50, dtype=np.float32):
    # thwc
    # video = load_video(path, image_size=image_size)
    
    # set t=frame_num
    t,h,w,c = video.shape
    if t < frame_num:
        fill_n = frame_num - t
        video = np.concatenate([video, torch.zeros((fill_n, h, w, c), dtype=torch.uint8)], axis=0)
    elif t > frame_num:
        video = video[:frame_num]
    
    # cthw
    video = np.transpose(video, (3,0,1,2))
    
    # ncthw
    video = np.expand_dims(video, axis=0)
    
    video = video.astype(dtype)
    if dtype == np.float32:
        video /= 255
    
    return video

def log_softmax(x):
    e_x = np.exp(x - np.max(x))
    return np.log(e_x / e_x.sum())

def print_topk(out):
    # ados_dict = {0: 'abnormal', 1: 'normal'}
    # abnormal_list = ["doing_sudoku", "drawing", "taking_photo", "writing"]
    # doubt_list = ["calligraphy", "calculating", "talking_on_cell_phone", "texting"]

    abnormal_list = ["texting", "drawing", "taking_photo"]
    doubt_list = ["calling"]
    
    a = [log_softmax(o) for o in out[0]]
    b = np.argsort(a, axis=1)

    k = 2
    for i in range(len(b)):
        topk = [classes_dict[str(p)] for p in b[i][-k:][::-1]]
        print(topk)
        if any([tk in abnormal_list for tk in topk]):
            print("pred_: abnormal")
        elif any([tk in doubt_list for tk in topk]):
            print("pred_: doubt")
        else:
            print("pred_: normal")
            
class OpenMax:
    def __init__(self, 
                 dataset_root='/home/workspaces/datasets/ados_act/ados/all/',
                 classes_json_path="14classes.json", 
                 num_videos=300, 
                 device=None,
                 classifier_weight_path='fine14_iter18_a2.onnx',
                 mean_path=None,
                 weibull_param_path=None
                ):
        # assert num_videos should be larger than 2
        
        
        # load classes
        self.classes_dict = {}
        with open(classes_json_path, 'r') as f:
            classes = json.load(f)
        for key, value in classes.items():
            self.classes_dict[value] = key
        print("classes: ", self.classes_dict)
        self.num_classes = len(self.classes_dict.keys())
        
        self.num_videos = num_videos
        self.video_dict = {}
        for class_name in list(self.classes_dict.values()):
            self.video_dict[class_name] = glob.glob(os.path.join(dataset_root, class_name) + '/*.mp4')
        
        # device
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        
        # load model
        self.classifier_weight_path = classifier_weight_path
        if 'onnx' in classifier_weight_path:
            self.model_type = 'onnx'
            
            # onnx inference session
            if 'cuda' in self.device:
                providers = ["CUDAExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
            
            self.model = onnxruntime.InferenceSession(classifier_weight_path, providers=providers)
            self.input_name = self.model.get_inputs()[0].name
            self.output_name = self.model.get_outputs()[0].name
 
            if 'float16' in classifier_weight_path:
                self.input_type = np.float16
            else:
                self.input_type = np.float32
                
        elif 'pth' in classifier_weight_path:
            self.model_type = 'pytorch'
            self.model = MoViNet(_C.MODEL.MoViNetA2, causal = False, pretrained = False)
            self.model.classifier[3] = torch.nn.Conv3d(2048, self.num_classes, (1,1,1))
            self.model.load_state_dict(torch.load(classifier_weight_path, map_location='cpu'))
            self.model.eval()
            self.model.to(self.device)
        else:
            raise NotImplementedError
        
        print(f"{self.model_type} model successfully loaded.")
        
        
        if mean_path is not None and weibull_param_path is not None:
            self.load_openmax_params(mean_path, weibull_param_path)
        else:
            self.mean = None
            self.weibull_param = None
    
    def check_fitted(self):
        if self.mean is None\
            or self.weibull_param is None:
            return False
        return True
    
    def compute_logit(self, input_data):
        if self.model_type == 'pytorch':
            with torch.no_grad():
                pred = self.model(input_data.to(self.device))
        elif self.model_type == 'onnx':
            pred = self.model.run([self.output_name], {self.input_name: input_data.astype(self.input_type)})
            pred = torch.from_numpy(pred[0]).to(self.device)
            
        output = F.softmax(pred, dim=1)

        return output

    
    def calc_logit(self):
        
        mean = torch.zeros(self.num_classes, 1, self.num_classes)
        output_matrix = torch.zeros(self.num_classes, self.num_videos, self.num_classes)

        c = 0
        for class_name in sorted(list(self.classes_dict.values())):
            s = torch.zeros(1, self.num_classes).to(self.device)
            v = 0
            for path in tqdm(self.video_dict[class_name][:self.num_videos], desc=class_name):
                
                time.sleep(0.01)
                video = load_video(path)
                input_data = preprocess_video(video)
                
                output = self.compute_logit(input_data)
                    
                output_matrix[c][v] = output
                s = s.add(output.to(self.device))
                v += 1
                
                
                
            mean[c] = torch.mul(s, 1/self.num_videos)
            c += 1
            
        return output_matrix, mean
    
    def fit_weibull(self, x, iters=100, eps=1e-6):
        k = 1.0
        k_t_1 = k
        ln_x = torch.log(x)

        for i in tqdm(range(min(iters, self.num_videos))):
            # Partial derivative df/dk
            x_k = x ** k
            x_k_ln_x = x_k * ln_x
            ff = torch.sum(x_k_ln_x)
            fg = torch.sum(x_k)
            f1 = torch.mean(ln_x)
            f = ff/fg - f1 - (1.0 / k)

            ff_prime = torch.sum(x_k_ln_x * ln_x)
            fg_prime = ff
            f_prime = (ff_prime / fg - (ff / fg * fg_prime / fg)) + (1. / (k * k))

            # Newton-Raphson method k = k - f(k;x)/f'(k;x)
            k -= f / f_prime
            # print('f=% 7.5f, dk=% 7.5f, k=% 7.5f' % (f.data[0], k.grad.data[0], k.data[0]))
            if np.isnan(f):
                return np.nan, np.nan
            if abs(k - k_t_1) < eps:
                break

            k_t_1 = k

        # Lambda (scale) can be calculated directly
        lam = torch.mean(x ** k) ** (1.0 / k)

        return torch.Tensor([[k, lam]])  # Shape (SC), Scale (FE)
        
    def fit(self, num_sample=10, eps=1e-6):
        output_matrix, mean = self.calc_logit()
        
        distance = torch.zeros(self.num_classes, 1, self.num_videos)
        for i in range(self.num_classes):
            distance[i], indices = torch.sort(torch.norm(output_matrix[i] - mean[i], dim=1), descending=True)
            
        sampled_distance = distance[:, :, 0:num_sample]

        weibull_param = torch.zeros(self.num_classes, 2)
        for i in range(self.num_classes):
            sample = sampled_distance[i][0]
            weibull_param[i] = torch.Tensor(self.fit_weibull(sample, iters=100, eps=eps))
        
        self.mean = mean
        self.weibull_param = weibull_param
    
    def get_dist(self, output):
        
        if not self.check_fitted():
            raise ValueError("you must use fit() before this operation")
        # input_data = preprocess_video(video)
        # output = self.compute_logit(input_data)
        
        A = output[0].repeat(self.num_classes, 1)
        B = torch.squeeze(self.mean, dim=1)
        
        self.A = A
        self.B = B
        dist = torch.norm(A - B, dim=1)

        return dist.to(self.device)
    
    def compute_weibull_probability(self, dist):
        if not self.check_fitted():
            raise ValueError("you must use fit() before this operation")
        
        weibull_probability = torch.Tensor(self.num_classes).to(self.device)
        
        for i in range(self.num_classes):
            k, lam = self.weibull_param[i]
            weibull_probability[i] = torch.ones(1).to(self.device) - torch.exp(-(dist[i]/lam.to(self.device))**k)
        
        return weibull_probability
    
    def save_openmax_params(self):
        
        model_name = os.path.splitext(self.classifier_weight_path)[0]
        np.save(model_name + '_mean.npy', self.mean.cpu().numpy())
        np.save(model_name + '_weibull_param.npy', self.weibull_param.cpu().numpy())
        
        print("openmax parameters saved")
        
    def load_openmax_params(self, mean_path, weibull_param_path):
        self.mean = torch.from_numpy(np.load(mean_path))
        self.weibull_param = torch.from_numpy(np.load(weibull_param_path))
        
        self.mean = self.mean.to(self.device)
        self.weibull_param = self.weibull_param.to(self.device)
        
        print("openmax parameters loaded")
    
    
    
    
if __name__ == '__main__':
    openmax = OpenMax(classes_json_path="14classes.json", 
                  device='cuda:0', 
                  classifier_weight_path='fine14_iter18_a2.onnx',
                  num_videos=100,
                  mean_path='fine14_iter18_a2_mean.npy',
                  weibull_param_path='fine14_iter18_a2_weibull_param.npy'
                 )

    # openmax.fit(num_sample=20, eps=1e-6)