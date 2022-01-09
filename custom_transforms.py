import torch
import random
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageFilter

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        target = sample['target']
        img = np.array(img).astype(np.float32)
        target = np.array(target).astype(np.float32)
        # img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'target': target}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        target = sample['target']
        target = np.array(target).astype(np.float32)
        ## normalize target size
        # w
        target[:, 0] = target[:, 0]/img.shape[1]
        # h
        target[:, 1] = target[:, 1]/img.shape[0]
        
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img /= 255.0
        img = torch.from_numpy(img).float()
        target = torch.from_numpy(target).float()
        
        ## flatten target
        if len(target.shape) > 1:
            target = np.reshape(target,(8))
        
        return {'image': img,
                'target': target}


class SortAxis(object):
    def __call__(self, sample):
        img = sample['image']
        target = sample['target']
        
        target = sorted(target, key=lambda x:x[0])
        target = np.array(target)

        return {'image': img,
                'target': target}
    
class FixedResize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample['image']
        target = sample['target']

        h_factor = self.size[1]/img.shape[0]
        w_factor = self.size[0]/img.shape[1]
        
        target[:, 0] = target[:, 0]*w_factor
        target[:, 1] = target[:, 1]*h_factor
        
        img = cv2.resize(img, dsize=self.size)

        return {'image': img,
                'target': target}
    

class RandomRotate(object):
    def __init__(self, degree=30.0):
        self.degree = random.uniform(-degree, degree)

    def __call__(self, sample):
        deg = self.degree
        img = sample['image']
        keypoints = sample['target']

        ori_w = img.shape[1]
        ori_h = img.shape[0]

        center = (ori_w * 0.5, ori_h * 0.5)  # x, y

        #회전이 끝나면 이미지가 약간 커짐
        #그래서 타겟 이미지는 회전의 최대 사이즈보다 커야 함
        max_len = int(np.sqrt(ori_w * ori_w + ori_h * ori_h) * 1.1)
        target_w = max_len
        target_h = max_len

        #회전하면서 동시에 타겟 가운대로 평행이동
        trans_x = (target_w - ori_w) // 2
        trans_y = (target_h - ori_h) // 2   

        #회전 + 평행이동 매트릭스
        rot_m = cv2.getRotationMatrix2D((int(center[0]), int(center[1])), deg, 1)
        rot_m[0, 2] += trans_x
        rot_m[1, 2] += trans_y

        #이미지 변환
        ret = cv2.warpAffine(img, rot_m, (target_w, target_h), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)
        # if img.ndim == 3 and ret.ndim == 2:
        #     ret = ret[:, :, np.newaxis]

        #포인트 변환
        jlen = len(keypoints)
        ones = np.ones(shape=(jlen, 1))
        points_ones = np.hstack([keypoints, ones])
        rotated = rot_m.dot(points_ones.T).T
        rotated = np.int32(rotated)

        #포인트를 기준으로 크롭해야 하는데
        #기존 변환 후 바운딩박스 사이즈부터 구하고
        col_x = rotated[:, 0]
        col_y = rotated[:, 1]
        bx1 = min(col_x)
        bx2 = max(col_x)
        by1 = min(col_y)
        by2 = max(col_y)
        bw = bx2 - bx1
        bh = by2 - by1
        rw = bw

        ret_w = int(bw * np.random.uniform(1.05, 1.2))
        ret_h = int(bh * np.random.uniform(1.05, 1.2))

        ret_w = max(ret_w, bw)
        ret_h = max(ret_h, bh)

        pad_x = ret_w - bw
        pad_y = ret_h - bh

        px = random.randint(max(bx1 - pad_x, 0), bx1)
        py = random.randint(max(by1 - pad_y, 0), by1)

        if px < 0 or py < 0:
            print("????")

        image = ret[py:py+ret_h, px:px+ret_w, :]
        keypoints = rotated - [px, py]

        return {'image': image,
                'target': keypoints}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'label': mask}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}
