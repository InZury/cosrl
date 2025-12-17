import os
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as vfunc

from cosrl.utils.transforms import Affine

# ImageNet base measure
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class COD10K(Dataset):
    def __init__(self, path, image_size=352, only_camo=False ,is_train=True):
        super().__init__()
        self.image_size = [image_size, image_size]
        self.only_camo = only_camo
        self.is_train = is_train

        self.image_path = os.path.join(path, "Image")
        self.gt_path = os.path.join(path, "GT_Object")

        if only_camo: # Magic Number that found from dataset [3040, 2026]
            if is_train:
                self.images = os.listdir(self.image_path)[:3040]
            else:
                self.images = os.listdir(self.image_path)[:2026]
        else:
            self.images = os.listdir(self.image_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self._get(index, is_origin=False)

    def get_origin(self, index):
        return self._get(index, is_origin=True)

    def _get(self, index, is_origin=False):
        image_path = os.path.join(self.image_path, self.images[index])
        gt_path = os.path.join(self.gt_path, self.images[index].replace(".jpg", ".png"))

        image = Image.open(image_path).convert("RGB")
        gt = Image.open(gt_path).convert('L')

        image = image.resize(self.image_size)
        gt = gt.resize(self.image_size)

        if is_origin:
            return image, gt
        else:
            image = vfunc.to_tensor(image)
            gt = vfunc.to_tensor(gt)

        if self.is_train:
            if random.random() < 0.5:
                image, gt = Affine(
                    image=image, gt=gt, angle=15, translate_ratio=0.1, scale_ratio=0.1, shear=[0.0],
                    image_size=self.image_size
                )()
            if random.random() < 0.5:
                image = vfunc.adjust_brightness(img=image, brightness_factor=random.uniform(0.8, 1.2))
                image = vfunc.adjust_contrast(img=image, contrast_factor=random.uniform(0.8, 1.2))

        image = vfunc.normalize(image, mean=MEAN, std=STD)

        return image, gt

