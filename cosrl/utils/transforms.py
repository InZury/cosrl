import random

from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import functional as vfunc


class RandomResizeCrop:
    def __init__(self, image, gt, scale, ratio, size):
        self.image = image
        self.gt = gt
        self.scale = scale
        self.ratio = ratio
        self.size = size

    def __call__(self):
        top, left, height, width = RandomResizedCrop.get_params(
            img=self.image, scale=self.scale, ratio=self.ratio
        )

        image = vfunc.resized_crop(
            img=self.image, top=top, left=left, height=height, width=width,
            size=self.size, interpolation=vfunc.InterpolationMode.BILINEAR
        )
        gt = vfunc.resized_crop(
            img=self.gt, top=top, left=left, height=height, width=width,
            size=self.size, interpolation=vfunc.InterpolationMode.NEAREST
        )

        return image, gt


class Affine:
    def __init__(self, image, gt, angle, translate_ratio, scale_ratio, shear, image_size=352):
        self.image = image
        self.gt = gt
        self.angle = random.uniform(-angle, angle)
        self.translate = [
            int(random.uniform(-translate_ratio, translate_ratio) * image_size),
            int(random.uniform(-translate_ratio, translate_ratio) * image_size)
        ]
        self.scale = random.uniform(1.0 - scale_ratio, 1.0 + scale_ratio)
        self.shear = shear

    def __call__(self):
        image = vfunc.affine(
            img=self.image, angle=self.angle, translate=self.translate, scale=self.scale,
            shear=self.shear, interpolation=vfunc.InterpolationMode.BILINEAR
        )
        gt = vfunc.affine(
            img=self.gt, angle=self.angle, translate=self.translate, scale=self.scale,
            shear=self.shear, interpolation=vfunc.InterpolationMode.NEAREST
        )

        return image, gt
