import torch
from torchvision import transforms

# SimCLR Transformations
def simclr_transforms(image_size: int, jitter: tuple = (0.4, 0.4, 0.2, 0.1),
                      p_blur: float = 1.0, p_solarize: float = 0.0,
                      normalize: list = [[0.485],[0.229]], gray_scale=True):
    
    trans_list = [transforms.RandomResizedCrop(image_size, scale=(0.4, 0.8), interpolation=transforms.InterpolationMode.BICUBIC),
                  transforms.RandomHorizontalFlip(p=0.5),
                  transforms.ConvertImageDtype(torch.float32), 
                  transforms.RandomApply([transforms.ColorJitter(*jitter)], p=0.8)]

    #If image is not grayscale add RandomGrayscale
    if not gray_scale:
        trans_list.append(transforms.RandomGrayscale(p=0.2))
    # Turn off blur for small images
    if image_size<=32:
        p_blur = 0.0
    # Add Gaussian blur
    if p_blur==1.0:
        trans_list.append(transforms.GaussianBlur(image_size//20*2+1))
    elif p_blur>0.0:
        trans_list.append(transforms.RandomApply([transforms.GaussianBlur(image_size//20*2+1)], p=p_blur))
    # Add RandomSolarize
    if p_solarize>0.0:
        trans_list.append(transforms.RandomSolarize(0.42, p=p_solarize))
    
    if normalize:
        trans_list.extend([transforms.Normalize(*normalize)])
    
    return transforms.Compose(trans_list)


# A wrapper that performs returns two augmented images
class TwoTransform(object):
    """Applies data augmentation two times."""

    def __init__(self, base_transform, sec_transform = None, temporal=False):
        self.base_transform = base_transform
        self.sec_transform = base_transform if sec_transform is None else sec_transform
        self.temporal = temporal

    def __call__(self, x):
        x1 = self.base_transform(x)
        if self.temporal:
            return x1
        x2 = self.sec_transform(x)
        return x1,x2


class VICReg_augmentaions(TwoTransform):
    def __init__(self, image_size, normalize=[[0.485],[0.229]], gray_scale=True, temporal=False):
        self.trans1 = simclr_transforms(image_size,
                                   p_blur = 0.8,
                                   p_solarize = 0.2,
                                   normalize = normalize,
                                   gray_scale = gray_scale)
        
        self.trans2 = simclr_transforms(image_size,
                                   p_blur = 0.8,
                                   p_solarize = 0.2,
                                   normalize = normalize,
                                   gray_scale = gray_scale)
        
        # If not temporal then it applies standart VicReg augmentations
        self.temporal = temporal

    def __call__(self, x):
        x1 = self.trans1(x)
        if self.temporal:
            return x1
        x2 = self.sec_transform(x)
        return x1,x2