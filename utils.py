import os
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms

def _normalizer(denormalize=False):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]    
    
    if denormalize:
        MEAN = [-mean/std for mean, std in zip(MEAN, STD)]
        STD = [1/std for std in STD]
    
    return transforms.Normalize(mean=MEAN, std=STD)

def _transformer(imsize=None, cropsize=None, cencrop=False):
    transformer = []
    if imsize:
        transformer.append(transforms.Resize(imsize))
    if cropsize:
        if cencrop:
            transformer.append(transforms.CenterCrop(cropsize))
        else:
            transformer.append(transforms.RandomCrop(cropsize))
            
    transformer.append(transforms.ToTensor())
    transformer.append(_normalizer())
    return transforms.Compose(transformer)

def imload(path, imsize=None, cropsize=None, cencrop=False):
    transformer = _transformer(imsize=imsize, cropsize=cropsize, cencrop=cencrop)
    return transformer(Image.open(path).convert("RGB")).unsqueeze(0)

def maskload(path):
    mask = Image.open(path).convert('L')
    return transforms.functional.to_tensor(mask).unsqueeze(0)

def imsave(tensor, path):
    denormalize = _normalizer(denormalize=True)    
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = torchvision.utils.make_grid(tensor)
    torchvision.utils.save_image(denormalize(tensor).clamp_(0.0, 1.0), path)    
    return None

def imshow(tensor):
    denormalize = _normalizer(denormalize=True)    
    if tensor.is_cuda:
        tensor = tensor.cpu()    
    tensor = torchvision.utils.make_grid(denormalize(tensor.squeeze()))
    image = transforms.functional.to_pil_image(tensor.clamp_(0.0, 1.0))
    return image


class Image_Folder(torch.utils.data.Dataset):
    def __init__(self, root_path, imsize, cropsize, cencrop):
        super(Image_Folder, self).__init__()

        self.root_path = root_path
        self.file_names = sorted(os.listdir(self.root_path))
        self.transformer = _transformer(imsize, cropsize, cencrop)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root_path + self.file_names[index])).convert("RGB")
        return self.transformer(image)
    
    
def lastest_average_value(values, length=100):
    if len(values) < length:
        length = len(values)
    return sum(values[-length:])/length

def mean_covsqrt(f, inverse=False, eps=1e-10):
    c, h, w = f.size()
    
    f_mean = torch.mean(f.view(c, h*w), dim=1, keepdim=True)
    f_zeromean = f.view(c, h*w) - f_mean
    f_cov = torch.mm(f_zeromean, f_zeromean.t())

    u, s, v = torch.svd(f_cov)

    k = c
    for i in range(c):
        if s[i] < eps:
            k = i
            break
            
    if inverse:
        p = -0.5
    else:
        p = 0.5
        
    f_covsqrt = torch.mm(torch.mm(v[:, 0:k], torch.diag(s[0:k].pow(p))), v[:, 0:k].t())
    return f_mean, f_covsqrt

def whitening(f):
    c, h, w = f.size()

    f_mean, f_inv_covsqrt = mean_covsqrt(f, inverse=True)
    
    whiten_f = torch.mm(f_inv_covsqrt, f.view(c, h*w) - f_mean)
    
    return whiten_f.view(c, h, w)

def coloring(f, t):
    f_c, f_h, f_w = f.size()
    t_c, t_h, t_w = t.size()
    
    t_mean, t_covsqrt = mean_covsqrt(t)
    
    colored_f = torch.mm(t_covsqrt, f.view(f_c, f_h*f_w)) + t_mean
    
    return colored_f.view(f_c, f_h, f_w)

def batch_wct(source, target):
    whiten_source = batch_whitening(source)
    return batch_coloring(whiten_source, target)

def batch_whitening(f):
    b, c, h, w = f.size()

    whiten_f = torch.Tensor(b, c, h, w).type_as(f)
    for i, f_ in enumerate(torch.split(f, 1)):
        whiten_f[i] = whitening(f_.squeeze())
        
    return whiten_f

def batch_coloring(f, t):
    b, c, h, w = f.size()

    colored_f = torch.Tensor(b, c, h, w).type_as(f)
    for i, (f_, t_) in enumerate(zip(torch.split(f, 1), torch.split(t, 1))):
        colored_f[i] = coloring(f_.squeeze(), t_.squeeze())

    return colored_f

