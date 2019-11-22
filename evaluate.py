import torch

from utils import imload, imsave, maskload
from network import Style_Transfer_Network

def evaluate_network(args):
    device = torch.device('cuda' if args.gpu_no >=0 else 'cpu')

    if args.load_path is None:
        raise RuntimeError("Need a model to load !!")
    check_point = torch.load(args.load_path)

    transfer_network = Style_Transfer_Network().to(device)
    transfer_network.load_state_dict(check_point['state_dict'])

    content_img = imload(args.content, args.imsize, args.cropsize, args.cencrop).to(device)
    style_imgs = [imload(_style, args.imsize, args.cropsize, args.cencrop).to(device) for _style in args.style]

    masks = None
    if args.mask:
        masks = [maskload(mask).to(device) for mask in args.mask]

    with torch.no_grad():
        stylized_img = transfer_network(content_img, style_imgs, args.style_strength, args.interpolation_weights, masks, args.preserve_color)

    imsave(stylized_img, 'stylized_image.jpg')
    return stylized_img

