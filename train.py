import time

import torch
import torch.nn as nn

from network import Style_Transfer_Network, Encoder
from utils import Image_Folder, lastest_average_value, imsave

# loss
mse_loss = nn.MSELoss(reduction='mean')

def calc_Contenet_Loss(source, target):
    content_loss = mse_loss(source, target)
    return content_loss

def calc_MeanStd_Loss(features, targets, weights=None):
    if weights is None:
        weights = [1/len(features)] * len(features)
    
    loss = 0
    for feature, target, weight in zip(features, targets, weights):
        b, c, h, w = feature.size()
        feature_std, feature_mean = torch.std_mean(feature.view(b, c, -1), dim=2)
        target_std, target_mean = torch.std_mean(target.view(b, c, -1), dim=2)
        loss += (mse_loss(feature_std, target_std) + mse_loss(feature_mean, target_mean))*weight
    return loss

# optimizer
def _optimizer(style_transfer_network, lr):
    for param in style_transfer_network.encoder.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam(style_transfer_network.decoder.parameters(), lr=lr)
    return optimizer    

# train
def train_network(args):
    # set device
    device = torch.device("cuda" if args.gpu_no >= 0 else "cpu")

    # style transfer network
    style_transfer_network = Style_Transfer_Network(args.layers).to(device)

    # datasets
    content_dataset = Image_Folder(args.content_dir, args.imsize, args.cropsize, args.cencrop)
    style_dataset = Image_Folder(args.style_dir, args.imsize, args.cropsize, args.cencrop)
    # optimizer
    optimizer = _optimizer(style_transfer_network, args.lr)
    
    # loss network
    loss_network = Encoder().to(device)
    for param in loss_network.parameters():
        param.requires_grad = False
    loss_logs = {'content':[], 'style':[], 'total':[]}
    
    # start training
    for iteration in range(args.max_iter):
        # image loading
        content_loader = torch.utils.data.DataLoader(content_dataset, batch_size=args.batch_size, shuffle=True)
        style_loader = torch.utils.data.DataLoader(style_dataset, batch_size=args.batch_size, shuffle=True)
        content_image = next(iter(content_loader)).to(device)
        style_image = next(iter(style_loader)).to(device)
        
        output_image, transformed_feature = style_transfer_network(content_image, [style_image], train=True)

        # loss calculation
        output_features = loss_network(output_image)
        style_features = loss_network(style_image)
        
        content_loss = calc_Contenet_Loss(output_features[-1], transformed_feature)
        loss_logs['content'].append(content_loss.item())
        style_loss = calc_MeanStd_Loss(output_features, style_features)
        loss_logs['style'].append(style_loss.item())
        total_loss = content_loss + style_loss * args.style_weight
        loss_logs['total'].append(total_loss.item())

        # optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # check training
        if (iteration+1) % args.check_iter == 0:
            loss_str = '%s: iteration: [%d/%d]'%(time.ctime(), iteration+1, args.max_iter)
            for key, value in loss_logs.items():
                loss_str += '\t%s: %2.4f'%(key, lastest_average_value(value))
            print(loss_str)
            imsave(torch.cat([content_image, style_image, output_image], dim=0), "training_image.png")
            torch.save({'iteration':iteration+1,
                'state_dict':style_transfer_network.state_dict(),
                'loss_seq':loss_logs},
                'check_point.pth')

    return style_transfer_network
