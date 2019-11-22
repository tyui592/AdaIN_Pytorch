import torch
import torch.nn as nn
import torchvision

from utils import batch_wct

class Style_Transfer_Network(nn.Module):
    def __init__(self, layers=[1, 6, 11, 20]):
        super(Style_Transfer_Network, self).__init__()        
        self.encoder = Encoder(layers)
        self.decoder = Decoder(layers)        
        self.adain = AdaIN()
    
    def forward(self, content, styles, style_strength=1.0, interpolation_weights=None, masks=None, preserve_color=False, train=False):
        if interpolation_weights is None:
            interpolation_weights = [1/len(styles)] * len(styles)
        if masks is None:
            masks = [1] * len(styles)
            
        # encode the content image
        content_feature = self.encoder(content)

        # encode multiple style images
        style_features = []
        for style in styles:
            if preserve_color:
                style = batch_wct(style, content)
            style_features.append(self.encoder(style))
        
        # style transform
        transformed_feature = []
        for style_feature, interpolation_weight, mask in zip(style_features, interpolation_weights, masks):
            if isinstance(mask, torch.Tensor):
                b, c, h, w = content_feature[-1].size()
                mask = torch.nn.functional.interpolate(mask, size=(h, w))
            transformed_feature.append(self.adain(content_feature[-1], style_feature[-1], style_strength) * interpolation_weight * mask)
        transformed_feature = sum(transformed_feature)
        
        # decode the stylized feature
        stylized_image = self.decoder(transformed_feature)

        # get output image and transformed feature for content loss calulcation
        if train:
            return stylized_image, transformed_feature
        else:
            return stylized_image
    
class AdaIN(nn.Module):
    def __init__(self):
        super(AdaIN, self).__init__()
    
    def forward(self, content, style, style_strength=1.0, eps=1e-5):
        b, c, h, w = content.size()
        
        content_std, content_mean = torch.std_mean(content.view(b, c, -1), dim=2, keepdim=True)
        style_std, style_mean = torch.std_mean(style.view(b, c, -1), dim=2, keepdim=True)
    
        normalized_content = (content.view(b, c, -1) - content_mean)/(content_std+eps)
        
        stylized_content = (normalized_content * style_std) + style_mean

        output = (1-style_strength)*content + style_strength*stylized_content.view(b, c, h, w)
        return output
    
class Encoder(nn.Module):
    def __init__(self,  layers=[1, 6, 11, 20]):        
        super(Encoder, self).__init__()
        vgg = torchvision.models.vgg19(pretrained=True).features
        
        self.encoder = nn.ModuleList()
        temp_seq = nn.Sequential()
        for i in range(max(layers)+1):
            temp_seq.add_module(str(i), vgg[i])
            if i in layers:
                self.encoder.append(temp_seq)
                temp_seq = nn.Sequential()
        
    def forward(self, x):
        features = []
        for layer in self.encoder:
            x = layer(x)
            features.append(x)
        return features

class Decoder(nn.Module):
    def __init__(self, layers=[1, 6, 11, 20]):
        super(Decoder, self).__init__()
        vgg = torchvision.models.vgg19(pretrained=False).features
        
        self.decoder = nn.ModuleList()
        temp_seq  = nn.Sequential()
        count = 0
        for i in range(max(layers)-1, -1, -1):
            if isinstance(vgg[i], nn.Conv2d):
                # get number of in/out channels
                out_channels = vgg[i].in_channels
                in_channels = vgg[i].out_channels
                kernel_size = vgg[i].kernel_size

                # make a [reflection pad + convolution + relu] layer
                temp_seq.add_module(str(count), nn.ReflectionPad2d(padding=(1,1,1,1)))
                count += 1
                temp_seq.add_module(str(count), nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size))
                count += 1
                temp_seq.add_module(str(count), nn.ReLU())
                count += 1

            # change down-sampling(MaxPooling) --> upsampling
            elif isinstance(vgg[i], nn.MaxPool2d):
                temp_seq.add_module(str(count), nn.Upsample(scale_factor=2))
                count += 1

            if i in layers:
                self.decoder.append(temp_seq)
                temp_seq  = nn.Sequential()

        # append last conv layers without ReLU activation
        self.decoder.append(temp_seq[:-1])    
        
    def forward(self, x):
        y = x
        for layer in self.decoder:
            y = layer(y)
        return y
