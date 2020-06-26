import torch.nn as nn
import torch
import torch.nn.functional as F
from debugUtils import out

padding_param = 0
__filePrefix__ = "MODELS"

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        #print(__filePrefix__, "forward Residual Block IN: ", out(x))
        res = x + self.conv_block(x)
        #print(__filePrefix__, "forward Residual Block OUT: ", out(res))
        return res

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=padding_param),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        # In: (1, 256, 50, 76)
        scale_factor = 2
        
        #model += [
        #        nn.Conv2d(in_features, in_features*4, kernel_size=3, padding=padding_param),
        #        nn.PixelShuffle(2),
        #        nn.InstanceNorm2d(out_features),
        #        nn.ReLU(inplace=True)
        #]

        #(1, 64, 96, 148)
        #for _ in range(2):
        #    model += [
        #        nn.Conv2d(in_features, in_features * 4, kernel_size=3, padding=padding_param),
        #        nn.PixelShuffle(scale_factor),
        #        nn.InstanceNorm2d(in_features),
        #        nn.ReLU(inplace=True)
        #    ]
        #    in_features = in_features*4
        #(1, 256, 188, 292)
        
        model += [
                nn.Conv2d(in_features, in_features*4, kernel_size=3, padding=padding_param+2),
                nn.PixelShuffle(2),
                nn.InstanceNorm2d(in_features),
                nn.ReLU(inplace=True)
        ]
        model += [
                nn.Conv2d(in_features, in_features, kernel_size=3, padding=padding_param),
                nn.PixelShuffle(2),
                nn.InstanceNorm2d(in_features),
                nn.ReLU(inplace=True)
        ]
        # This as is returns (1, 64, 204, 308)
        # However, one might ask himself whether this is optimal. Let's just assume that yes       

        
        #out_features = in_features//2
        #for _ in range(2):
        #    model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=padding_param, output_padding=0),
        #                nn.InstanceNorm2d(out_features),
        #                nn.ReLU(inplace=True) ]
        #    in_features = out_features
        #    out_features = in_features//2
        #(1, 64, 204, 307)


        #Output layer: Expects (1, 64, 204, 307), but we may change feature amount
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        #print(__filePrefix__, "forward Generator IN: ", out(x))
        res = self.model(x)
        s = out(res)
        #if (s[2] == 408 and s[3]==616): #616 is the thing we wanna drop
        #    res = res[::1,::1,::1,0:614:1]
        if (s[2] == 203 and s[3]==307): #We wanna pad 203 to 204
            padding = (0,0,1,0)
            res=F.pad(res, padding, "constant", 0)
        elif (s[2] == 204 and s[3]==308): #We drop one column
            res = res[::1,::1,::1,0:307:1]
        
        #print(__filePrefix__, "forward Generator OUT: ", out(res))
        return res

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=padding_param),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=padding_param),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=padding_param),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=padding_param),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=padding_param)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        #print(__filePrefix__, "forward Discriminator IN: ", out(x))
        x =  self.model(x)
        #print(__filePrefix__, "forward Discriminator TMP: ", out(x))
        res = F.avg_pool2d(x, x.size()[2:]).view(x.size()[0])
        #print(__filePrefix__, "forward Discriminator OUT: ", out(res))
        # Average pooling and flatten
        return res
