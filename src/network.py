import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import dataloader
import torchvision.datasets as datasets
import torchvision.transforms.functional as F

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Device: {device}")


################################################################################################
# Double Convolution layer
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        kernel_size = 3
        padding = 1
        stride = 1
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, padding, stride, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size, padding, stride, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    # Moving forward inside the layer
    def forward(self, x):
        return self.conv(x)


# Model
class UNET(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=2,
        features=[8, 16, 32, 64, 128, 256, 512],  # , 1024],
    ):
        super(UNET, self).__init__()
        self.downs = (
            nn.ModuleList()
        )  # a way for the model to evaluate while moving down (through the encoder)
        self.ups = (
            nn.ModuleList()
        )  # a way for the model to evaluate while moving up (through the decoder)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (moving down)
        for feature in features:
            self.downs.append(
                DoubleConv(in_channels, feature)
            )  # Moving from the in_channels to the first output channel (in the features list)
            in_channels = feature  # To move from 64 to 256, etc

        # Decoder (moving up)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2
                )  # We use feature*2 because we concatenate - skip connections)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Final Convolution Layer (Upscaling)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        # Move forward through the model

    def forward(self, x):
        skip_connections = []
        # Moving through the model
        for idx, down in enumerate(self.downs):
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

            # Reaching the Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Upsampling
        # Here we do up and double-conv steps
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Here, essentially, we do the ConvTranspose2d

            skip_connection = skip_connections[idx // 2]  # We add the skip connection

            # If the size of the previous output does not match the skip connection to concatenate
            if x.shape != skip_connection.shape:
                x = F.resize(
                    x, size=skip_connection.shape[2:], antialias=True
                )  # We may have a difference of 1 pixel (insignificant)

            concat_skip = torch.cat(
                (skip_connection, x), dim=1
            )  # We concatenate them along the channel dimension (dim=1)
            x = self.ups[idx + 1](concat_skip)
        return self.final_conv(x)


#################################################################
