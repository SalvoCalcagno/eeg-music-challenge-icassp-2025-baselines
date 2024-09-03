# This is the model presented in the work: S. Palazzo, C. Spampinato, I. Kavasidis, D. Giordano, J. Schmidt, M. Shah, Decoding Brain Representations by 
# Multimodal Learning of Neural Activity and Visual Features,  IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, 2020, doi: 10.1109/TPAMI.2020.2995909
import torch
import torch.nn as nn

from src.models.layers import * 

class FeaturesExtractor(nn.Module):
    def __init__(self, in_channels, temp_channels, out_channels, input_width, in_height,
                 temporal_kernel, temporal_stride, temporal_dilation_list, num_temporal_layers,
                 num_spatial_layers, spatial_stride, num_residual_blocks, down_kernel, down_stride):
        super().__init__()

        self.temporal_block = TemporalBlock(
            in_channels, temp_channels, num_temporal_layers, temporal_kernel, temporal_stride, temporal_dilation_list, input_width
        )

        self.spatial_block = SpatialBlock(
            temp_channels * num_temporal_layers, out_channels, num_spatial_layers, spatial_stride, in_height
        )

        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(
                    out_channels * num_spatial_layers, out_channels * num_spatial_layers
                ),
                ConvLayer2D(
                    out_channels * num_spatial_layers, out_channels * num_spatial_layers, down_kernel, down_stride, 0, 1
                )
            ) for i in range(num_residual_blocks)
        ])

        self.final_conv = ConvLayer2D(
            out_channels * num_spatial_layers, out_channels, down_kernel, 1, 0, 1
        )

    def forward(self, x):
        #print(f"[FE INPUT] {x.shape}")
        out = self.temporal_block(x)
        #print(f"[TEMPORAL] {out.shape}")
        out = self.spatial_block(out)
        #print(f"[SPATIAL] {out.shape}")
        if len(self.res_blocks) > 0:
            for res_block in self.res_blocks:
                out = res_block(out)
                #print(f"[RESIDUAL] {out.shape}")
        out = self.final_conv(out)
        #print(f"[FINAL] {out.shape}")

        return out

class Model(nn.Module):
    '''The model for EEG classification.
    The imput is a tensor where each row is a channel the recorded signal and each colums is a time sample.
    The model performs different 2D to extract temporal e spatial information.
    The output is a vector of classes where the maximum value is the predicted class.
    Args:
        in_channels: number of input channels
        temp_channels: number of features of temporal block
        out_channels: number of features before classification
        num_classes: number possible classes
        embedding_size: size of the embedding vector
        input_width: width of the input tensor (necessary to compute classifier input size)
        input_height: height of the input tensor (necessary to compute classifier input size)
        temporal_dilation_list: list of dilations for temporal convolutions, second term must be even
        temporal_kernel: size of the temporal kernel, second term must be even (default: (1, 32))
        temporal_stride: size of the temporal stride, control temporal output size (default: (1, 2))
        num_temp_layers: number of temporal block layers
        num_spatial_layers: number of spatial layers
        spatial_stride: size of the spatial stride
        num_residual_blocks: the number of residual blocks
        down_kernel: size of the bottleneck kernel
        down_stride: size of the bottleneck stride
        '''
    def __init__(self, args):
        super(Model, self).__init__()
        args_defaults=dict(
            in_channels=1, 
            temp_channels=10, 
            out_channels=50, 
            num_classes=40, 
            embedding_size=1000,
            input_width=440, 
            input_height=128, 
            temporal_dilation_list=[(1,1),(1,2),(1,4),(1,8),(1,16)],
            temporal_kernel=(1,33), 
            temporal_stride=(1,2),
            num_temp_layers=4,
            num_spatial_layers=4, 
            spatial_stride=(2,1), 
            num_residual_blocks=4, 
            down_kernel=3, 
            down_stride=2
        )
        for arg,default in args_defaults.items():
            setattr(self, arg, args[arg] if arg in args and args[arg] is not None else default)

        self.encoder = FeaturesExtractor(
            self.in_channels, self.temp_channels, self.out_channels, 
            self.input_width, self.input_height, self.temporal_kernel, 
            self.temporal_stride, self.temporal_dilation_list, 
            self.num_temp_layers, self.num_spatial_layers, 
            self.spatial_stride, self.num_residual_blocks, 
            self.down_kernel, self.down_stride)

        encoding_size = self.encoder(torch.zeros(1, self.in_channels, self.input_height, self.input_width)).contiguous().view(-1).size()[0]

        self.classifier = nn.Sequential(
            nn.Linear(encoding_size, self.embedding_size),
            nn.ReLU(True),
            nn.Linear(self.embedding_size, self.num_classes), 
        )

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        #print(f"[INPUT] {x.shape}")
        out = self.encoder(x)
        #print(f"[ENCODER] {out.shape}")
        out = out.view(x.size(0), -1)
        #print(f"[FLATTEN] {out.shape}")
        out = self.classifier(out)
        #print(f"[CLASSIFIER] {out.shape}")
        return out