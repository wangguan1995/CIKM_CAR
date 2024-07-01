# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Code is heavily based on paper "Geometry-Informed Neural Operator for Large-Scale 3D PDEs", we use paddle to reproduce the results of the paper


import paddle
from functools import partial
from .base_model import BaseModel


def create_conv(in_channels, out_channels, kernel_size, order, num_groups,
    padding, is3d):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size(int or tuple): size of the convolving kernel
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'gcr' -> groupnorm + conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'bcr' -> batchnorm + conv + ReLU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        is3d (bool): is3d (bool): if True use Conv3d, otherwise use Conv2d
    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, 'Conv layer MUST be present'
    assert order[0
        ] not in 'rle', 'Non-linearity cannot be the first operation in the layer'
    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', paddle.nn.ReLU()))
        elif char == 'l':
            modules.append(('LeakyReLU', paddle.nn.LeakyReLU()))
        elif char == 'e':
            modules.append(('ELU', paddle.nn.ELU()))
        elif char == 'c':
            bias = not ('g' in order or 'b' in order)
            if is3d:
                conv = paddle.nn.Conv3D(in_channels=in_channels,
                    out_channels=out_channels, kernel_size=kernel_size,
                    padding=padding, bias_attr=bias)
            else:
                conv = paddle.nn.Conv2D(in_channels=in_channels,
                    out_channels=out_channels, kernel_size=kernel_size,
                    padding=padding, bias_attr=bias)
            modules.append(('conv', conv))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels
            if num_channels < num_groups:
                num_groups = 1
            assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm', paddle.nn.GroupNorm(num_groups=
                num_groups, num_channels=num_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is3d:
                bn = paddle.nn.BatchNorm3D
            else:
                bn = paddle.nn.BatchNorm2D
            if is_before_conv:
                modules.append(('batchnorm', bn(in_channels)))
            else:
                modules.append(('batchnorm', bn(out_channels)))
        else:
            raise ValueError(
                f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']"
                )
    return modules


class SingleConv(paddle.nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding
        is3d (bool): if True use Conv3d, otherwise use Conv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order=
        'gcr', num_groups=8, padding=1, is3d=True):
        super(SingleConv, self).__init__()
        for name, module in create_conv(in_channels, out_channels,
            kernel_size, order, num_groups, padding, is3d):
            self.add_sublayer(name=name, sublayer=module)


class DoubleConv(paddle.nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        is3d (bool): if True use Conv3d instead of Conv2d layers
    """

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3,
        order='gcr', num_groups=8, padding=1, is3d=True):
        super(DoubleConv, self).__init__()
        self.in_channels = in_channels
        if encoder:
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = (conv1_out_channels,
                out_channels)
        else:
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels
        self.add_sublayer(name='SingleConv1', sublayer=SingleConv(
            conv1_in_channels, conv1_out_channels, kernel_size, order,
            num_groups, padding=padding, is3d=is3d))
        self.add_sublayer(name='SingleConv2', sublayer=SingleConv(
            conv2_in_channels, conv2_out_channels, kernel_size, order,
            num_groups, padding=padding, is3d=is3d))


class Encoder(paddle.nn.Layer):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    from the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed by
    a basic module (DoubleConv or ResNetBlock).

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the convolving kernel
        apply_pooling (bool): if True use MaxPool3d before DoubleConv
        pool_kernel_size (int or tuple): the size of the window
        pool_type (str): pooling layer: 'max' or 'avg'
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        is3d (bool): use 3d or 2d convolutions/pooling operation
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3,
        apply_pooling=True, pool_kernel_size=2, pool_type='max',
        basic_module=DoubleConv, conv_layer_order='gcr', num_groups=8,
        padding=1, is3d=True):
        super(Encoder, self).__init__()
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                if is3d:
                    self.pooling = paddle.nn.MaxPool3D(kernel_size=
                        pool_kernel_size)
                else:
                    self.pooling = paddle.nn.MaxPool2D(kernel_size=
                        pool_kernel_size)
            elif is3d:
                self.pooling = paddle.nn.AvgPool3D(kernel_size=
                    pool_kernel_size, exclusive=False)
            else:
                self.pooling = paddle.nn.AvgPool2D(kernel_size=
                    pool_kernel_size, exclusive=False)
        else:
            self.pooling = None
        self.basic_module = basic_module(in_channels, out_channels, encoder
            =True, kernel_size=conv_kernel_size, order=conv_layer_order,
            num_groups=num_groups, padding=padding, is3d=is3d)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


class Decoder(paddle.nn.Layer):
    """
    A single module for decoder path consisting of the upsampling layer
    (either learned ConvTranspose3d or nearest neighbor interpolation)
    followed by a basic module (DoubleConv or ResNetBlock).

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the convolving kernel
        scale_factor (tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose3d, must reverse the MaxPool3d operation
            from the corresponding encoder
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        upsample (bool): should the input be upsampled
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3,
        scale_factor=(2, 2, 2), basic_module=DoubleConv, conv_layer_order=
        'gcr', num_groups=8, mode='nearest', padding=1, upsample=True, is3d
        =True):
        super(Decoder, self).__init__()
        if upsample:
            if basic_module == DoubleConv:
                self.upsampling = InterpolateUpsampling(mode=mode)
                self.joining = partial(self._joining, concat=True)
            else:
                self.upsampling = TransposeConvUpsampling(in_channels=
                    in_channels, out_channels=out_channels, kernel_size=
                    conv_kernel_size, scale_factor=scale_factor)
                self.joining = partial(self._joining, concat=False)
                in_channels = out_channels
        else:
            self.upsampling = NoUpsampling()
            self.joining = partial(self._joining, concat=True)
        self.basic_module = basic_module(in_channels, out_channels, encoder
            =False, kernel_size=conv_kernel_size, order=conv_layer_order,
            num_groups=num_groups, padding=padding, is3d=is3d)

    def forward(self, encoder_features, x):
        x = self.upsampling(encoder_features=encoder_features, x=x)
        x = self.joining(encoder_features, x)
        x = self.basic_module(x)
        return x

    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return paddle.concat(x=(encoder_features, x), axis=1)
        else:
            return encoder_features + x


def create_encoders(in_channels, f_maps, basic_module, conv_kernel_size,
    conv_padding, layer_order, num_groups, pool_kernel_size, is3d):
    encoders = []
    for i, out_feature_num in enumerate(f_maps):
        if i == 0:
            encoder = Encoder(in_channels, out_feature_num, apply_pooling=
                False, basic_module=basic_module, conv_layer_order=
                layer_order, conv_kernel_size=conv_kernel_size, num_groups=
                num_groups, padding=conv_padding, is3d=is3d)
        else:
            encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=
                basic_module, conv_layer_order=layer_order,
                conv_kernel_size=conv_kernel_size, num_groups=num_groups,
                pool_kernel_size=pool_kernel_size, padding=conv_padding,
                is3d=is3d)
        encoders.append(encoder)
    return paddle.nn.LayerList(sublayers=encoders)


def create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding,
    layer_order, num_groups, is3d):
    decoders = []
    reversed_f_maps = list(reversed(f_maps))
    for i in range(len(reversed_f_maps) - 1):
        if basic_module == DoubleConv:
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
        else:
            in_feature_num = reversed_f_maps[i]
        out_feature_num = reversed_f_maps[i + 1]
        decoder = Decoder(in_feature_num, out_feature_num, basic_module=
            basic_module, conv_layer_order=layer_order, conv_kernel_size=
            conv_kernel_size, num_groups=num_groups, padding=conv_padding,
            is3d=is3d)
        decoders.append(decoder)
    return paddle.nn.LayerList(sublayers=decoders)


class AbstractUpsampling(paddle.nn.Layer):
    """
    Abstract class for upsampling. A given implementation should upsample a given 5D input tensor using either
    interpolation or learned transposed convolution.
    """

    def __init__(self, upsample):
        super(AbstractUpsampling, self).__init__()
        self.upsample = upsample

    def forward(self, encoder_features, x):
        output_size = encoder_features.shape[2:]
        return self.upsample(x, output_size)


class InterpolateUpsampling(AbstractUpsampling):
    """
    Args:
        mode (str): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'. Default: 'nearest'
            used only if transposed_conv is False
    """

    def __init__(self, mode='nearest'):
        upsample = partial(self._interpolate, mode=mode)
        super().__init__(upsample)

    @staticmethod
    def _interpolate(x, size, mode):
        return paddle.nn.functional.interpolate(x=x, size=size, mode=mode, data_format="NCDHW")


class TransposeConvUpsampling(AbstractUpsampling):
    """
    Args:
        in_channels (int): number of input channels for transposed conv
            used only if transposed_conv is True
        out_channels (int): number of output channels for transpose conv
            used only if transposed_conv is True
        kernel_size (int or tuple): size of the convolving kernel
            used only if transposed_conv is True
        scale_factor (int or tuple): stride of the convolution
            used only if transposed_conv is True

    """

    def __init__(self, in_channels=None, out_channels=None, kernel_size=3,
        scale_factor=(2, 2, 2)):
        upsample = paddle.nn.Conv3DTranspose(in_channels=in_channels,
            out_channels=out_channels, kernel_size=kernel_size, stride=
            scale_factor, padding=1)
        super().__init__(upsample)


class NoUpsampling(AbstractUpsampling):

    def __init__(self):
        super().__init__(self._no_upsampling)

    @staticmethod
    def _no_upsampling(x, size):
        return x


def number_of_features_per_level(init_channel_number, num_levels):
    return [(init_channel_number * 2 ** k) for k in range(num_levels)]


class AbstractUNet(BaseModel):
    """
    Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the final 1x1 convolution,
            otherwise apply nn.Softmax. In effect only if `self.training == False`, i.e. during validation/testing
        basic_module: basic model for the encoder/decoder (DoubleConv, ResNetBlock, ....)
        layer_order (string): determines the order of layers in `SingleConv` module.
            E.g. 'crg' stands for GroupNorm3d+Conv3d+ReLU. See `SingleConv` for more info
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
            default: 4
        is_segmentation (bool): if True and the model is in eval mode, Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
        is3d (bool): if True the model is 3D, otherwise 2D, default: True
    """

    def __init__(self, in_channels, out_channels, final_sigmoid,
        basic_module, f_maps=64, layer_order='gcr', num_groups=8,
        num_levels=4, is_segmentation=False, conv_kernel_size=3,
        pool_kernel_size=2, conv_padding=1, is3d=True):
        super(AbstractUNet, self).__init__()
        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels
                )
        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, 'Required at least 2 levels in the U-Net'
        if 'g' in layer_order:
            assert num_groups is not None, 'num_groups must be specified if GroupNorm is used'
        self.encoders = create_encoders(in_channels, f_maps, basic_module,
            conv_kernel_size, conv_padding, layer_order, num_groups,
            pool_kernel_size, is3d)
        self.decoders = create_decoders(f_maps, basic_module,
            conv_kernel_size, conv_padding, layer_order, num_groups, is3d)
        if is3d:
            self.final_conv = paddle.nn.Conv3D(in_channels=f_maps[0],
                out_channels=out_channels, kernel_size=1)
        else:
            self.final_conv = paddle.nn.Conv2D(in_channels=f_maps[0],
                out_channels=out_channels, kernel_size=1)
        if is_segmentation:
            if final_sigmoid:
                self.final_activation = paddle.nn.Sigmoid()
            else:
                self.final_activation = paddle.nn.Softmax(axis=1)
        else:
            self.final_activation = None

    def forward(self, x):
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.insert(0, x)
        encoders_features = encoders_features[1:]
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(encoder_features, x)
        x = self.final_conv(x)
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)
        return x


class UNet3D(AbstractUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=False,
        f_maps=64, layer_order='gcr', num_groups=8, num_levels=4,
        is_segmentation=False, conv_padding=1, **kwargs):
        super(UNet3D, self).__init__(in_channels=in_channels, out_channels=
            out_channels, final_sigmoid=final_sigmoid, basic_module=
            DoubleConv, f_maps=f_maps, layer_order=layer_order, num_groups=
            num_groups, num_levels=num_levels, is_segmentation=
            is_segmentation, conv_padding=conv_padding, is3d=True)


class UNet3DWithSamplePoints(UNet3D):

    def __init__(self, in_channels: int, out_channels: int, hidden_channels:
        int, num_levels: int, use_position_input: bool=True):
        super(UNet3DWithSamplePoints, self).__init__(in_channels,
            hidden_channels, f_maps=hidden_channels, num_levels=num_levels)
        self.final_mlp = paddle.nn.Sequential(paddle.nn.Linear(in_features=
            hidden_channels, out_features=hidden_channels), paddle.nn.GELU(
            ), paddle.nn.Linear(in_features=hidden_channels, out_features=
            out_channels))
        self.use_position_input = use_position_input

    def forward(self, x, output_points):
        x = super(UNet3DWithSamplePoints, self).forward(x)
        output_points = output_points.unsqueeze(axis=2).unsqueeze(axis=2)
        x = paddle.nn.functional.grid_sample(x=x, grid=output_points,
            align_corners=False)
        x = x.squeeze(axis=3).squeeze(axis=3)
        x = x.transpose(perm=[0, 2, 1])
        x = self.final_mlp(x)
        return x

    def data_dict_to_input(self, data_dict):
        input_grid_features = data_dict['sdf'].unsqueeze(axis=1)

        if self.use_position_input:
            grid_points = data_dict['sdf_query_points']
            input_grid_features = paddle.concat(x=(input_grid_features,
                grid_points), axis=1)
        output_points = data_dict['vertices']
        return input_grid_features, output_points

    @paddle.no_grad()
    def eval_dict(self, data_dict, loss_fn=None, decode_fn=None, **kwargs):
        input_grid_features, output_points = self.data_dict_to_input(data_dict)
        pred_var = self(input_grid_features, output_points)
        true_var = None
        true_var_key = None
        if 'pressure' in data_dict.keys():
            true_var = data_dict['pressure'].unsqueeze(-1)
            true_var_key = 'pressure'
        elif "velocity" in data_dict.keys():
            true_var =  data_dict['velocity']
            true_var_key = 'velocity'
        elif "cd" in data_dict.keys():
            true_var =  data_dict['cd']
            true_var_key = 'cd'
        else:
            raise NotImplementedError("only pressure velocity works")
        return {'l2 eval loss': loss_fn(pred_var, true_var), true_var_key:true_var}

    def loss_dict(self, data_dict, loss_fn=None, **kwargs):
        input_grid_features, output_points = self.data_dict_to_input(data_dict)
        pred_var = self(input_grid_features, output_points)
        true_var = None
        if 'pressure' in data_dict.keys():
            true_var = data_dict['pressure'].unsqueeze(-1)
        elif "velocity" in data_dict.keys():
            true_var =  data_dict['velocity']
        elif "cd" in data_dict.keys():
            true_var =  data_dict['cd']
        else:
            raise NotImplementedError("only pressure velocity works")

        return {'loss': loss_fn(pred_var, true_var)}

