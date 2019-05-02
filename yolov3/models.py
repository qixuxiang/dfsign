import os

import torch.nn.functional as F

from utils.parse_config import *
from utils.utils import *

import pdb

def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    yolo_layer_count = 0
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1],
                                                        out_channels=filters,
                                                        kernel_size=kernel_size,
                                                        stride=int(module_def['stride']),
                                                        padding=pad,
                                                        bias=not bn))
            if bn:
                modules.add_module('batch_norm_%d' % i, nn.BatchNorm2d(filters))
            if module_def['activation'] == 'leaky':
                modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1, inplace=True))

        elif module_def['type'] == 'maxpool':
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            if kernel_size == 2 and stride == 1:
                modules.add_module('_debug_padding_%d' % i, nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module('maxpool_%d' % i, maxpool)

        elif module_def['type'] == 'upsample':
            # upsample = nn.Upsample(scale_factor=int(module_def['stride']), mode='nearest')  # WARNING: deprecated
            upsample = Upsample(scale_factor=int(module_def['stride']))
            modules.add_module('upsample_%d' % i, upsample)

        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def['layers'].split(',')]
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
            modules.add_module('route_%d' % i, EmptyLayer())

        elif module_def['type'] == 'shortcut':
            filters = output_filters[int(module_def['from'])]
            modules.add_module('shortcut_%d' % i, EmptyLayer())

        elif module_def['type'] == 'yolo':
            anchor_idxs = [int(x) for x in module_def['mask'].split(',')]
            # Extract anchors
            anchors = [float(x) for x in module_def['anchors'].split(',')]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            nC = int(module_def['classes'])  # number of classes
            img_size = int(hyperparams['height'])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, nC, img_size, yolo_layer_count, cfg=hyperparams['cfg'])
            modules.add_module('yolo_%d' % i, yolo_layer)
            yolo_layer_count += 1

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class Upsample(nn.Module):
    # Custom Upsample layer (nn.Upsample gives deprecated warning message)

    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nC, img_size, yolo_layer, cfg):
        super(YOLOLayer, self).__init__()

        self.nC = nC  # number of classes (80)
        self.nA = len(anchors)  # number of anchors (3)
        self.register_buffer('anchors', torch.FloatTensor(anchors))
        self.register_buffer('img_size', torch.FloatTensor([32]))
        self.register_buffer('nG', torch.FloatTensor([1]))

    def forward(self, p, img_size, var=None):
        bs, nG = p.shape[0], p.shape[-1]
        if self.img_size != img_size:
            self.img_size.copy_(torch.FloatTensor([img_size]))
            self.nG.copy_(torch.FloatTensor([nG]))
        
        stride, grid_xy, anchor_vec, anchor_wh = self.create_grids(p.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.nA, self.nC + 5, nG, nG).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p

        else:  # inference
            io = p.clone()  # inference output
            io[..., 0:2] = torch.sigmoid(io[..., 0:2]) + grid_xy  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * anchor_wh  # wh yolo method
            io[..., :4] *= stride
            io[..., 4] = torch.sigmoid(io[..., 4])  # p_conf
            io[..., 5:] = F.softmax(io[..., 5:], dim=4)  # p_cls

            # reshape from [1, 3, 13, 13, 85] to [1, 507, 85]
            return io.view(bs, -1, 5 + self.nC), p

    def create_grids(self, device='cpu'):
        stride = self.img_size / self.nG

        nG = int(self.nG.item())
        # build xy offsets
        grid_x = torch.arange(nG).repeat((nG, 1)).view((1, 1, nG, nG)).float()
        grid_y = grid_x.permute(0, 1, 3, 2)
        grid_xy = torch.stack((grid_x, grid_y), 4).to(device)

        # build wh gains
        anchor_vec = self.anchors / stride
        anchor_wh = anchor_vec.view(1, self.nA, 1, 1, 2).to(device)

        return stride, grid_xy, anchor_vec, anchor_wh


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, cfg_path, img_size=416):
        super(Darknet, self).__init__()

        self.module_defs = parse_model_cfg(cfg_path)
        self.module_defs[0]['cfg'] = cfg_path
        self.module_defs[0]['height'] = img_size
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.yolo_layers = get_yolo_layers(self)

    def forward(self, x, var=None):
        img_size = x.shape[-1]
        layer_outputs = []
        output = []

        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = module_def['type']
            if mtype in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif mtype == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                if len(layer_i) == 1:
                    x = layer_outputs[layer_i[0]]
                else:
                    x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif mtype == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif mtype == 'yolo':
                x = module[0](x, img_size)
                output.append(x)
            layer_outputs.append(x)

        if self.training:
            return output
        else:
            io, p = list(zip(*output))  # inference output, training output
            return torch.cat(io, 1), p


def get_yolo_layers(model):
    a = [module_def['type'] == 'yolo' for module_def in model.module_defs]
    return [i for i, x in enumerate(a) if x]  # [82, 94, 106] for yolov3


def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'
    # cutoff: save layers between 0 and cutoff (if cutoff = -1 all are saved)
    weights_file = weights.split(os.sep)[-1]

    # Try to download weights if not available locally
    if not os.path.isfile(weights):
        try:
            os.system('wget https://pjreddie.com/media/files/' + weights_file + ' -O ' + weights)
        except IOError:
            print(weights + ' not found')

    # Establish cutoffs
    if weights_file == 'darknet53.conv.74':
        cutoff = 75
    elif weights_file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Open the weights file
    fp = open(weights, 'rb')
    header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values

    # Needed to write header when saving weights
    self.header_info = header

    self.seen = header[3]  # number of images seen during training
    weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
    fp.close()

    ptr = 0
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            if module_def['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                # Load conv. bias
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w

    return cutoff


def save_weights(self, path, cutoff=-1):
    fp = open(path, 'wb')
    self.header_info[3] = self.seen  # number of images seen during training
    self.header_info.tofile(fp)

    # Iterate through layers
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            # If batch norm, load bn first
            if module_def['batch_normalize']:
                bn_layer = module[1]
                bn_layer.bias.data.cpu().numpy().tofile(fp)
                bn_layer.weight.data.cpu().numpy().tofile(fp)
                bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                bn_layer.running_var.data.cpu().numpy().tofile(fp)
            # Load conv bias
            else:
                conv_layer.bias.data.cpu().numpy().tofile(fp)
            # Load conv weights
            conv_layer.weight.data.cpu().numpy().tofile(fp)

    fp.close()
