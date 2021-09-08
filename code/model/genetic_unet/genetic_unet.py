import torch
import torch.nn as nn
import numpy as np
from scipy.special import comb
from .blocks import get_func


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


def flatten(input_list):
    output_list = []
    while True:
        if input_list == []:
            break
        for index, value in enumerate(input_list):

            if type(value) == list:
                input_list = value + input_list[index + 1:]
                break
            else:
                output_list.append(value)
                input_list.pop(index)
                break
    return output_list


def check_active(node_num, connect_gene):
    active = [None for _ in range(node_num)]
    node_connect = []
    j = 1
    i = 0
    for _ in range(node_num - 1):
        node_connect.append(connect_gene[i:i + j])
        i = i + j
        j += 1
    for p, node in enumerate(node_connect):
        if p != node_num - 2:
            if sum(node) >= 1:
                active[p + 1] = True
    for k in range(node_num):
        for node in node_connect:
            if k < len(node) and k != node_num - 1:
                if node[k] == 1:
                    active[k] = True

            elif k == node_num - 1:
                if sum(node) >= 1:
                    active[k] = True

    pre_index = [None for _ in range(node_num)]
    for m in range(node_num):
        if active[m]:
            if m == 0:
                pre_index[m] = [m]
            else:
                p_index = []
                if sum(node_connect[m - 1]) == 0:
                    pre_index[m] = [0]
                else:
                    for index, con in enumerate(node_connect[m - 1]):
                        if con == 1:
                            p_index.append(index + 1)
                    if len(p_index) > 0:
                        pre_index[m] = p_index
    out_index = []
    for t in range(node_num):
        pre_index_ = flatten(pre_index[t + 1:])
        if active[t] and t + 1 not in pre_index_:
            out_index.append(t + 1)
    if sum([1 for act in active if act is not None]) == 0:
        out_index = [0]
    return active, pre_index, out_index


class UnetBlock(nn.Module):
    def __init__(self, base_ch, active, pre_index, out_index, node_func_type):
        super(UnetBlock, self).__init__()
        self.active = active
        self.pre_index = pre_index
        self.out_index = out_index
        channels = [None for _ in range(len(active))]
        middle_channel = base_ch
        for i in range(len(self.active)):
            if self.active[i]:
                for j, index in enumerate(self.pre_index[i]):
                    if j == 0 and index == 0:
                        channels[i] = [base_ch, middle_channel]
                    else:
                        channels[i] = [middle_channel, middle_channel]

        self.node_operations = []
        for i in range(len(self.active)):
            if self.active[i]:
                self.node_operations.append(
                    get_func(node_func_type, in_channel=channels[i][0], out_channel=channels[i][1]))
            else:
                self.node_operations.append(None)

        self.node_operations = nn.ModuleList(self.node_operations)

        if self.out_index == [0]:
            middle_channel = base_ch

        self.init_conv = get_func(node_func_type, in_channel=base_ch, out_channel=base_ch)
        self.final_conv = get_func(node_func_type, in_channel=middle_channel, out_channel=base_ch)
        self.outputs = [None for _ in range(len(self.active) + 1)]

    def forward(self, x):
        outputs = self.outputs
        x = self.init_conv(x)
        outputs[0] = x
        for i in range(1, len(self.active) + 1):
            if self.active[i - 1]:
                for j, index in enumerate(self.pre_index[i - 1]):
                    if j == 0:
                        input_t = outputs[index]
                    else:
                        input_t = input_t + outputs[index]
                outputs[i] = self.node_operations[i - 1](input_t)
        for y, o_index in enumerate(self.out_index):
            if y == 0:
                out = outputs[o_index]
            else:
                out = out + outputs[o_index]
        out = self.final_conv(out)
        return out
"""
The gene of the searhed architecture on DRIVE is  [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0,
                                         1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1,
                                         1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1]
"""
class Net(nn.Module):
    def __init__(self, gene, model_settings, in_ch=3, out_ch=1):
        super(Net, self).__init__()
        channel = model_settings['channel']
        sample_num = model_settings['sample_num']
        en_func_type = model_settings['en_func_type']
        de_func_type = model_settings['de_func_type']
        en_node_num_list = model_settings['en_node_num_list']
        de_node_num_list = model_settings['de_node_num_list']

        de_func_type_num = len(de_func_type)
        en_func_type_num = len(en_func_type)

        de_node_func_gene_len = int(np.ceil(np.log2(de_func_type_num)))
        en_node_func_gene_len = int(np.ceil(np.log2(en_func_type_num)))

        de_connect_gene_len_list = [None for _ in range(len(de_node_num_list))]
        en_connect_gene_len_list = [None for _ in range(len(en_node_num_list))]

        for i in range(len(de_node_num_list)):
            de_connect_gene_len_list[i] = int(comb(de_node_num_list[i], 2))
        for i in range(len(en_node_num_list)):
            en_connect_gene_len_list[i] = int(comb(en_node_num_list[i], 2))

        de_gene_len_list = [None for _ in range(len(de_node_num_list))]
        en_gene_len_list = [None for _ in range(len(en_node_num_list))]

        for i in range(len(de_node_num_list)):
            de_gene_len_list[i] = de_node_func_gene_len + de_connect_gene_len_list[i]
        for i in range(len(en_node_num_list)):
            en_gene_len_list[i] = en_node_func_gene_len + en_connect_gene_len_list[i]

        gene_len = sum(de_gene_len_list) + sum(en_gene_len_list)

        de_gene_list = [None for _ in range(len(de_node_num_list))]
        en_gene_list = [None for _ in range(len(en_node_num_list))]

        end_point = gene_len
        for i in range(len(de_node_num_list) - 1, -1, -1):
            de_gene_list[i] = gene[end_point - de_gene_len_list[i]:end_point]
            end_point -= de_gene_len_list[i]
        start_point = 0
        for i in range(len(en_node_num_list)):
            en_gene_list[i] = gene[start_point:start_point + en_gene_len_list[i]]
            start_point += en_gene_len_list[i]

        de_node_func_gene_list = [None for _ in range(len(de_node_num_list))]
        en_node_func_gene_list = [None for _ in range(len(en_node_num_list))]
        for i in range(len(de_node_num_list)):
            de_node_func_gene_list[i] = de_gene_list[i][0: de_node_func_gene_len]
        for i in range(len(en_node_num_list)):
            en_node_func_gene_list[i] = en_gene_list[i][0: en_node_func_gene_len]

        de_connect_gene_list = [None for _ in range(len(de_node_num_list))]
        en_connect_gene_list = [None for _ in range(len(en_node_num_list))]
        for i in range(len(de_node_num_list)):
            de_connect_gene_list[i] = de_gene_list[i][
                                      -de_connect_gene_len_list[i]:]
        for i in range(len(en_node_num_list)):
            en_connect_gene_list[i] = en_gene_list[i][
                                      -en_connect_gene_len_list[i]:]

        de_node_func_type_list = [None for _ in range(len(de_node_num_list))]
        for i in np.arange(len(de_node_num_list)):
            index = int(''.join([str(j) for j in de_node_func_gene_list[i]]), 2)
            if index > de_func_type_num - 1:
                index = de_func_type_num - 1
            de_node_func_type_list[i] = de_func_type[index]

        en_node_func_type_list = [None for _ in range(len(en_node_num_list))]
        for i in np.arange(len(en_node_num_list)):
            index = int(''.join([str(j) for j in en_node_func_gene_list[i]]), 2)
            if index > en_func_type_num - 1:
                index = en_func_type_num - 1
            en_node_func_type_list[i] = en_func_type[index]

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up_operations = nn.ModuleList()
        for _ in range(sample_num):
            self.up_operations.append(
                nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=2, stride=2))

        self.init_conv = nn.Conv2d(in_channels=in_ch, out_channels=channel, kernel_size=3, stride=1, padding=1)

        self.encode_operations = nn.ModuleList()
        for i in range(sample_num + 1):
            en_active, en_pre_index, en_out_index = check_active(en_node_num_list[i], en_connect_gene_list[i])
            self.encode_operations.append(
                UnetBlock(channel, en_active, en_pre_index, en_out_index, en_node_func_type_list[i]))

        self.decode_operations = nn.ModuleList()
        for i in range(sample_num):
            de_active, de_pre_index, de_out_index = check_active(de_node_num_list[i], de_connect_gene_list[i])
            self.decode_operations.append(
                UnetBlock(channel, de_active, de_pre_index, de_out_index, de_node_func_type_list[i]))

        self.final_conv = nn.Conv2d(in_channels=channel, out_channels=out_ch, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.size_x = 0
        self.size_y = 0

    def forward(self, x):
        x = self._same_padding(x)
        x = self.init_conv(x)
        encode_outputs = [None for _ in range(len(self.encode_operations))]

        for i, op in enumerate(self.encode_operations):
            if i == 0:
                encode_outputs[i] = op(x)
            else:
                encode_outputs[i] = op(self.maxpool(encode_outputs[i - 1]))

        for i, op in enumerate(self.decode_operations):
            if i == 0:
                out = op(self.up_operations[i](encode_outputs[-1]) + encode_outputs[-(2 + i)])
            else:
                out = op(self.up_operations[i](out) + encode_outputs[-(2 + i)])

        out = self.final_conv(out)
        out = self.sigmoid(out)
        out = out[:, :, 0:self.size_x, 0:self.size_y]

        return out

    def _same_padding(self, input_):
        self.num = 16
        self.size_x = input_.size(2)
        self.size_y = input_.size(3)
        x_padding_num = 0
        y_padding_num = 0
        if self.size_x % self.num != 0:
            x_padding_num = (self.size_x // self.num + 1) * self.num - self.size_x
        if self.size_y % self.num != 0:
            y_padding_num = (self.size_y // self.num + 1) * self.num - self.size_y

        pad_parten = (0, y_padding_num, 0, x_padding_num)
        import torch.nn.functional as F
        output = F.pad(input=input_, pad=pad_parten,
                       mode='constant', value=0)
        return output
