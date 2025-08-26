import os, sys, random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from pbdl.torch.loader import Dataloader
# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

BATCH_SIZE = 10
loader_train, loader_val = Dataloader.new_split(
    [320, 80],
    "airfoils",
    batch_size=BATCH_SIZE, normalize_data=None,
)
# 获取一个批次的数据
inputs, targets = next(iter(loader_train))

print("=== 数据维度信息 ===")
print(f"输入数据形状: {inputs.shape}")
print(f"目标数据形状: {targets.shape}")
print(f"输入数据类型: {inputs.dtype}")
print(f"目标数据类型: {targets.dtype}")

print("\n=== 单个样本分析 ===")
sample_input = inputs[0]  # 第一个样本的输入
sample_target = targets[0]  # 第一个样本的目标

print(f"输入通道数: {sample_input.shape[0]} (应该是3个通道)")
print(f"输入分辨率: {sample_input.shape[1]}x{sample_input.shape[2]}")
print(f"输出通道数: {sample_target.shape[0]} (应该是3个通道)")

# 查看每个通道的统计信息
for i in range(3):
    print(f"\n输入通道{i} - min: {sample_input[i].min():.4f}, max: {sample_input[i].max():.4f}, mean: {sample_input[i].mean():.4f}")
    print(f"输出通道{i} - min: {sample_target[i].min():.4f}, max: {sample_target[i].max():.4f}, mean: {sample_target[i].mean():.4f}")

print(1)

def plot(a1, a2, mask=None, stats=False, bottom="NN Output", top="Reference", title=None):
    c = []
    if mask is not None: mask = np.asarray(mask)
    for i in range(3):
        a2i = np.asarray(a2[i])
        if mask is not None: a2i = a2i - mask*a2i  # 可选地遮盖内部区域
        b = np.flipud(np.concatenate((a2i, a1[i]), axis=1).transpose())
        min, mean, max = np.min(b), np.mean(b), np.max(b)
        if stats:
            print("Stats %d: " % i + format([min, mean, max]))
        b -= min
        b /= max - min
        c.append(b)
    fig, axes = plt.subplots(1, 1, figsize=(16, 5))
    axes.set_xticks([]) ; axes.set_yticks([])
    im = axes.imshow(np.concatenate(c, axis=1), origin="upper", cmap="magma")
    fig.colorbar(im, ax=axes)
    axes.set_xlabel("p, ux, uy")
    axes.set_ylabel("%s %s" % (bottom, top))
    if title is not None: plt.title(title)
    plt.show()

inputs, targets = next(iter(loader_train))
plot(inputs[0], targets[0], stats=False, bottom="Target Output", top="Inputs", title="Training sample")


def blockUNet(in_c, out_c, name, size=4, pad=1, transposed=False, bn=True, activation=True, relu=False, dropout=0.):
    block = nn.Sequential()
    if not transposed:
        block.add_module(
            "%s_conv" % name,
            nn.Conv2d(in_c, out_c, kernel_size=size, stride=2, padding=pad, bias=True)
        )
    else:
        block.add_module(
            "%s_upsam" % name, nn.Upsample(scale_factor=2, mode="bilinear")
        )
        # 为上采样（即解码器部分）减少一个核大小
        block.add_module(
            "%s_tconv" % name,
            nn.Conv2d(in_c, out_c, kernel_size=(size - 1), stride=1, padding=pad, bias=True)
        )
    if bn:
        block.add_module("%s_bn" % name, nn.BatchNorm2d(out_c))
    if dropout > 0.0:
        block.add_module("%s_dropout" % name, nn.Dropout2d(dropout, inplace=True))
    if activation:
        if relu:
            block.add_module("%s_relu" % name, nn.ReLU(inplace=True))
        else:
            block.add_module("%s_leakyrelu" % name, nn.LeakyReLU(0.2, inplace=True))
    return block


class DfpNet(nn.Module):
    def __init__(self, channelExponent=6, dropout=0.0):
        super(DfpNet, self).__init__()
        channels = int(2 ** channelExponent + 0.5)
        self.layer1 = blockUNet(3, channels * 1, "enc_layer1", transposed=False, bn=False, relu=False, dropout=dropout)
        self.layer2 = blockUNet(channels, channels * 2, "enc_layer2", transposed=False, bn=True, relu=False,
                                dropout=dropout)
        self.layer3 = blockUNet(channels * 2, channels * 2, "enc_layer3", transposed=False, bn=True, relu=False,
                                dropout=dropout)
        self.layer4 = blockUNet(channels * 2, channels * 4, "enc_layer4", transposed=False, bn=True, relu=False,
                                dropout=dropout)
        self.layer5 = blockUNet(channels * 4, channels * 8, "enc_layer5", transposed=False, bn=True, relu=False,
                                dropout=dropout)
        self.layer6 = blockUNet(channels * 8, channels * 8, "enc_layer6", transposed=False, bn=True, relu=False,
                                dropout=dropout)
        self.layer7 = blockUNet(channels * 8, channels * 8, "enc_layer7", transposed=False, bn=True, relu=False,
                                dropout=dropout)

        # 注意，内核大小在解码器部分内部减少一个
        self.dlayer7 = blockUNet(channels * 8, channels * 8, "dec_layer7", transposed=True, bn=True, relu=True,
                                 dropout=dropout)
        self.dlayer6 = blockUNet(channels * 16, channels * 8, "dec_layer6", transposed=True, bn=True, relu=True,
                                 dropout=dropout)
        self.dlayer5 = blockUNet(channels * 16, channels * 4, "dec_layer5", transposed=True, bn=True, relu=True,
                                 dropout=dropout)
        self.dlayer4 = blockUNet(channels * 8, channels * 2, "dec_layer4", transposed=True, bn=True, relu=True,
                                 dropout=dropout)
        self.dlayer3 = blockUNet(channels * 4, channels * 2, "dec_layer3", transposed=True, bn=True, relu=True,
                                 dropout=dropout)
        self.dlayer2 = blockUNet(channels * 4, channels, "dec_layer2", transposed=True, bn=True, relu=True,
                                 dropout=dropout)
        self.dlayer1 = blockUNet(channels * 2, 3, "dec_layer1", transposed=True, bn=False, activation=False,
                                 dropout=dropout)

    def forward(self, input):
        # 注意，这个Unet堆栈当然可以用循环来分配...
        out1 = self.layer1(input)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        # ... 瓶颈 ...
        dout6 = self.dlayer7(out7)
        dout6_out6 = torch.cat([dout6, out6], 1)
        dout6 = self.dlayer6(dout6_out6)
        dout6_out5 = torch.cat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2 = torch.cat([dout3, out2], 1)
        dout2 = self.dlayer2(dout3_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)
        return dout1

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# 通道指数来控制网络大小
EXPO = 4
#torch.set_default_device("cuda:0")
torch.set_default_device("cpu")
device = torch.get_default_device()
net = DfpNet(channelExponent=EXPO)
net.apply(weights_init)

# 需要关注的关键参数：我们有多少参数？
nn_parameters = filter(lambda p: p.requires_grad, net.parameters())
print("可训练参数：{} -> 关键！始终保持关注...".format(sum([np.prod(p.size()) for p in nn_parameters])))

LR = 0.0002  # 学习率
loss = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=LR, betas=(0.5, 0.999), weight_decay=0.)

EPOCHS = 200  # 训练轮数
loss_hist = []
loss_hist_val = []

if os.path.isfile("dfpnet"):  # NT_DEBUG
    print("发现现有网络，加载并跳过训练")
    net.load_state_dict(torch.load("dfpnet"))
else:
    print("从头开始训练...")
    pbar = tqdm(initial=0, total=EPOCHS, ncols=96)
    for epoch in range(EPOCHS):
        # 训练
        net.train()
        loss_acc = 0
        for i, (inputs, targets) in enumerate(loader_train):
            inputs = inputs.float()
            targets = targets.float()
            net.zero_grad()
            outputs = net(inputs)
            lossL1 = loss(outputs, targets)
            lossL1.backward()
            optimizer.step()
            loss_acc += lossL1.item()
        loss_hist.append(loss_acc / len(loader_train))

        # 评估验证样本
        net.eval()
        loss_acc_v = 0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(loader_val):
                inputs = inputs.float()
                targets = targets.float()
                outputs = net(inputs)
                loss_acc_v += loss(outputs, targets).item()
        loss_hist_val.append(loss_acc_v / len(loader_val))

        pbar.set_description("训练损失：{:7.5f}，验证损失：{:7.5f}".format(loss_hist[-1], loss_hist_val[-1]))
        pbar.update(1)

    torch.save(net.state_dict(), "dfpnet")
    print("训练完成，保存网络权重")

loss_hist = np.asarray(loss_hist)
loss_hist_val = np.asarray(loss_hist_val)

plt.plot(np.arange(loss_hist.shape[0]), loss_hist, "b", label="训练损失")
plt.plot(np.arange(loss_hist_val.shape[0]), loss_hist_val, "g", label="验证损失")
plt.legend()
plt.show()

net.eval()
inputs, targets = next(iter(loader_val))
inputs = inputs.float()
targets = targets.float()
outputs = net(inputs)
outputs = outputs.data.cpu().numpy()
inputs = inputs.cpu()
targets = targets.cpu()
plot(targets[0], outputs[0], mask=inputs[0][2], title="验证样本")

loader_test = Dataloader("airfoils-test", batch_size=1, normalize_data=None, shuffle=False)

loss = nn.L1Loss()
net.eval()
L1t_accum = 0.
for i, testdata in enumerate(loader_test, 0):
    inputs_curr, targets_curr = testdata
    inputs = inputs_curr.float()
    targets = targets_curr.float()
    outputs = net(inputs)

    outputs_curr = outputs.data.cpu().numpy()
    inputs_curr = inputs_curr.cpu()
    targets_curr = targets_curr.cpu()

    L1t_accum += loss(outputs, targets).item()
    if i<3: plot(targets_curr[0], outputs_curr[0], mask=inputs_curr[0][2], title="测试样本 %d" % i)

print("\n平均相对测试误差：{}".format(L1t_accum/len(loader_test)))



