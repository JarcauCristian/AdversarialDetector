import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class AdvDetectorModel(nn.Module):
    def __init__(self):
        super(AdvDetectorModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.res1 = self._make_layer(BasicBlock, 16, 16, num_blocks=5)
        self.res2 = self._make_layer(BasicBlock, 16, 32, num_blocks=5, stride=2)
        self.res3 = self._make_layer(BasicBlock, 32, 64, num_blocks=5, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = nn.Linear(64, 200)
        self.adv_detectors = nn.ModuleList([self._make_adv_detector(i) for i in range(5)])

    def _make_layer(self, block, in_planes, planes, num_blocks, stride=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(in_planes, planes, stride))
            in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_adv_detector(self, index):
        if index == 4:
            layers = [
                nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(192, 2, kernel_size=3, stride=1, padding=1),
                nn.AvgPool2d((1, 1))
            ]
        elif index == 3:
            layers = [
                nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(192, 2, kernel_size=3, stride=1, padding=1),
                nn.AvgPool2d((1, 1))
            ]
        elif index == 2:
            layers = [
                nn.Conv2d(16, 96, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(192, 2, kernel_size=3, stride=1, padding=1),
                nn.AvgPool2d((1, 1))
            ]
        elif index == 1:
            layers = [
                nn.Conv2d(16, 96, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(192, 2, kernel_size=3, stride=1, padding=1),
                nn.AvgPool2d((1, 1))
            ]
        elif index == 0:
            layers = [
                nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(192, 2, kernel_size=3, stride=1, padding=1),
                nn.AvgPool2d((1, 1))
            ]

        return nn.Sequential(*layers)

    def forward(self, x):
        adv_det_results = []

        adv_det_results.append(self.adv_detectors[0](x))

        x = self.conv1(x)
        adv_det_results.append(self.adv_detectors[1](x))
        x = self.res1(x)
        adv_det_results.append(self.adv_detectors[2](x))
        x = self.res2(x)
        adv_det_results.append(self.adv_detectors[3](x))
        x = self.res3(x)
        adv_det_results.append(self.adv_detectors[4](x))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x, adv_det_results

