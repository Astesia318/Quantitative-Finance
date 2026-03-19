import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size=32,
        activation=nn.LeakyReLU(),
        out_activation=nn.LeakyReLU(),
    ):
        """Generate a basic MLP

        Args:
            input_size (int)
            output_size (int)
            hidden_size (int, list, optional): Defaults to 32.
            activation (optional): Defaults to nn.LeakyReLU. 传入 None 则不使用激活函数。
            out_activation (optional): Defaults to nn.LeakyReLU. 传入 None 则不使用激活函数。
        """
        super(MLP, self).__init__()

        self.net = nn.Sequential()

        if type(hidden_size) is list:
            num_hidden_layer = len(hidden_size)
            
            # 1. 输入层
            self.net.add_module("input", nn.Linear(input_size, hidden_size[0]))
            if activation is not None:
                self.net.add_module("input_activ", activation)
                
            # 2. 中间隐藏层 (修复了原代码这里漏掉 activation 的 bug)
            for i in range(num_hidden_layer - 1):
                self.net.add_module(
                    f"hidden_{i}", nn.Linear(hidden_size[i], hidden_size[i + 1])
                )
                if activation is not None:
                    self.net.add_module(f"hidden_activ_{i}", activation)
                    
            # 3. 输出层
            self.net.add_module(
                "out", nn.Linear(hidden_size[num_hidden_layer - 1], output_size)
            )
            if out_activation is not None:
                self.net.add_module("out_activ", out_activation)
                
        else:
            # 单隐藏层情况
            self.net.add_module("layer1", nn.Linear(input_size, hidden_size))
            if activation is not None:
                self.net.add_module("activ1", activation)
                
            self.net.add_module("layer2", nn.Linear(hidden_size, output_size))
            if out_activation is not None:
                self.net.add_module("out_activ", out_activation)

    def forward(self, x):
        out = self.net(x)
        return out