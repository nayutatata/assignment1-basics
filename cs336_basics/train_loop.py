from .model import *
from .tokenizer import MyTokenizer
import torch
import math
def load_data(x, batch_size, context_length, device):
# 1. 随机生成 batch_size 个起始索引
    # 注意最大索引边界：必须给 Y 留出足够的空间，所以是 len(x) - context_length - 1
    # 假设 len(x) = 100, context_length = 10，能取的最大起点是 100-10-1 = 89
    ix = torch.randint(0, len(x) - context_length, (batch_size,))
    
    # 2. 根据随机生成的索引，去原始纸带 x 上切片
    # 利用列表推导式，把每一个切出的 numpy 片段转成 tensor，然后堆叠起来
    x_inputs = torch.stack([torch.from_numpy(x[i : i + context_length]) for i in ix])
    
    # 目标 Y 就是 X 往后错开一位
    y_targets = torch.stack([torch.from_numpy(x[i + 1 : i + 1 + context_length]) for i in ix])
    
    x_inputs = x_inputs.to(dtype=torch.long, device=device)
    y_targets = y_targets.to(dtype=torch.long, device=device)
    
    return x_inputs, y_targets

def save_checkpoint(model:torch.nn.Module, optimizer:torch.optim.Optimizer, iteration, out):
    save_obj = {}
    save_obj['model'] = model.state_dict()
    save_obj['optimizer'] = optimizer.state_dict()
    save_obj['it'] = iteration
    torch.save(save_obj, out)

def load_checkpoint(src, model:torch.nn.Module, optimizer:torch.optim.Optimizer):
    save_obj = torch.load(src)
    model.load_state_dict(save_obj['model'])
    if optimizer:
        optimizer.load_state_dict(save_obj['optimizer'])
    return save_obj['it']