from .model import *
from .tokenizer import MyTokenizer
from .train_loop import *
import torch
import os
import argparse
import numpy
import torch.nn.init as init
def init_model_weights(model: torch.nn.Module):
    """专门适配你手写类的初始化函数"""
    for name, module in model.named_modules():
        # 判断是不是你手写的 Linear 类
        if isinstance(module, Linear):
            # 获取输入输出维度
            d_in = module.weight.shape[1]
            d_out = module.weight.shape[0]
            std = math.sqrt(2.0 / (d_in + d_out))
            init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)
            
        # 判断是不是你手写的 Embedding 类
        elif isinstance(module, Embedding):
            init.trunc_normal_(module.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
            
        # 判断是不是你手写的 RmsNorm 类
        elif isinstance(module, RmsNorm):
            init.ones_(module.weight)
def one_step(model:torch.nn.Module, optimizer:torch.optim.Optimizer, x, batch_size, context_length, device):
    model.train()
    x_inputs, y_targets = load_data(numpy.array(x), batch_size, context_length, device)
    logits = model(x_inputs)
    loss = my_cross_entropy(logits=logits, target_token_ids=y_targets)
    optimizer.zero_grad()
    loss.backward()
    gradient_clipping(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    # tokenizer hyperparameters
    argparser.add_argument('--vocab_path', type=str, default='tokenizer-data/vocab.pkl')
    argparser.add_argument('--merges_path', type=str, default='tokenizer-data/merges.pkl')
    argparser.add_argument('--special_tokens', type=str, nargs='*', default=['<|endoftext|>'])
    # model hyperparameters
    argparser.add_argument('--context_length', type=int, default=128)
    argparser.add_argument('--d_model', type=int, default=128)
    argparser.add_argument('--num_head', type=int, default=4)
    argparser.add_argument('--num_layer', type=int, default=4)
    # training hyperparameters
    argparser.add_argument('--batch_size', type=int, default=64)
    argparser.add_argument('--num_iterations', type=int, default=1000)
    argparser.add_argument('--learning_rate', type=float, default=1e-3)
    argparser.add_argument('--checkpoint_interval', type=int, default=100)
    argparser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    argparser.add_argument('--resume_from', type=str, default=None)
    args = argparser.parse_args()
    tokenizer = MyTokenizer.from_files(args.vocab_path, args.merges_path, args.special_tokens)
    vocab_size = len(tokenizer.vocab)
    model = TransformerLm(vocab_size=vocab_size, context_length=args.context_length, d_model=args.d_model, num_heads=args.num_head, num_layers=args.num_layer, d_ff=8*args.d_model//3, rope_theta=10000.0, device='cuda:0')
    init_model_weights(model)
    optimizer = MyAdamW(model.parameters(), lr=args.learning_rate)
    if args.resume_from:
        start_iteration = load_checkpoint(args.resume_from, model, optimizer)
    else:
        start_iteration = 0
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open('data/TinyStories-valid.txt', 'r') as f:
        token_ids = tokenizer.encode(f.read())
    for it in range(start_iteration, args.num_iterations):
        for param_group in optimizer.param_groups:
            param_group['lr'] = cos_lr_schedule(lr_max=args.learning_rate, lr_min=args.learning_rate*0.1, warmup_steps=100, cos_steps=900, cur_step=it)
        loss = one_step(model, optimizer, token_ids, args.batch_size, args.context_length, 'cuda:0')
        print(f'iteration {it}, loss={loss}')
        if (it+1) % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_{it+1}.pt')
            save_checkpoint(model, optimizer, it+1, checkpoint_path)