from .model import *
from .tokenizer import MyTokenizer
from .train_loop import load_data, save_checkpoint, load_checkpoint
import torch
import os
def infer(model:torch.nn.Module, tokenizer:MyTokenizer, prompt:str, device:str, max_length:int=128, temperature:float=1.0, context_length:int=128):
    model.eval()
    prompt_ids = tokenizer.encode(prompt)
    output_ids = prompt_ids.copy()
    for _ in range(max_length):
        x_input = torch.tensor(output_ids[-context_length:], dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model(x_input)
        next_token_logits = logits[0, -1] / temperature
        sm = my_softmax(next_token_logits, dim=-1)
        output_ids.append(torch.argmax(sm).item())
    return tokenizer.decode(output_ids)
    

if __name__ == '__main__':
    tokenizer = MyTokenizer.from_files('tokenizer-data/vocab.pkl', 'tokenizer-data/merges.pkl', ['<|endoftext|>'])
    vocab_size = len(tokenizer.vocab)
    model = TransformerLm(vocab_size=vocab_size, context_length=128, d_model=128, num_heads=4, num_layers=4, d_ff=8*128//3, rope_theta=10000.0, device='cuda:0')
    checkpoint_path = os.path.join('checkpoints', 'checkpoint_500.pt')
    load_checkpoint(checkpoint_path, model, None)
    prompt = "Once upon a time"
    print(infer(model, tokenizer, prompt, device='cuda:0', max_length=128, temperature=1.0, context_length=128))