import torch
import json
from cs336_basics.bpe import Tokenizer
from cs336_basics.transformer import softmax,transformer_lm
from pathlib import Path
from typing import Any
from cs336_basics.tool import resolve_device

@torch.inference_mode()
def decoder(
        model,
        tokenizer,
        prompt: str,
        max_tokens: int,
        seq_len: int,
        temperature: float = 1.0,
        top_p: float = 0.9,
        device: torch.device = None,
) -> str:
    if prompt == "":
        raise ValueError("prompt 不能为空")
    if not 0 < top_p <= 1:
        raise ValueError("top_p 必须位于 (0, 1]")
    eos_id = tokenizer.reversed_vocab[b"<|endoftext|>"]
    model.eval()
    tokens_id = tokenizer.encode(prompt)
    x = torch.tensor(tokens_id, dtype = torch.long,device = device)
    len_new_tokens = 0
    while True:
        if len_new_tokens >= max_tokens or x[-1] == eos_id:
            break
        if len(x) > seq_len:
            context = x[-seq_len:]
        else:
            context = x
        logits = model(context)[-1,:]
        if temperature == 0:
            next_token_id = torch.argmax(logits).item()
        elif temperature < 0:
            raise ValueError('temperature 不能为负数')
        else:
            prob = softmax(logits / temperature, dim = -1)
            sorted_prob, sorted_id = torch.sort(prob, descending = True)
            cumulative_prob = torch.cumsum(sorted_prob, dim = -1)
            keep = (cumulative_prob - sorted_prob) < top_p  ##核心！！！
            final_prob = sorted_prob * keep
            final_prob /= final_prob.sum()
            index = torch.multinomial(final_prob,num_samples = 1)
            next_token_id = sorted_id[index].item()
        next_token = torch.tensor([next_token_id],dtype=x.dtype,device = x.device)
        x = torch.cat([x,next_token], dim = 0)
        len_new_tokens += 1
    return tokenizer.decode(x[len(tokens_id):].tolist())
        
def load_model(
        checkpoint_path: str, #储存着参数
        config_path: str,
        device: str = 'auto'
):
    device = resolve_device(device)
    with open(config_path, 'r',encoding='utf-8') as f:
        config = json.load(f)
    model = transformer_lm(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        num_layers=config["num_layers"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
        device=device,
    )
    checkpoint = torch.load(checkpoint_path,map_location=device,weights_only=True)
    model.load_state_dict(checkpoint['model'], strict = True)
    return model, config
def main():
    prompt = "Oh shit!"
    tokens_path =  "tokens_id/tinystories_train_tokenizer.json"
    tokenizer = Tokenizer.from_files(tokens_path)
    model,config = load_model('runs/experiment01/ckpt_final.pt',
                       'runs/experiment01/config.json')
    print(decoder(model,tokenizer,prompt,100,config['context_length'],0.9,0.9,device = next(model.parameters()).device))
if __name__ == "__main__":
    main()

