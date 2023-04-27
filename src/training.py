import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from pytz import timezone

"""
-----------------------------------------------------------------------------------
                                   Data loading
-----------------------------------------------------------------------------------
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print(device)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('data/tinyshakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))

d_input = len(chars) + 1 # let 0 be padding

stoi = lambda c: chars.index(c) + 1
itos = lambda n: "" if n == 0 else chars[n-1]
encode = lambda s: torch.tensor([stoi(c) for c in s], dtype=torch.long)
decode = lambda m: ''.join([itos(i) for i in m])

code = encode('abc')
#print(code)
#print(decode(code))

data = encode(text)
n_split = int(0.9 * len(data))
train_data = data[:n_split]
test_data = data[n_split:]

def get_batch(mode, seq_len, batch_size=1):
    source = train_data if mode == 'train' else test_data
    starts = torch.randint(len(source) - seq_len - 1, (batch_size, ))
    x = torch.stack([source[s:s+seq_len] for s in starts])
    y = torch.stack([source[s+1:s+1+seq_len] for s in starts])
    x, y = x.to(device), y.to(device)
    return x, y

eval_iters = 20 #200
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for mode in ['train', 'eval']:
        losses = torch.zeros(eval_iters)
        accs = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(mode, max_seq_len)
            predictions, loss = model(X, Y)

            probs = F.softmax(predictions, dim=-1)
            P = torch.multinomial(probs, num_samples=1)
            Y = torch.flatten(Y)
            P = torch.flatten(P)[:len(Y)]
            acc = torch.sum(Y == P).float() / len(Y)

            accs[k] = acc
            losses[k] = loss.item()
        out[mode] = (losses.mean(), accs.mean())
    model.train()
    return out

"""
-----------------------------------------------------------------------------------
                                Import and load model
-----------------------------------------------------------------------------------
"""

import gpt

max_seq_len = 256

model = gpt.GPT(d_model=384, d_input=d_input, max_seq_len=max_seq_len, N=6, num_heads=6, dropout=0.2, pos_embedding_encode=True)
try:
    NAME = "shakeGPTv3"
    PATH = f"models/{NAME}.pth"
    CHECKPOINT = f"models/checkpoints-{NAME}.pth"
    model.load_state_dict(torch.load(PATH))
    checkpoints = torch.load(CHECKPOINT)
    print("loaded model")
except:
    print("new model")
    checkpoints = []

step_offset = 1
timeoffset = datetime.strptime("0:0:1", '%H:%M:%S') - datetime.strptime("0:0:0", '%H:%M:%S')

if len(checkpoints) > 0:
    step_offset = checkpoints[-1]['step'] + 1
    timeoffset = datetime.strptime(checkpoints[-1]['T+'], '%H:%M:%S') - datetime.strptime("0:0:0", '%H:%M:%S')

torch.cuda.empty_cache()
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.00003)

"""
-----------------------------------------------------------------------------------
                                Training loop
-----------------------------------------------------------------------------------
"""

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(f"runs/{NAME}")
#writer.add_graph(model, get_batch('train', max_seq_len, 1))

tz = timezone('EST')
t_start = datetime.now(tz)
print(f"Start time: {t_start.strftime('%Y-%m-%d %H:%M:%S')}")

prev_delta = None
def trackTime(remain_gap=None):
    def timeStr(d):
        h = d.seconds//3600
        m = (d.seconds//60) % 60
        s = d.seconds - (h*3600 + m*60)
        if s < 10:
            s = f"0{s}"
        return f"{h}:{m}:{s}"
    delta = datetime.now(tz) - t_start + timeoffset
    delta_str = timeStr(delta)

    projected_remain_time = None
    global prev_delta
    if prev_delta != None and remain_gap != None:
        time_past = delta - prev_delta
        projected_delta = remain_gap * time_past
        projected_remain_time = timeStr(projected_delta)

    prev_delta = delta
    return delta_str, projected_remain_time

max_iters = 2500
for step in range(0, max_iters):
    
    x, y = get_batch('train', max_seq_len, 64) # batch size

    z, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % (max_iters // 300) == 0 or step == max_iters-1:
        res = estimate_loss()

        gap = max_iters // 300
        remainGap = (max_iters - step) / gap
        delta, proj_t = trackTime(remainGap)
        print(f"T+ {delta} - step {step+step_offset}/{max_iters+step_offset-1}: train loss {res['train'][0]:.4f}, val loss {res['eval'][0]:.4f}; train acc  {res['train'][1]:.4f}, val acc {res['eval'][1]:.4f}; t_remain {proj_t}")

        checkpoint = {
            "T+": delta,
            "step": step+step_offset,
            "train_loss": res['train'][0],
            "val_loss": res['eval'][0],
            "train_acc": res['train'][1],
            "val_acc": res['eval'][1]
        }
        
        writer.add_scalar('training loss', res['train'][0], step+step_offset)
        writer.add_scalar('validation loss', res['eval'][0], step+step_offset)
        writer.add_scalar('training acc', res['train'][1], step+step_offset)
        writer.add_scalar('validation acc', res['eval'][1], step+step_offset)

        # keep saving the model
        for i in range(3):
            try:
                torch.save(model.state_dict(), PATH)
                break
            except:
                print("try saving model again", i)
        checkpoints.append(checkpoint)
        torch.save(checkpoints, CHECKPOINT)

"""
-----------------------------------------------------------------------------------
                                Save model
-----------------------------------------------------------------------------------
"""

torch.save(model.state_dict(), PATH)

"""
-----------------------------------------------------------------------------------
                                Test output
-----------------------------------------------------------------------------------
"""

context = encode("ANTIGONUS:").unsqueeze(0).to(device) # add batch to 1

result = model.generate(context, new_seq_len=300)[0]
print(decode(result))

