"""Matched from-scratch FineWeb screen with a periodically refreshed self-teacher."""
import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F

from signal_screen import ROOT, definitions, mapped_tokens, sha, write

WIDTH, ROWS, STEPS, REFRESH = 16, 256, 24, 4
ARMS = ("hard", "ordinary", "gold", "shuffled")


def get_batch(mem, offset, rows):
    size = WIDTH * rows
    buf = torch.tensor(np.array(mem[offset:offset + size + 1], dtype=np.int64), device="cuda")
    return buf[:-1].reshape(rows, WIDTH), buf[1:].reshape(rows, WIDTH)


def logits(net, x, rows=None, positions=None):
    h = net.norm1(net.embed(x))
    for block in net.blocks:
        h = block(h)
    if positions is not None:
        h = h[rows, positions]
    z = net.proj(net.norm2(h)).float()
    return 15*z*(z.square()+225).rsqrt()


@torch.no_grad()
def targets(teacher, x, y, arm, shuffled_y):
    if arm == "ordinary":
        return logits(teacher, x).softmax(-1)
    rows, width = x.shape
    end = torch.arange(width, device=x.device).repeat(rows)
    n = end[:, None] + 1
    col = torch.arange(2*width+2, device=x.device)[None, :]
    idx = torch.where(col < n, col, col-n-2).clamp(0, width-1)
    prompt = x.repeat_interleave(width, 0).gather(1, idx)
    hint = y if arm == "gold" else shuffled_y
    prompt = torch.where(col == n, hint.reshape(-1, 1), prompt)
    prompt = torch.where(col == n+1, 50256, prompt)
    z = logits(teacher, prompt, torch.arange(rows*width, device=x.device), 2*end+3)
    return z.reshape(rows,width,-1).softmax(-1)


def initialize(net, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        for n,p in net.named_parameters():
            if n.endswith("weight") and "proj" in n:
                p.zero_()
            elif n.endswith("weight") and "embed" in n:
                p.normal_()
            elif n.endswith("weight"):
                p.normal_(std=0.33**0.5 / p.size(-1)**0.5)
            elif n.endswith("bias"):
                p.zero_()
            elif n.endswith("gains"):
                p.normal_(mean=1,std=0)
            else:
                raise ValueError(n)
    digest=hashlib.sha256()
    for n,p in net.state_dict().items():
        digest.update(n.encode())
        digest.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    dist.init_process_group("nccl", device_id=torch.device("cuda",0))
    ns=definitions()
    net=ns["GPT"](50304,12,768).cuda()
    teacher=ns["GPT"](50304,12,768).cuda().eval().requires_grad_(False)
    train=mapped_tokens(ROOT/"data/fineweb10B/fineweb_train_000001.bin")
    dev=mapped_tokens(ROOT/"data/fineweb10B/fineweb_val_000000.bin")
    dx,dy=get_batch(dev,12_000_000,4096)
    @torch.no_grad()
    def evaluate():
        return sum(net(x,y).item() for x,y in zip(dx.split(128),dy.split(128)))/dy.numel()

    config=dict(seeds=[17,42],arms=ARMS,steps=STEPS,refresh_updates=REFRESH,context=WIDTH,
                train_tokens_per_update=ROWS*WIDTH,train_offset=30_000_000,dev_offset=12_000_000,
                dev_tokens=dy.numel(),source_sha256=sha(Path(__file__)),
                objective="0.5 hard CE + 0.5 dense detached teacher CE, temperature 1",
                gate="gold below hard, ordinary and shuffled at step 24 on both seeds",
                timing="Includes periodic teacher copies, teacher forwards, student training and optimizer; excludes evaluation and initialization; shared host.")
    write(out/"plan.json",config)
    for seed in config["seeds"]:
        for arm in ARMS:
            init=initialize(net,seed)
            adam=torch.optim.AdamW([dict(params=[net.embed.weight],lr=.7),dict(params=[net.proj.weight],lr=.004),
                 dict(params=[p for p in net.parameters() if p.ndim<2],lr=.015)],betas=(.8,.95),eps=1e-10,weight_decay=.001,fused=True)
            mu=ns["Muon"]([p for p in net.blocks.parameters() if p.ndim>=2],lr=.025,weight_decay=.05)
            curve=[dict(step=0,loss=evaluate(),train_seconds=0.)]
            elapsed=0.
            for step in range(STEPS):
                torch.cuda.synchronize()
                t=time.monotonic()
                if arm!="hard" and step%REFRESH==0:
                    teacher.load_state_dict(net.state_dict())
                x,y=get_batch(train,30_000_000+step*ROWS*WIDTH,ROWS)
                shuffled=y.roll(1,0)
                for b in range(0,ROWS,16):
                    xx,yy=x[b:b+16],y[b:b+16]
                    z=logits(net,xx)
                    hard=F.cross_entropy(z.flatten(0,1),yy.flatten(),reduction="sum")
                    if arm=="hard":
                        loss=hard
                    else:
                        q=targets(teacher,xx,yy,arm,shuffled[b:b+16])
                        loss=.5*hard-.5*(q*z.log_softmax(-1)).sum()
                    (loss*(524288/(ROWS*WIDTH))).backward()
                adam.step()
                mu.step()
                net.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                elapsed+=time.monotonic()-t
                if (step+1)%8==0:
                    point=dict(step=step+1,loss=evaluate(),train_seconds=elapsed)
                    assert np.isfinite(point["loss"])
                    curve.append(point)
                    write(out/f"{seed}-{arm}.json",dict(seed=seed,arm=arm,init_sha256=init,curve=curve))
                    print(seed,arm,point,flush=True)
            del adam,mu
    dist.destroy_process_group()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    main(parser.parse_args().output)
