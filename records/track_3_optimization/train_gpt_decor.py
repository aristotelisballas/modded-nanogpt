"""
train_gpt_decor.py

PerturbationDECOR applied to GPT language model training.

First-order realization of the DECOR objective: instead of explicit Hessian-trace
penalties (which require create_graph / second-order backward), structured parameter
noise eps is injected before the gradient step.  By the leading-order identity

    E_eps[ grad L(theta+eps) ] ≈ grad L(theta)
                                 + (rho_c^2/2) grad tr(H)
                                 + (rho_a^2/2) grad tr(F^{-1} Sigma_g)

a noise covariance  Sigma_eps = rho_c^2 I + rho_a^2 F_alpha^{-1} Sigma_g F_alpha^{-1}
implicitly regularizes both curvature and gradient alignment.
No create_graph; ERM-level memory overhead.

Pseudo-domains are formed by splitting each GPU's local mini-batch into DECOR_K
equal chunks.  The Gram matrix is computed locally and all-reduced (approximate for
world_size > 1).  The alignment noise vector is made globally consistent via a
subsequent all-reduce.  The curvature noise uses a step-seeded generator so all
ranks draw the same random vector without communication.
"""

import os
import sys
with open(sys.argv[0]) as f:
    code = f.read()
import uuid
import time
import itertools
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist


########################################
#        DECOR Hyperparameters         #
########################################

DECOR_K          = 4      # pseudo-domain count; must divide local batch size
DECOR_RHO_C      = 0.05   # curvature noise scale (dimension-free, tune like SAM rho)
DECOR_RHO_A      = 0.05   # alignment noise scale
DECOR_FISHER_EPS = 0.01   # relative Tikhonov shift for F_alpha
DECOR_ANTITHETIC = True   # average g(theta+eps) and g(theta-eps) for lower variance

# False  → DECOR gradient feeds into Muon + AdamW  (Newton-Schulz on top)
# True   → DECOR gradient feeds into AdamW only    (no orthogonalization)
DECOR_STANDALONE       = False
STANDALONE_BLOCKS_LR   = 0.002   # AdamW lr for 2-D block params in standalone mode


########################################
#              Dataloader              #
########################################

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len=1024):
    pattern = Path(filename_pattern)
    files = sorted(pattern.parent.glob(pattern.name))
    assert batch_size % dist.get_world_size() == 0
    local_batch_size = batch_size // dist.get_world_size()
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + dist.get_rank() * local_batch_size:][:local_batch_size + 1]
        inputs  = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)


########################################
#             Architecture             #
########################################

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))

class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim//4)]))

    def forward(self, x_BTHD: Tensor):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim=128):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim  = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), scale=0.12, is_causal=True).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc   = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.relu().square()
        return self.proj(x)

class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn  = CausalSelfAttention(dim)
        self.mlp   = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim) for _ in range(num_layers)])
        self.proj   = Linear(model_dim, vocab_size)
        self.norm1  = RMSNorm(model_dim)
        self.norm2  = RMSNorm(model_dim)

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")


########################################
#              Optimizer               #
########################################

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.compile
def muon_update(grad, momentum, mu=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum"], mu=group["mu"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])


########################################
#                Setup                 #
########################################

device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
assert 8 % dist.get_world_size() == 0

if dist.get_rank() == 0:
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{uuid.uuid4()}.txt"
    print(logfile)
def print0(s, console=False, log=True):
    if dist.get_rank() == 0:
        if console:
            print(s)
        if log:
            with open(logfile, "a") as f:
                print(s, file=f)

print0(code)
print0("="*100)
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
       + f" on {torch.cuda.get_device_name(device)} with world_size {dist.get_world_size()}")
print0(f"PerturbationDECOR: K={DECOR_K} rho_c={DECOR_RHO_C} rho_a={DECOR_RHO_A} "
       f"fisher_eps={DECOR_FISHER_EPS} antithetic={DECOR_ANTITHETIC} "
       f"standalone={DECOR_STANDALONE}")
print0("="*100)

val_tokens = 20 * 524288
batch_size = 8 * 64 * 1024
mbs = 64
val_inputs, val_targets = next(distributed_data_generator("/var/local/storage/aballas/fineweb10B/fineweb_val_*.bin", val_tokens))

model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()
model.compile(dynamic=False)

num_trials = int(sys.argv[-1]) if len(sys.argv) > 1 else 1

for _ in range(num_trials):

    ########################################
    #       Init & Optim Hyperparams       #
    ########################################

    train_steps = 3350

    for name, p in model.named_parameters():
        w = p.data
        if name.endswith("weight"):
            if "proj" in name:
                w.zero_()
            elif "embed" in name:
                w.normal_()
            else:
                w.normal_(std=0.33**0.5 / w.size(-1)**0.5)
        elif name.endswith("bias"):
            w.zero_()
        elif name.endswith("gains"):
            w.normal_(mean=1, std=0)
        else:
            raise Exception(f"Uninitialized parameter: {name}")

    block_2d = [p for p in model.blocks.parameters() if p.ndim >= 2]
    if DECOR_STANDALONE:
        optimizer1 = AdamW([dict(params=[model.embed.weight], lr=0.3),
                            dict(params=[model.proj.weight], lr=1/320),
                            dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01),
                            dict(params=block_2d, lr=STANDALONE_BLOCKS_LR,
                                 betas=(0.9, 0.95), weight_decay=0.1)],
                           betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=True)
        optimizers = [optimizer1]
    else:
        optimizer1 = AdamW([dict(params=[model.embed.weight], lr=0.3),
                            dict(params=[model.proj.weight], lr=1/320),
                            dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01)],
                           betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=True)
        optimizer2 = Muon(block_2d, lr=0.035, weight_decay=0.025)
        optimizers = [optimizer1, optimizer2]
    assert set(p for opt in optimizers for group in opt.param_groups
               for p in group["params"]) == set(model.parameters())
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    def set_hparams(step, cooldown_frac=0.7):
        progress = step / train_steps
        assert 0 <= progress < 1
        eta = 1.0 if progress < 1 - cooldown_frac else (1 - progress) / cooldown_frac
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * eta


    ########################################
    #        Training and Validation       #
    ########################################

    train_loader = distributed_data_generator("/var/local/storage/aballas/fineweb10B/fineweb_train_*.bin", batch_size)
    for p in model.parameters():
        dist.broadcast(p.detach(), 0)

    training_time = 0
    last_val_step = 0
    dist.barrier()
    t0 = time.perf_counter()

    for step in range(train_steps + 1):

        # --------------- VALIDATION ----------------
        val_step_freq = 125 if step / train_steps < 0.9 else 25
        if step == train_steps or step % val_step_freq == 0:
            dist.barrier()
            time_since_last_val = time.perf_counter() - t0
            step_avg = time_since_last_val / (step - last_val_step) if step > 0 else float("nan")
            last_val_step = step
            training_time += time_since_last_val
            model.eval()
            val_loss = 0
            with torch.no_grad():
                assert len(val_inputs) % mbs == 0
                for i in range(len(val_inputs) // mbs):
                    val_loss += model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            val_loss /= val_tokens
            print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} train_time:{training_time:.3f}s"
                   + f" step_avg:{1000*step_avg:.2f}ms", console=True)
            model.train()
            dist.barrier()
            t0 = time.perf_counter()

        if step == train_steps:
            break

        # --------------- DECOR TRAINING STEP ----------------
        inputs, targets = next(train_loader)
        K = DECOR_K
        assert len(inputs) % K == 0, (
            f"local batch ({len(inputs)} seqs) must be divisible by DECOR_K={K}"
        )
        chunk = len(inputs) // K
        inputs_k  = [inputs[k*chunk:(k+1)*chunk]  for k in range(K)]
        targets_k = [targets[k*chunk:(k+1)*chunk] for k in range(K)]

        params = [p for p in model.parameters() if p.requires_grad]

        # (1) Per-domain gradients at theta — no create_graph
        # Each g_k is a per-token mean gradient (normalized by domain token count).
        G_cols = []
        for k in range(K):
            model.zero_grad(set_to_none=True)
            n_tok = inputs_k[k].numel()
            (model(inputs_k[k], targets_k[k]) / n_tok).backward()
            gk = torch.cat([
                p.grad.float() if p.grad is not None
                else torch.zeros(p.numel(), dtype=torch.float32, device=p.device)
                for p in params
            ]).detach().clone()
            G_cols.append(gk)

        # (2) Gram matrix — computed locally, then all-reduced.
        # For world_size > 1 this is an approximation of the global Gram matrix
        # (cross-rank terms are absent), but it is consistent across ranks after
        # the all-reduce and is sufficient for the noise direction.
        d = G_cols[0].numel()
        M = torch.stack([
            torch.stack([torch.dot(G_cols[j], G_cols[l]) for l in range(K)])
            for j in range(K)
        ])
        dist.all_reduce(M, op=dist.ReduceOp.SUM)

        M_scale = torch.diagonal(M).mean()
        # Tikhonov shift; the 1e-30 floor guards against an all-zero gradient.
        alpha  = DECOR_FISHER_EPS * M_scale + 1e-30
        eye_K  = torch.eye(K, device=M.device, dtype=M.dtype)
        Cmat   = eye_K - torch.ones(K, K, device=M.device, dtype=M.dtype) / K

        # (3) Build noise — curvature part is rank-independent (seeded by step);
        # alignment part is assembled locally then all-reduced for global consistency.
        gen = torch.Generator(device=inputs.device)
        gen.manual_seed(step)   # same seed on every rank → identical eps_curv and z

        eps_curv  = (DECOR_RHO_C / d**0.5) * torch.randn(
            d, device=inputs.device, dtype=torch.float32, generator=gen)
        z         = torch.randn(K, device=inputs.device, dtype=torch.float32, generator=gen)

        A     = alpha * K * eye_K + M
        B     = (eye_K - torch.linalg.solve(A, M)) @ Cmat / alpha  # (K, K)
        coeff = (DECOR_RHO_A / K**0.5) * (B @ z)                   # (K,)

        # eps_align_global = sum_k coeff_k * g_k_global
        #                  = all_reduce_SUM( sum_k coeff_k * g_k_local )
        # Division by world_size converts from SUM to mean so that the noise
        # scale is independent of world_size (consistent with per-token g_k).
        eps_align = sum(coeff[k] * G_cols[k] for k in range(K))
        dist.all_reduce(eps_align, op=dist.ReduceOp.SUM)
        eps_align.div_(dist.get_world_size())

        eps = eps_curv + eps_align

        # (4) Perturb parameters and evaluate ERM gradient(s) — no create_graph.
        @torch.no_grad()
        def perturb(sign: float):
            i = 0
            for p in params:
                n = p.numel()
                p.data.add_(eps[i:i + n].view_as(p).to(p.dtype), alpha=sign)
                i += n

        def erm_flat_grad():
            """ERM gradient at current (perturbed) params; all-reduced across ranks."""
            model.zero_grad(set_to_none=True)
            assert len(inputs) % mbs == 0
            for i in range(len(inputs) // mbs):
                model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs]).backward()
            g = torch.cat([
                p.grad.float() if p.grad is not None
                else torch.zeros(p.numel(), dtype=torch.float32, device=p.device)
                for p in params
            ])
            dist.all_reduce(g, op=dist.ReduceOp.SUM)
            return g

        perturb(+1.0)                       # theta → theta + eps
        g_plus = erm_flat_grad()
        if DECOR_ANTITHETIC:
            perturb(-2.0)                   # → theta - eps
            g_minus = erm_flat_grad()
            perturb(+1.0)                   # → theta  (restored)
            update_grad = 0.5 * (g_plus + g_minus)
        else:
            perturb(-1.0)                   # → theta  (restored)
            update_grad = g_plus

        # (5) Write update gradient into .grad and take an optimizer step.
        # The gradient is in float32; cast to each parameter's native dtype.
        model.zero_grad(set_to_none=True)
        i = 0
        for p in params:
            n = p.numel()
            p.grad = update_grad[i:i + n].view_as(p).to(p.dtype).detach().clone()
            i += n

        set_hparams(step)
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)

        # Diagnostics
        with torch.no_grad():
            avg_sim = 1.0
            if K > 1:
                norms   = torch.sqrt(torch.diagonal(M).clamp_min(1e-12))
                cos_mat = M / (norms.unsqueeze(0) * norms.unsqueeze(1))
                pairs   = list(itertools.combinations(range(K), 2))
                avg_sim = torch.stack([cos_mat[a, b] for a, b in pairs]).mean().item()

        approx_training_time = training_time + (time.perf_counter() - t0)
        print0(
            f"step:{step+1}/{train_steps} train_time:{approx_training_time:.3f}s"
            + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms"
            + f" M_scale:{M_scale.item():.4e} alpha:{alpha.item():.4e}"
            + f" eps_c:{eps_curv.norm().item():.4e} eps_a:{eps_align.norm().item():.4e}"
            + f" grad_sim:{avg_sim:.4f}",
            console=True, log=False,
        )

dist.destroy_process_group()
