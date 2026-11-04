"""Imitation-learning baselines in plain PyTorch.

ACT (Zhao et al. 2023, "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"): ResNet-18 image
tokens + proprio into a transformer encoder-decoder that predicts an action chunk; a CVAE latent (from a separate
encoder over the ground-truth chunk) captures multimodality during training, and is zero at test time.

Diffusion Policy (Chi et al. 2023): a 1-D temporal U-Net with FiLM conditioning on image and proprio features
denoises an action chunk (DDPM training, DDIM sampling).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def resnet18_trunk(pretrained=True):
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    try:
        net = torchvision.models.resnet18(weights=weights)
    except Exception as e:  # offline: train from scratch
        print(f"[models] ImageNet weights unavailable ({type(e).__name__}); training the backbone from scratch")
        net = torchvision.models.resnet18(weights=None)
    return nn.Sequential(*list(net.children())[:-2])  # (B, 512, H/32, W/32)


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def sine_pos_2d(h, w, d, device):
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    k = torch.arange(d // 4, device=device)
    f = 1.0 / (10000 ** (4 * k / d))
    pe = torch.cat([torch.sin(x[..., None] * f), torch.cos(x[..., None] * f), torch.sin(y[..., None] * f), torch.cos(y[..., None] * f)], -1)
    return pe.view(h * w, d)


class ACT(nn.Module):
    def __init__(self, act_dim=20, n_cams=3, chunk=16, d=256, heads=8, enc_layers=4, dec_layers=6, latent=32, pretrained=True):
        super().__init__()
        self.chunk, self.latent, self.d = chunk, latent, d
        self.backbone = resnet18_trunk(pretrained)
        self.proj = nn.Conv2d(512, d, 1)
        self.cam_embed = nn.Parameter(torch.zeros(n_cams, d))
        self.proprio = nn.Linear(act_dim, d)
        self.latent_in = nn.Linear(latent, d)
        self.extra_pos = nn.Parameter(torch.randn(2, d) * 0.02)
        self.tf = nn.Transformer(d, heads, enc_layers, dec_layers, dim_feedforward=4 * d, dropout=0.1, batch_first=True)
        self.queries = nn.Parameter(torch.randn(chunk, d) * 0.02)
        self.head = nn.Linear(d, act_dim)
        # CVAE encoder over [CLS, proprio, chunk]
        self.vae_cls = nn.Parameter(torch.randn(1, d) * 0.02)
        self.vae_act = nn.Linear(act_dim, d)
        self.vae_prop = nn.Linear(act_dim, d)
        self.vae_pos = nn.Parameter(torch.randn(chunk + 2, d) * 0.02)
        self.vae_enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, batch_first=True), 4)
        self.vae_out = nn.Linear(d, 2 * latent)

    def image_tokens(self, imgs):
        B, C = imgs.shape[:2]
        x = (imgs.flatten(0, 1) - IMAGENET_MEAN.to(imgs.device)) / IMAGENET_STD.to(imgs.device)
        f = self.proj(self.backbone(x))  # (B*C, d, h, w)
        h, w = f.shape[-2:]
        f = f.flatten(2).transpose(1, 2).view(B, C, h * w, self.d)
        f = f + sine_pos_2d(h, w, self.d, imgs.device)[None, None] + self.cam_embed[None, :, None]
        return f.flatten(1, 2)

    def forward(self, imgs, prop, actions=None, pad=None):
        B = imgs.shape[0]
        mu = logvar = None
        if actions is not None:
            tok = torch.cat([self.vae_cls.expand(B, 1, -1), self.vae_prop(prop)[:, None], self.vae_act(actions)], 1) + self.vae_pos
            mask = None if pad is None else torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=imgs.device), pad], 1)
            stats = self.vae_out(self.vae_enc(tok, src_key_padding_mask=mask)[:, 0])
            mu, logvar = stats.chunk(2, -1)
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        else:
            z = torch.zeros(B, self.latent, device=imgs.device)
        src = torch.cat([torch.stack([self.latent_in(z), self.proprio(prop)], 1) + self.extra_pos, self.image_tokens(imgs)], 1)
        out = self.tf(src, self.queries.expand(B, -1, -1))
        return self.head(out), mu, logvar

    def loss(self, imgs, prop, actions, pad, kl_weight=10.0):
        pred, mu, logvar = self(imgs, prop, actions, pad)
        l1 = (F.l1_loss(pred, actions, reduction="none").mean(-1) * (~pad)).sum() / (~pad).sum()
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
        return l1 + kl_weight * kl, dict(l1=l1.item(), kl=kl.item())

    @torch.no_grad()
    def act(self, imgs, prop):
        return self(imgs, prop)[0]


# ---------------------------------------------------------------- diffusion policy


class SpatialSoftmax(nn.Module):
    def forward(self, f):
        B, C, H, W = f.shape
        p = F.softmax(f.flatten(2), -1).view(B, C, H, W)
        ys, xs = torch.linspace(-1, 1, H, device=f.device), torch.linspace(-1, 1, W, device=f.device)
        return torch.cat([(p.sum(-1) * ys).sum(-1), (p.sum(-2) * xs).sum(-1)], -1)


class ObsEncoder(nn.Module):
    def __init__(self, act_dim, n_cams, out=256, pretrained=True):
        super().__init__()
        self.backbone = resnet18_trunk(pretrained)
        self.kp = nn.Conv2d(512, 64, 1)
        self.ss = SpatialSoftmax()
        self.fc = nn.Linear(n_cams * 128 + act_dim, out)

    def forward(self, imgs, prop):
        B, C = imgs.shape[:2]
        x = (imgs.flatten(0, 1) - IMAGENET_MEAN.to(imgs.device)) / IMAGENET_STD.to(imgs.device)
        k = self.ss(self.kp(self.backbone(x))).view(B, -1)
        return F.mish(self.fc(torch.cat([k, prop], -1)))


def time_embed(t, d):
    k = torch.arange(d // 2, device=t.device)
    f = torch.exp(-math.log(10000) * k / (d // 2))
    a = t.float()[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], -1)


class ResBlock1d(nn.Module):
    def __init__(self, cin, cout, cond):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv1d(cin, cout, 5, padding=2), nn.GroupNorm(8, cout), nn.Mish())
        self.c2 = nn.Sequential(nn.Conv1d(cout, cout, 5, padding=2), nn.GroupNorm(8, cout), nn.Mish())
        self.film = nn.Linear(cond, 2 * cout)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, c):
        h = self.c1(x)
        s, b = self.film(c).unsqueeze(-1).chunk(2, 1)
        return self.c2(h * (1 + s) + b) + self.skip(x)


class UNet1d(nn.Module):
    def __init__(self, act_dim, cond, dims=(128, 256, 512)):
        super().__init__()
        self.inp = nn.Conv1d(act_dim, dims[0], 1)
        self.down = nn.ModuleList()
        c = dims[0]
        for d in dims:
            self.down.append(nn.ModuleList([ResBlock1d(c, d, cond), ResBlock1d(d, d, cond)]))
            c = d
        self.mid = ResBlock1d(c, c, cond)
        self.up = nn.ModuleList()
        for d in reversed(dims[:-1]):
            self.up.append(nn.ModuleList([ResBlock1d(c + d, d, cond), ResBlock1d(d, d, cond)]))
            c = d
        self.out = nn.Conv1d(c, act_dim, 1)

    def forward(self, x, c):
        x = self.inp(x.transpose(1, 2))
        skips = []
        for i, (a, b) in enumerate(self.down):
            x = b(a(x, c), c)
            skips.append(x)
            if i < len(self.down) - 1:
                x = F.avg_pool1d(x, 2)
        x = self.mid(x, c)
        skips.pop()
        for a, b in self.up:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = b(a(torch.cat([x, skips.pop()], 1), c), c)
        return self.out(x).transpose(1, 2)


class DiffusionPolicy(nn.Module):
    def __init__(self, act_dim=20, n_cams=3, chunk=16, steps=100, infer_steps=10, pretrained=True):
        super().__init__()
        self.chunk, self.steps, self.infer_steps = chunk, steps, infer_steps
        self.obs = ObsEncoder(act_dim, n_cams, 256, pretrained)
        self.net = UNet1d(act_dim, 256 + 128)
        s = torch.linspace(0, steps, steps + 1) / steps
        ac = torch.cos((s + 0.008) / 1.008 * math.pi / 2) ** 2  # squared-cosine schedule
        ac = ac / ac[0]
        self.register_buffer("alpha_bar", ac[1:].clamp(1e-5, 1.0))

    def cond(self, imgs, prop, t):
        return torch.cat([self.obs(imgs, prop), time_embed(t, 128)], -1)

    def loss(self, imgs, prop, actions, pad):
        B = actions.shape[0]
        t = torch.randint(0, self.steps, (B,), device=actions.device)
        ab = self.alpha_bar[t].view(B, 1, 1)
        noise = torch.randn_like(actions)
        x = ab.sqrt() * actions + (1 - ab).sqrt() * noise
        pred = self.net(x, self.cond(imgs, prop, t))
        l = (F.mse_loss(pred, noise, reduction="none").mean(-1) * (~pad)).sum() / (~pad).sum()
        return l, dict(mse=l.item())

    @torch.no_grad()
    def act(self, imgs, prop):
        """DDIM (eta 0) sampling."""
        B = imgs.shape[0]
        obs = self.obs(imgs, prop)
        x = torch.randn(B, self.chunk, self.net.out.out_channels, device=imgs.device)
        ts = torch.linspace(self.steps - 1, 0, self.infer_steps, device=imgs.device).long()
        for i, t in enumerate(ts):
            tt = t.expand(B)
            eps = self.net(x, torch.cat([obs, time_embed(tt, 128)], -1))
            ab = self.alpha_bar[t]
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1.5, 1.5)
            ab_prev = self.alpha_bar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=x.device)
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        return x


def build(kind, **kw):
    return {"act": ACT, "dp": DiffusionPolicy}[kind](**kw)
