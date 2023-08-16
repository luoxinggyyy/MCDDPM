
   
import math
from telnetlib import PRAGMA_HEARTBEAT
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


def drop_connect(x, drop_ratio): #随机丢弃，drop-ratio为丢弃率
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(p=keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask) #张量与mask相乘
    return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module): #时间位置编码
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb, freeze=False),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class ConditionalEmbedding(nn.Module): #条件编码
    def __init__(self, num_labels, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        self.condEmbedding = nn.Sequential(
            nn.Embedding(num_embeddings=num_labels + 1, embedding_dim=d_model, padding_idx=0),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        emb = self.condEmbedding(t)
        return emb

class ConditionalEmbedding1(nn.Module): #条件编码，这个是针对孔隙度调整的
    def __init__(self, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        self.condEmbedding1 = nn.Sequential(
            nn.Linear(1, d_model),
            Swish(),
            nn.Linear(d_model, dim),
        )

    def forward(self, t):
        t = torch.unsqueeze(t, dim=1)
        emb = self.condEmbedding1(t)
        return emb

def rockEmbedding(label,dim):
    a = torch.zeros(len(label),dim)
    for i in range(len(label)):
        a[i,:] = torch.full((1,dim),label[i])
    return a
    
class DownSample(nn.Module):  #下采样
    def __init__(self, in_ch):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(in_ch, in_ch, 5, stride=2, padding=2)

    def forward(self, x, temb, cemb):
        x = self.c1(x) + self.c2(x)  #拼接
        return x


class UpSample(nn.Module):   #上采样
    def __init__(self, in_ch):
        super().__init__()
        self.c = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.t = nn.ConvTranspose2d(in_ch, in_ch, 5, 2, 2, 1)

    def forward(self, x, temb, cemb):
        _, _, H, W = x.shape
        x = self.t(x)
        x = self.c(x)
        return x


class AttnBlock(nn.Module):  #注意力机制模块
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(8, in_ch) #归一化，是bn层的变种
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)

    def forward(self, x): #torch-bmm函数 ：一个批量矩阵乘法操作，计算查询和关键特征图之间的点积，
                                           #代表了输入特征图中每一对空间位置之间的注意权重。
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)  #这里使用注意力权重w和特征map-v的加权和。这产生了一个新的特征图h
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h   #这里是输出原图x与注意力特征map-h的和



class ResBlock(nn.Module):  #残差模块
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.cond_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()


    def forward(self, x, temb, labels):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None] #时间
        # labels = labels.to('cuda:0')
        h += self.cond_proj(labels)[:, :, None, None] #标签
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h  #该函数的输出是剩余连接和注意力块的输出（如果attn为真）或卷积块的输出（如果attn为假）之和


class UNet(nn.Module): #unet 来进行编码解码
    def __init__(self, T, ch, ch_mult, num_res_blocks, dropout):
        super().__init__()
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        # self.cond_embedding = ConditionalEmbedding(num_labels, ch, tdim)
        self.cond_embedding1 = ConditionalEmbedding1(ch, tdim)
        self.head = nn.Conv3d(1, ch, kernel_size=3, stride=1, padding=1)

        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(in_ch=now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout, attn=False))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(8, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 1, 3, stride=1, padding=1)
        )
 
    def rockEmbedding(label,dim):
        for i in range(len(label)):
            label[i,:] = torch.full((1,dim),label[i])
        return label

    def forward(self, x, t, labels):
        # Timestep embedding
        temb = self.time_embedding(t)
        # cemb = self.cond_embedding1(labels)
        cemb = self.cond_embedding1(labels)  #以作测试，放在cpu上的
        # cemb = rockEmbedding(labels,512).to('cuda')
        # cemb = labels.repeat(32,1).transpose(0,1).requires_grad_(True)  #这两个是将通道扩张成512，结果发现gpu计算不够
        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb, cemb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb, cemb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb, cemb)
        h = self.tail(h)

        assert len(hs) == 0
        return h


if __name__ == '__main__':
    batch_size = 8
    model = UNet(
        T=1000, ch=128, ch_mult=[1, 2, 2, 2],
        num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 3, 64, 64)
    t = torch.randint(1000, size=[batch_size])
    labels = torch.randint(10, size=[batch_size]).float()
    # resB = ResBlock(128, 256, 64, 0.1)
    # x = torch.randn(batch_size, 128, 32, 32)
    # t = torch.randn(batch_size, 64)
    # labels = torch.randn(batch_size, 64)
    # y = resB(x, t, labels)
    y = model(x, t, labels)
    print(y.shape)

