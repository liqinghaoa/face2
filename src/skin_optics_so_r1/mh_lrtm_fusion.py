"""Exploratory RGB--M-sensitive--H-sensitive fusion candidates for X4-S0 only."""
from __future__ import annotations
import math
from typing import Any
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def split_rgb_m_h(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.ndim != 4 or x.shape[1:] != (5, 320, 256): raise ValueError("expected Bx5x320x256")
    return x[:, 0:3, :, :], x[:, 3:4, :, :], x[:, 4:5, :, :]


def _conv_init(module: nn.Module) -> None:
    for item in module.modules():
        if isinstance(item, nn.Conv2d): nn.init.kaiming_normal_(item.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(item, nn.Linear): nn.init.xavier_uniform_(item.weight); nn.init.zeros_(item.bias)
        elif isinstance(item, (nn.GroupNorm, nn.LayerNorm)): nn.init.ones_(item.weight); nn.init.zeros_(item.bias)


class ResidualDownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__(); self.main = nn.Sequential(nn.Conv2d(in_ch,out_ch,3,stride,1,bias=False),nn.GroupNorm(8,out_ch),nn.ReLU(),nn.Conv2d(out_ch,out_ch,3,1,1,bias=False),nn.GroupNorm(8,out_ch))
        self.skip = nn.Identity() if stride == 1 and in_ch == out_ch else nn.Sequential(nn.Conv2d(in_ch,out_ch,1,stride,bias=False),nn.GroupNorm(8,out_ch)); self.relu=nn.ReLU()
    def forward(self,x): return self.relu(self.main(x)+self.skip(x))


class MHEncoder(nn.Module):
    def __init__(self):
        super().__init__(); self.stem=nn.Sequential(nn.Conv2d(1,32,3,2,1,bias=False),nn.GroupNorm(8,32),nn.ReLU()); self.stage1=ResidualDownBlock(32,32,2); self.stage2=ResidualDownBlock(32,64,2); self.stage3=ResidualDownBlock(64,128,2); self.stage4=ResidualDownBlock(128,128,2); _conv_init(self)
    def forward_stages(self,x):
        a=self.stem(x); b=self.stage1(a); c=self.stage2(b); d=self.stage3(c); e=self.stage4(d); return a,b,c,d,e
    @staticmethod
    def gap(x): return torch.flatten(torch.mean(x,dim=(2,3)),1)


class TriMMTM(nn.Module):
    def __init__(self, rgb_ch:int,mh_ch:int,bottleneck:int):
        super().__init__(); self.shared=nn.Sequential(nn.Linear(rgb_ch+mh_ch+mh_ch,bottleneck),nn.ReLU()); self.rgb_head=nn.Linear(bottleneck,rgb_ch); self.m_head=nn.Linear(bottleneck,mh_ch); self.h_head=nn.Linear(bottleneck,mh_ch); _conv_init(self)
        for head in (self.rgb_head,self.m_head,self.h_head): nn.init.normal_(head.weight,0.,1e-3); nn.init.zeros_(head.bias)
    def forward(self,r,m,h):
        q=self.shared(torch.cat([r.mean((2,3)),m.mean((2,3)),h.mean((2,3))],1)); gates=[1+.1*torch.tanh(head(q)) for head in (self.rgb_head,self.m_head,self.h_head)]
        return r*gates[0][...,None,None],m*gates[1][...,None,None],h*gates[2][...,None,None],gates


class LowRankMH(nn.Module):
    rank=4
    def __init__(self):
        super().__init__(); self.U=nn.Parameter(torch.empty(4,129,128)); self.V=nn.Parameter(torch.empty(4,129,128)); self.norm=nn.LayerNorm(128); nn.init.xavier_uniform_(self.U); nn.init.xavier_uniform_(self.V)
    def forward(self,m,h):
        one=torch.ones((m.shape[0],1),dtype=m.dtype,device=m.device); ma=torch.cat([m,one],1); ha=torch.cat([h,one],1)
        return self.norm((torch.einsum('bi,rij->brj',ma,self.U)*torch.einsum('bi,rij->brj',ha,self.V)).sum(1)/math.sqrt(self.rank))


class FusionNet(nn.Module):
    residual_scale=0.1
    def __init__(self, variant:str):
        super().__init__(); assert variant in {'LATE_CONCAT','MH_LRF','MH_LRTM'}; self.variant=variant
        self.rgb=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1); self.m_encoder=MHEncoder(); self.h_encoder=MHEncoder()
        self.mmtm1=TriMMTM(128,64,64) if variant=='MH_LRTM' else None; self.mmtm2=TriMMTM(256,128,128) if variant=='MH_LRTM' else None
        if variant=='LATE_CONCAT': self.late_head=nn.Sequential(nn.Linear(768,256),nn.ReLU(),nn.Dropout(.2),nn.Linear(256,2)); _conv_init(self.late_head)
        else:
            self.rgb_head=nn.Linear(512,2); self.lmf=LowRankMH(); self.mh_projection=nn.Sequential(nn.Linear(128,512),nn.LayerNorm(512),nn.ReLU()); self.delta_head=nn.Sequential(nn.Linear(640,128),nn.ReLU(),nn.Dropout(.2),nn.Linear(128,2)); _conv_init(self.mh_projection); _conv_init(self.delta_head); nn.init.normal_(self.delta_head[-1].weight,0.,1e-3); nn.init.zeros_(self.delta_head[-1].bias)
    def _rgb_stages(self,x):
        r=self.rgb.maxpool(self.rgb.relu(self.rgb.bn1(self.rgb.conv1(x)))); r=self.rgb.layer1(r); r2=self.rgb.layer2(r); return r2
    def forward(self,x,return_audit=False):
        rgb,m,h=split_rgb_m_h(x); r2=self._rgb_stages(rgb); ms=self.m_encoder.stem(m); hs=self.h_encoder.stem(h); m1=self.m_encoder.stage1(ms); h1=self.h_encoder.stage1(hs); m2=self.m_encoder.stage2(m1); h2=self.h_encoder.stage2(h1); gates=[]
        if self.mmtm1: r2,m2,h2,g=self.mmtm1(r2,m2,h2); gates+=g
        r3=self.rgb.layer3(r2); m3=self.m_encoder.stage3(m2); h3=self.h_encoder.stage3(h2)
        if self.mmtm2: r3,m3,h3,g=self.mmtm2(r3,m3,h3); gates+=g
        r4=self.rgb.layer4(r3); m4=self.m_encoder.stage4(m3); h4=self.h_encoder.stage4(h3); zr=torch.flatten(self.rgb.avgpool(r4),1); zm=self.m_encoder.gap(m4); zh=self.h_encoder.gap(h4)
        audit={'rgb_stage2':r2,'rgb_stage3':r3,'rgb_stage4':r4,'m_stage4':m4,'h_stage4':h4,'z_rgb':zr,'z_m':zm,'z_h':zh,'gates':gates}
        if self.variant=='LATE_CONCAT': logits=self.late_head(torch.cat([zr,zm,zh],1)); audit['late_embedding']=torch.cat([zr,zm,zh],1); audit['rgb_logits']=None; audit['delta_logits']=None
        else:
            rgb_logits=self.rgb_head(zr); zmh=self.lmf(zm,zh); v=self.mh_projection(zmh); delta=self.delta_head(torch.cat([zmh,zr*v],1)); logits=rgb_logits+self.residual_scale*delta; audit.update({'z_mh':zmh,'rgb_logits':rgb_logits,'delta_logits':delta,'residual':self.residual_scale*delta})
        return (logits,audit) if return_audit else logits
    def gradient_groups(self):
        groups={'RGB conv1':[self.rgb.conv1.weight],'RGB layer4':list(self.rgb.layer4.parameters()),'M stem':list(self.m_encoder.stem.parameters()),'M stage4':list(self.m_encoder.stage4.parameters()),'H stem':list(self.h_encoder.stem.parameters()),'H stage4':list(self.h_encoder.stage4.parameters())}
        if self.variant != 'LATE_CONCAT': groups['RGB classifier']=list(self.rgb_head.parameters())
        if self.variant=='LATE_CONCAT': groups['late concat classifier']=list(self.late_head.parameters())
        else:
            groups.update({'LMF U factors':[self.lmf.U],'LMF V factors':[self.lmf.V],'MH projection':list(self.mh_projection.parameters()),'delta classifier first layer':list(self.delta_head[0].parameters()),'delta classifier final layer':list(self.delta_head[-1].parameters())})
        for n,unit in (('MMTM-1',self.mmtm1),('MMTM-2',self.mmtm2)):
            if unit: groups.update({f'{n} shared bottleneck':list(unit.shared.parameters()),f'{n} RGB gate head':list(unit.rgb_head.parameters()),f'{n} M gate head':list(unit.m_head.parameters()),f'{n} H gate head':list(unit.h_head.parameters())})
        return groups
