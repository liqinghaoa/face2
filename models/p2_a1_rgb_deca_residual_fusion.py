from __future__ import annotations
import math,torch
from torch import nn
from models.nyha_backbone_factory import build_nyha_classification_model
class P2A1Fusion(nn.Module):
 def __init__(self,input_dim=278,hidden=128,dropout=.1,initial_alpha=.1):
  super().__init__();self.rgb=build_nyha_classification_model('resnet18',2,'imagenet',False);self.rgb.fc=nn.Identity();self.aux=nn.Sequential(nn.Linear(input_dim,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,512),nn.LayerNorm(512));self.alpha_logit=nn.Parameter(torch.tensor(math.log(initial_alpha/(1-initial_alpha)),dtype=torch.float32));self.classifier=nn.Linear(512,2)
 def forward(self,image,aux_vector):
  z=self.rgb(image);a=torch.sigmoid(self.alpha_logit);za=self.aux(aux_vector);zf=z+a*za;return {'logits':self.classifier(zf),'z_rgb':z,'z_aux':za,'z_fused':zf,'alpha':a}
