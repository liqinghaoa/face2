"""One-command P1-RGB audit, isolated smoke, formal five-fold training, and OOF summary."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();cfg=a.config if a.config.is_absolute() else ROOT/a.config;out=a.output_dir if a.output_dir.is_absolute() else ROOT/a.output_dir
 train=ROOT/'scripts/train/train_p1_rgb_p0aligned_resnet18_5fold.py';summary=ROOT/'scripts/evaluate/summarize_p1_rgb_p0aligned_resnet18_5fold.py';smoke=ROOT/'experiments/smoke/P1_RGB_P0Aligned_ResNet18_smoke'
 subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(), "CUDA is required for P1-RGB"'],cwd=ROOT,check=True)
 subprocess.run([sys.executable,str(train),'--config',str(cfg),'--output-dir',str(smoke),'--fold','0','--epochs','2','--smoke'],cwd=ROOT,check=True)
 subprocess.run([sys.executable,str(train),'--config',str(cfg),'--output-dir',str(out)],cwd=ROOT,check=True)
 subprocess.run([sys.executable,str(summary),'--config',str(cfg),'--experiment-dir',str(out)],cwd=ROOT,check=True)
 print('P1_RGB_EXPERIMENT_COMPLETED='+str(out))
if __name__=='__main__':main()
