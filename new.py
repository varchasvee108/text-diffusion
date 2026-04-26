import torch

ckpt = torch.load("checkpoints/ckpt_best_loss.pt", map_location="cpu")
print(ckpt["step"], ckpt["best_loss"])
