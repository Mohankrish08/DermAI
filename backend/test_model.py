from backend.model import EfficientNet_ViT_Metadata
import torch

model = EfficientNet_ViT_Metadata()

img = torch.randn(2, 3, 224, 224)
meta = torch.randn(2, 3)

out = model(img, meta)

print(out.shape)