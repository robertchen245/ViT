import torch
import torch.nn as nn
import torch.nn.functional as F
class MultiHeadAttention(nn.Module):
    def __init__(self,emb_dim,num_heads) -> None:
        pass
class PatchEmbedding(nn.Module):
    def __init__(self,image_size,patch_size,in_channel,emb_dim=1024) -> None:
        super().__init__()
        self.image_size=image_size
        self.patch_size=patch_size
        self.emb_dim=emb_dim
        self.num_patches=(image_size//patch_size)
        self.projection=nn.Linear(in_channel*(patch_size**2),emb_dim)
        self.unfold=nn.Unfold(kernel_size=patch_size,stride=patch_size)
    def forward(self,x:torch.Tensor):
        batch_size, in_channels, _, _ = x.size()
        x:torch.Tensor=self.unfold(x)
        x:torch.Tensor=x.transpose(1,2).contiguous()
        x:torch.Tensor=self.projection(x)
        return x
class ViT(nn.Module):
    def __init__(self,image_size,patch_size,in_channel,num_classes,emb_dim) -> None:
        super().__init__()
        self.embedding=PatchEmbedding(image_size,patch_size,in_channel,emb_dim)
        self.FFN=nn.Linear(emb_dim,num_classes)
        self.cls_token=nn.Parameter(torch.randn(1,1,emb_dim))
    def forward(self,x):
        pass
a=torch.tensor(range(2*5*5*3),dtype=torch.float32).view(2,3,5,5)
unfold=nn.Unfold(kernel_size=2,stride=2)
print(a)
a=unfold(a)
print(a.transpose(1, 2))
print(a.transpose(1, 2).view(2,-1,3*2*2))
print(pos_embed = nn.Parameter(torch.zeros(1, 2 + 1, 10)))
##embed=PatchEmbedding(4,2,3,20)
#a=torch.tensor(range(2*5*5*3),dtype=torch.float32).view(2,3,5,5)
#print(embed(a))
