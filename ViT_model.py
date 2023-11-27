import torch
import torch.nn as nn
import torch.nn.functional as F
class MultiHeadAttention(nn.Module):
    def __init__(self,emb_dim,num_heads,attn_drop_partial=0.,proj_drop_partial=0.) -> None:
        super().__init__()
        self.emb_dim=emb_dim
        self.num_heads=num_heads
        self.head_dim=emb_dim//num_heads
        self.QKV=nn.Linear(emb_dim,emb_dim*3)
        self.linear_projection=nn.Linear(emb_dim,emb_dim)
        self.scale=emb_dim**(-0.5) # 1/sqrt(dim)
        self.attn_dropout=nn.Dropout(p=attn_drop_partial)
        self.proj_dropout = nn.Dropout(p=proj_drop_partial)
    def forward(self,x):
        B,N,D=x.size()
        qkv:torch.Tensor=self.QKV(x).view(B,N,3,self.num_heads,self.emb_dim//self.num_heads).permute(2,0,3,1,4) #(B,P,3,head,sub_dim) -> (3,B,head,P,subdim) -> (B,head,P,subdim)
        Q,K,V=qkv[0],qkv[1],qkv[2]
        attn:torch.Tensor=(Q@K.transpose(2,3))*self.scale
        attn=attn.softmax(-1) #last dimension ,that is the one patch's attn to other patches
        attn=self.attn_dropout(attn) #atte.shape -> (B,head,P,P) V.shape->(B,head,P,subdim) -> x.shape (B,head,P,subdim) ->(B,P,head,subdim)
        x=(attn@V).transpose(1,2).reshape(B,N,D).contiguous()
        x=self.linear_projection(x)
        x=self.proj_dropout(x)
        return x
class MLP(nn.Module):
    def __init__(self,emb_dim,activate=nn.GELU) -> None:
        super().__init__()
        self.linear1=nn.Linear(emb_dim,emb_dim)
        self.nonlinear=activate()
        self.linear2=nn.Linear(emb_dim,emb_dim)
    def forward(self,x):
        x=self.linear1(x)
        x=self.nonlinear(x)
        x=self.linear2(x)
        return x
class AttentionLayer(nn.Module):
    def __init__(self,emb_dim,num_heads,activate=nn.GELU,norm_type=nn.LayerNorm,attn_drop_partial=0.,proj_drop_partial=0.) -> None:
        super().__init__()
        self.norm1=norm_type(emb_dim)
        self.MultiHeadAttention=MultiHeadAttention(emb_dim,num_heads,attn_drop_partial,proj_drop_partial)
        self.norm2=norm_type(emb_dim)
        self.MLP=MLP(emb_dim,activate)
    def forward(self,x):
        x=self.MultiHeadAttention(self.norm1(x))+x
        x=self.MLP(self.norm2(x))+x
        return x
class PatchEmbedding(nn.Module):
    def __init__(self,image_size,patch_size,in_channel,emb_dim=1024) -> None:
        super().__init__()
        self.image_size=image_size
        self.patch_size=patch_size
        self.emb_dim=emb_dim
        self.num_patches=(image_size//patch_size)
        self.linear_projection=nn.Linear(in_channel*(patch_size**2),emb_dim)
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
a=torch.tensor(range(2*3*4*5),dtype=torch.float32).view(2,3,4,5)
print(a)
print(a.permute(3,1,0,2))#(5,3,2,4)
unfold=nn.Unfold(kernel_size=2,stride=2)
print(a)
a=unfold(a)
b=torch.tensor(range(2*3*4*4),dtype=torch.float32).view(2,3,4,4)
c=torch.tensor(range(2*3*4*5),dtype=torch.float32).view(2,3,4,5)
b=torch.tensor(range(2*2)).reshape(2,2)
c=torch.tensor(range(2*4)).reshape(2,4)
print(b)
print(c)
print(b@c.transpose(0,1))
'''print(a.transpose(1, 2))
print(a.transpose(1, 2).view(2,-1,3*2*2))
print(pos_embed = nn.Parameter(torch.zeros(1, 2 + 1, 10)))'''
##embed=PatchEmbedding(4,2,3,20)
#a=torch.tensor(range(2*5*5*3),dtype=torch.float32).view(2,3,5,5)
#print(embed(a))
