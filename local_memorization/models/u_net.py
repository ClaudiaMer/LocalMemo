import torch 
from torch import nn
from torch.nn import functional as F

def sinusoidal_embedding(T, d):
    """Returns the standard positional embedding

    Args:
        T (int): number of time steps
        d (int): dimension

    Returns:
        torch.Tensor: embedding of shape (dim, T) 
    """
    embedding = torch.tensor([[i / 10_000 ** (2 * j / d) for j in range(d)]
                               for i in range(T)])
    sin_mask = torch.arange(0, T, 2)

    embedding[sin_mask] = torch.sin(embedding[sin_mask])
    embedding[1 - sin_mask] = torch.cos(embedding[sin_mask])

    return embedding

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class down_layer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_layer, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2, padding=0)
        self.conv = double_conv(in_ch,
                                 out_ch)

    def forward(self, x):
        x = self.conv(self.pool(x))
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.up_scale = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

    def forward(self, x1, x2): # x1 (bs,out_ch,w1,h1) x2 (bs,in_ch,w2,h2)
        x2 = self.up_scale(x2) # (bs,out_ch,2*w2,2*h2)
        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]

        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2]) # (bs,out_ch,w1,h1)
        x = torch.cat([x2, x1], dim=1) # (bs,2*out_ch,w1,h1)
        return x

class up_layer(nn.Module):
    def __init__(self, in_ch, out_ch): # !! 2*out_ch = in_ch !!
        super(up_layer, self).__init__()
        self.up = up(in_ch, out_ch)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2): # x1 (bs,out_ch,w1,h1) x2 (bs,in_ch,w2,h2)
        a = self.up(x1, x2) # (bs,2*out_ch,w1,h1)
        x = self.conv(a) # (bs,out_ch,w1,h1) because 2*out_ch = in_ch
        return x
    
class UNet(nn.Module):
    def __init__(self, T, mask_time=False, in_channels=1, time_emb_dim=100):
        super(UNet, self).__init__()
        self.conv1 = double_conv(in_channels, 64)
        self.down1 = down_layer(64, 128)
        self.down2 = down_layer(128, 256)
        self.down3 = down_layer(256, 512)
        self.down4 = down_layer(512, 1024)
        self.up1 = up_layer(1024, 512)
        self.up2 = up_layer(512, 256)
        self.up3 = up_layer(256, 128)
        self.up4 = up_layer(128, 64)
        self.last_conv = nn.Conv2d(64, in_channels, 1)
        
        self.time_mask = (not mask_time)*1
        # Time embedding
        self.time_embed = nn.Embedding(T, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(T, time_emb_dim)
        self.time_embed.requires_grad_(True)
        self.te1 = self._make_te(time_emb_dim, in_channels)
        self.te2 = self._make_te(time_emb_dim, 64)
        self.te3 = self._make_te(time_emb_dim, 128)
        self.te4 = self._make_te(time_emb_dim, 256)
        self.te5 = self._make_te(time_emb_dim, 512)
        self.te1_up = self._make_te(time_emb_dim, 1024)
        self.te2_up = self._make_te(time_emb_dim, 512)
        self.te3_up = self._make_te(time_emb_dim, 256)
        self.te4_up = self._make_te(time_emb_dim, 128)

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(nn.Linear(dim_in, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out))
    
    def forward(self, x , t): # x (bs,in_channels,w,d)
        bs = x.shape[0]
        t = self.time_embed(t) # -1 because diffusion uses t=1, ... ,1000
        # but embedding uses t= 0, ..., 999
        x1 = self.conv1(x+self.te1(t).reshape(bs, -1, 1, 1)*self.time_mask) # (bs,64,w,d)
        x2 = self.down1(x1+self.te2(t).reshape(bs, -1, 1, 1)*self.time_mask) # (bs,128,w/2,d/2)
        x3 = self.down2(x2+self.te3(t).reshape(bs, -1, 1, 1)*self.time_mask) # (bs,256,w/4,d/4)
        x4 = self.down3(x3+self.te4(t).reshape(bs, -1, 1, 1)*self.time_mask) # (bs,512,w/8,h/8)
        x5 = self.down4(x4+self.te5(t).reshape(bs, -1, 1, 1)*self.time_mask) # (bs,1024,w/16,h/16)
        x1_up = self.up1(x4, x5+self.te1_up(t).reshape(bs, -1, 1, 1)*self.time_mask) # (bs,512,w/8,h/8)
        x2_up = self.up2(x3, x1_up+self.te2_up(t).reshape(bs, -1, 1, 1)*self.time_mask) # (bs,256,w/4,h/4)
        x3_up = self.up3(x2, x2_up+self.te3_up(t).reshape(bs, -1, 1, 1)*self.time_mask) # (bs,128,w/2,h/2)
        x4_up = self.up4(x1, x3_up+self.te4_up(t).reshape(bs, -1, 1, 1)*self.time_mask) # (bs,64,w,h)
        output = self.last_conv(x4_up) # (bs,in_channels,w,h)
        return output
    

class UNetMedium(nn.Module):
    def __init__(self, T, in_channels=1, mask_time=False, time_emb_dim=100):
        super(UNetMedium, self).__init__()
        self.conv1 = double_conv(in_channels, 64)
        self.down1 = down_layer(64, 128)
        self.down2 = down_layer(128, 256)
        self.down3 = down_layer(256, 512)
        self.up1 = up_layer(512, 256)
        self.up2 = up_layer(256, 128)
        self.up3 = up_layer(128, 64)
        self.last_conv = nn.Conv2d(64, in_channels, 1)
        
        self.time_mask = (not mask_time)*1

        # Time embedding
        self.time_embed = nn.Embedding(T, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(T, time_emb_dim)
        self.time_embed.requires_grad_(True)
        self.te1 = self._make_te(time_emb_dim, in_channels)
        self.te2 = self._make_te(time_emb_dim, 64)
        self.te3 = self._make_te(time_emb_dim, 128)
        self.te4 = self._make_te(time_emb_dim, 256)
        self.te1_up = self._make_te(time_emb_dim, 512)
        self.te2_up = self._make_te(time_emb_dim, 256)
        self.te3_up = self._make_te(time_emb_dim, 128)

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(nn.Linear(dim_in, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out))
    
    def forward(self, x , t): # x (bs,in_channels,w,d)
        bs = x.shape[0]
        t = self.time_embed(t) # -1 because diffusion uses t=1, ... ,1000
        # but embedding uses t= 0, ..., 999
        x1 = self.conv1(x+self.te1(t).reshape(bs, -1, 1, 1)*self.time_mask) # (bs,64,w,d)
        x2 = self.down1(x1+self.te2(t).reshape(bs, -1, 1, 1)*self.time_mask) # (bs,128,w/2,d/2)
        x3 = self.down2(x2+self.te3(t).reshape(bs, -1, 1, 1)*self.time_mask) # (bs,256,w/4,d/4)
        x4 = self.down3(x3+self.te4(t).reshape(bs, -1, 1, 1)*self.time_mask) # (bs,512,w/8,h/8)
        x1_up = self.up1(x3, x4+self.te1_up(t).reshape(bs, -1, 1, 1)*self.time_mask) # (bs,512,w/8,h/8)
        x2_up = self.up2(x2, x1_up+self.te2_up(t).reshape(bs, -1, 1, 1)*self.time_mask) # (bs,256,w/4,h/4)
        x3_up = self.up3(x1, x2_up+self.te3_up(t).reshape(bs, -1, 1, 1)*self.time_mask) # (bs,128,w/2,h/2)
        output = self.last_conv(x3_up) # (bs,in_channels,w,h)
        return output
    


class UNetMini(nn.Module):
    def __init__(self, T, in_channels=1,mask_time=False, time_emb_dim=100):
        super(UNetMini, self).__init__()
        self.conv1 = double_conv(in_channels, 64)
        self.down1 = down_layer(64, 128)
        self.down2 = down_layer(128, 256)
        self.up1 = up_layer(256, 128)
        self.up2 = up_layer(128, 64)
        self.last_conv = nn.Conv2d(64, in_channels, 1)
        
        self.time_mask = (not mask_time)*1

        # Time embedding
        self.time_embed = nn.Embedding(T, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(T, time_emb_dim)
        self.time_embed.requires_grad_(True)
        self.te1 = self._make_te(time_emb_dim, in_channels)
        self.te2 = self._make_te(time_emb_dim, 64)
        self.te3 = self._make_te(time_emb_dim, 128)
        self.te1_up = self._make_te(time_emb_dim, 256)
        self.te2_up = self._make_te(time_emb_dim, 128)

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(nn.Linear(dim_in, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out))
    
    def forward(self, x , t): # x (bs,in_channels,w,d)
        bs = x.shape[0]
        t = self.time_embed(t) # -1 because diffusion uses t=1, ... ,1000
        # but embedding uses t= 0, ..., 999
        x1 = self.conv1(x+self.te1(t).reshape(bs, -1, 1, 1)*self.time_mask) # (bs,64,w,d)
        x2 = self.down1(x1+self.te2(t).reshape(bs, -1, 1, 1)*self.time_mask) # (bs,128,w/2,d/2)
        x3 = self.down2(x2+self.te3(t).reshape(bs, -1, 1, 1)*self.time_mask) # (bs,256,w/4,d/4)
        x1_up = self.up1(x2, x3+self.te1_up(t).reshape(bs, -1, 1, 1)*self.time_mask) # (bs,128,w/8,h/8)
        x2_up = self.up2(x1, x1_up+self.te2_up(t).reshape(bs, -1, 1, 1)*self.time_mask) # (bs,64,w/4,h/4)
        output = self.last_conv(x2_up) # (bs,in_channels,w,h)
        return output
    
class UNetMiniMini(nn.Module):
    def __init__(self, T, in_channels=1, mask_time=False, time_emb_dim=100):
        super(UNetMiniMini, self).__init__()
        self.conv1 = double_conv(in_channels, 16)
        self.down1 = down_layer(16, 32)
        self.down2 = down_layer(32, 64)
        self.up1 = up_layer(64, 32)
        self.up2 = up_layer(32, 16)
        self.last_conv = nn.Conv2d(16, in_channels, 1)
        
        # Time embedding
        self.time_mask = (not mask_time)*1

        self.time_embed = nn.Embedding(T, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(T, time_emb_dim)
        self.time_embed.requires_grad_(True)
        self.te1 = self._make_te(time_emb_dim, in_channels)
        self.te2 = self._make_te(time_emb_dim, 16)
        self.te3 = self._make_te(time_emb_dim, 32)
        self.te1_up = self._make_te(time_emb_dim, 64)
        self.te2_up = self._make_te(time_emb_dim, 32)

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(nn.Linear(dim_in, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out))
    
    def forward(self, x , t): # x (bs,in_channels,w,d)
        bs = x.shape[0]
        t = self.time_embed(t) # -1 because diffusion uses t=1, ... ,1000
        # but embedding uses t= 0, ..., 999
        x1 = self.conv1(x+self.te1(t).reshape(bs, -1, 1, 1)*self.time_mask) # (bs,64,w,d)
        x2 = self.down1(x1+self.te2(t).reshape(bs, -1, 1, 1)*self.time_mask) # (bs,128,w/2,d/2)
        x3 = self.down2(x2+self.te3(t).reshape(bs, -1, 1, 1)*self.time_mask) # (bs,256,w/4,d/4)
        x1_up = self.up1(x2, x3+self.te1_up(t).reshape(bs, -1, 1, 1)*self.time_mask) # (bs,128,w/8,h/8)
        x2_up = self.up2(x1, x1_up+self.te2_up(t).reshape(bs, -1, 1, 1)*self.time_mask) # (bs,64,w/4,h/4)
        output = self.last_conv(x2_up) # (bs,in_channels,w,h)
        return output

if __name__ == "__main__": 

    unet = UNetMini(1000,in_channels=1)
    x = torch.randn(10,1,80,80)
    print(unet(x, torch.ones(10).long()).shape)

