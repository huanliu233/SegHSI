import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from functools import partial, reduce, wraps
import math
from einops import rearrange

class Conv2dBlock(nn.Module): # padding 卷积+norm+激活 
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='none', pad_type='zero',  use_bias = True,groups=1):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'softmax':
            self.activation == nn.Softmax()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,groups=groups, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
def default(val, default_val):
    return default_val if val is None else val
def cache_method_decorator(cache_attr, cache_namespace, reexecute = False):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, key_namespace=None, fetch=False, set_cache=True, **kwargs):
            namespace_str = str(default(key_namespace, ''))
            _cache = getattr(self, cache_attr)
            _keyname = f'{cache_namespace}:{namespace_str}'

            if fetch:
                val = _cache[_keyname]
                if reexecute:
                    fn(self, *args, **kwargs)
            else:
                val = fn(self, *args, **kwargs)
                if set_cache:
                    setattr(self, cache_attr, {**_cache, **{_keyname: val}})
            return val
        return wrapper
    return inner_fn

def sort_key_val(t1, t2, dim=-1):
    values, indices = t1.sort(dim=dim)
    t2 = t2.expand_as(t1)
    return values, t2.gather(dim, indices)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class newMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2dBlock(in_features,hidden_features,1,1,0,norm='bn',activation='gelu')
        self.dw = Conv2dBlock(hidden_features,hidden_features,3,1,1,norm='bn',activation='gelu',groups=hidden_features)
        self.fc2 = Conv2dBlock(hidden_features,in_features,1,1,0,norm='bn',activation='gelu')
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B,N,C = x.shape
        H = W = int(math.sqrt(N))
        x = x.transpose(1,2).reshape(B,C,H,W)
        x = self.fc1(x)
        x = self.dw(x)
        x = self.fc2(x)
        x = x.reshape(B,C,N).transpose(1,2)
        return x
class DwMlp(nn.Module):
    def __init__(self, in_features,Ch, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,window={3:2,5:3,7:3}):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2dBlock(in_features, hidden_features,1,1,0,norm='bn',activation='gelu')
        self.fc2 = Conv2dBlock(hidden_features, out_features,1,1,0,norm='bn',activation='gelu')
        self.drop = nn.Dropout(drop)          
        self.dwconv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1                                                                 # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) * (dilation - 1)) // 2         # Determine padding size. Ref: https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338
            cur_conv = Conv2dBlock(cur_head_split*Ch, cur_head_split*Ch,
                kernel_size=cur_window, 
                padding=padding_size,
                stride = 1,
                norm= 'bn', 
                activation= 'gelu',                        
                groups=cur_head_split*Ch,
            )
            self.dwconv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x*Ch for x in self.head_splits]
    def forward(self, x, image_size):
        B,N,C = x.shape
        H,W= image_size[0],image_size[1]
        x = x.transpose(1,2).reshape(B,C,H,W)             
        x_list = torch.split(x, self.channel_splits, dim=1)                      # Split according to channels.
        conv_x_img_list = [conv(x) for conv, x in zip(self.dwconv_list, x_list)]
        conv_x_img = torch.cat(conv_x_img_list, dim=1)
        x = self.fc2(self.fc1(conv_x_img)).reshape(B,C,H*W).transpose(1,2) 
        return x

    
class newBlock(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                  attn_drop=0., proj_drop=0.,  n_buckets=8,crpe=None):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        
        buckets_dic={n_buckets*2:2,n_buckets:3,n_buckets//2:3}
        self.z = nn.Linear(dim, n_buckets*2*2 + n_buckets*3 + n_buckets//2*3 , bias=qkv_bias)
        self.head_splits = []
        self.channel_splits = []
        for buckets, num in buckets_dic.items():                                                             # Use dilation=1 at default.
            self.head_splits.append(num)
            self.channel_splits.append(num*buckets) 
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.n_buckets = n_buckets
        self.norm1 = nn.LayerNorm(self.dim)
        self.proj = DwMlp(self.dim,Ch=head_dim,hidden_features=4*self.dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm2 = nn.LayerNorm(self.dim)
        self.crpe = crpe
    def forward(self, x):
        B,C,H,W = x.shape
        N = H*W
        x = x.reshape(B,C,-1).transpose(1,2)
        x_ = self.norm1(x)
        qkv = self.qkv(x_).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0,  3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]         
        z_list = torch.split(self.z(x_), self.channel_splits, dim=2)                      
        z_list = [z.reshape(B,N,heads_num,-1).permute(0,2,1,3).softmax(dim=2)
                            for z, heads_num in zip(z_list,self.head_splits)]
        q_list = torch.split(q, self.head_splits, dim=1)    
        k_list = torch.split(k, self.head_splits, dim=1)    
        v_list = torch.split(v, self.head_splits, dim=1)    
        k_list = [torch.einsum('bhNd,bhNn->bhnd', k, z) for k,z in zip(k_list,z_list)]
        v_list = [torch.einsum('bhNd,bhNn->bhnd', v, z) for v,z in zip(v_list,z_list)]
        attn_list = [torch.einsum('bhNd,bhnd->bhNn', q, k* self.scale).softmax(dim=-1)  for q,k in zip(q_list,k_list)]
        x_list = [torch.einsum('bhNn,bhnd->bhNd', attn, v).transpose(1,2).reshape(B, N,-1) for attn, v in zip(attn_list,v_list)]
        x = x + self.attn_drop(torch.cat(x_list,dim=2)) + self.crpe(q,v,size=[H,W])
        x_ = self.proj(self.norm2(x),[H,W])
        x = x + self.proj_drop(x_)
        x = x.transpose(1,2).reshape(B,C,H,W)
        return x

class newdeBlock(nn.Module):
    def __init__(self, dim,dedim,  num_heads=8, qkv_bias=False, qk_scale=None,
                  attn_drop=0., proj_drop=0.,  n_buckets=8,crpe=None):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        buckets_dic={n_buckets*2:2,n_buckets:3,n_buckets//2:3}
        self.z = nn.Linear(dedim, n_buckets*2*2 + n_buckets*3 + n_buckets//2*3 , bias=qkv_bias)
        self.head_splits = []
        self.channel_splits = []
        for buckets, num in buckets_dic.items():                                                             # Use dilation=1 at default.
            self.head_splits.append(num)
            self.channel_splits.append(num*buckets) 
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dedim, 2*dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.n_buckets = n_buckets
        self.norm1 = nn.LayerNorm(self.dim)
        self.norm1_1 = nn.LayerNorm(dedim)
        self.proj = DwMlp(self.dim,Ch=head_dim,hidden_features=4*self.dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm2 = nn.LayerNorm(self.dim)
        self.crpe = crpe
    def forward(self, x):
        x,dx=x[0],x[1]
        B, C, H, W = x.shape
        N = H * W
        B, Cdx, Hdx, Wdx = dx.shape
        Ndx = Hdx * Wdx
        x = x.reshape(B,C,-1).transpose(1,2)
        dx = dx.reshape(B,Cdx,-1).transpose(1,2)
        x_ = self.norm1(x)
        dx_ = self.norm1_1(dx)
        kv = self.kv(dx_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0,  3, 1, 4)
        k, v = kv[0], kv[1] 
        q = self.q(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute( 0,  2, 1, 3)
        z_list = torch.split(self.z(dx_), self.channel_splits, dim=2)                      
        z_list = [z.reshape(B,Ndx,heads_num,-1).permute(0,2,1,3).softmax(dim=2)
                            for z, heads_num in zip(z_list,self.head_splits)]
        q_list = torch.split(q, self.head_splits, dim=1)    
        k_list = torch.split(k, self.head_splits, dim=1)    
        v_list = torch.split(v, self.head_splits, dim=1)    
        k_list = [torch.einsum('bhNd,bhNn->bhnd', k, z) for k,z in zip(k_list,z_list)]
        v_list = [torch.einsum('bhNd,bhNn->bhnd', v, z) for v,z in zip(v_list,z_list)]
        attn_list = [torch.einsum('bhNd,bhnd->bhNn', q, k* self.scale).softmax(dim=-1)  for q,k in zip(q_list,k_list)]
        x_list = [torch.einsum('bhNn,bhnd->bhNd', attn, v).transpose(1,2).reshape(B, N,-1) for attn, v in zip(attn_list,v_list)]
        x = x + self.attn_drop(torch.cat(x_list,dim=2)) + self.crpe(q,v,size=[H,W],dsize=[Hdx,Wdx])
        x_ = self.proj(self.norm2(x),[H,W])
        x = x + self.proj_drop(x_)
        x = x.transpose(1,2).reshape(B,C,H,W)
        dx = dx.transpose(1,2).reshape(B,Cdx,Hdx,Wdx)
        return [x,dx]



class ConvRelPosEnc(nn.Module):
    """ Convolutional relative position encoding. """
    def __init__(self, Ch, window={3:2,5:3,7:3}):
        """
        Initialization.
            Ch: Channels per head.
            h: Number of heads.
            window: Window size(s) in convolutional relative positional encoding. It can have two forms:
                    1. An integer of window size, which assigns all attention heads with the same window size in ConvRelPosEnc.
                    2. A dict mapping window size to #attention head splits (e.g. {window size 1: #attention head split 1, window size 2: #attention head split 2})
                       It will apply different window size to the attention head splits.
        """
        super().__init__()
         
        
        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1                                                                 # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) * (dilation - 1)) // 2         # Determine padding size. Ref: https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338
            cur_conv = nn.Conv2d(cur_head_split*Ch, cur_head_split*Ch,
                kernel_size=(cur_window, cur_window), 
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),                          
                groups=cur_head_split*Ch,
            )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x*Ch for x in self.head_splits]

    def forward(self, q, v, size,dsize=None):
        B, h, N, Ch = q.shape
        H, W = size
        B, h, Nv, Ch = v.shape
        assert N == H * W

        # Convolutional relative position encoding.
        q_img = q                                                            # Shape: [B, h, H*W, Ch].
        v_img = v  
                                                    # Shape: [B, h, H*W, Ch].
        if dsize != None:
            dH, dW = dsize[0], dsize[1]
            v_img = rearrange(v_img, 'B h (H W) Ch -> B (h Ch) H W', H=dH, W=dW)               # Shape: [B, h, H*W, Ch] -> [B, h*Ch, H, W].
            v_img = F.interpolate(v_img,size=(H,W),mode='bilinear',align_corners=False)
        else: 
            v_img = rearrange(v_img, 'B h (H W) Ch -> B (h Ch) H W', H=H, W=W) 
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)                      # Split according to channels.
        conv_v_img_list = [conv(x) for conv, x in zip(self.conv_list, v_img_list)]
        conv_v_img = torch.cat(conv_v_img_list, dim=1)
        conv_v_img = rearrange(conv_v_img, 'B (h Ch) H W -> B h (H W) Ch', h=h)          # Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].

        EV_hat_img = q_img * conv_v_img


        return EV_hat_img.transpose(1,2).flatten(2)
class Clusterformer(nn.Module):

    def __init__(self, in_chans=3, n_class=1, n_buckets=64, layers=4, dim = 64):
        super().__init__()
       
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dim, kernel_size=2, stride=2,padding=0),
            LayerNorm(dim, eps=1e-6, data_format="channels_first")
        )
        # self.skip = nn.Conv2d(in_chans, dim, kernel_size=1, stride=1,padding=0)


        rposembed_layer = ConvRelPosEnc(dim//8)
        self.stage = nn.Sequential(
            *[newBlock(dim=dim, num_heads=8,n_buckets=n_buckets,crpe=rposembed_layer) for j in range(layers)]
        )
        self.posembed_layer =  nn.Conv2d(dim,dim,3,1,1,groups=dim)
        self.fc = Conv2dBlock(dim + in_chans, dim,1,1,0,norm='bn',activation='gelu')
        self.class0 = nn.Conv2d(dim, n_class,1,1,0)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias != None: 
                nn.init.constant_(m.bias, 0)
    def forward_features(self, x):



        return x# global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        y = self.stem(x)
        y = y + self.posembed_layer(y)
        y = self.stage(y)
        y = F.interpolate(y,size=x.size()[2:],mode='bilinear', align_corners=False)   
        y = self.fc(torch.cat((x,y),dim=1)) 
        y = self.class0(y)
        
        return y #,y0,y1,y2,y3

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
