import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat

class NewNet(nn.Module):
    def __init__(self, input_channel, hidden_dim, downscaling_factors, layers, heads, head_dim, window_size, relative_pos_embedding):
        super(NewNet, self).__init__()
        self.stage1 = StageModule(in_channels=input_channel, hidden_dimension=hidden_dim, layers=layers[0],
                                  downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.stage2 = StageModule(in_channels=hidden_dim, hidden_dimension=hidden_dim * 1, layers=layers[1],
                                  downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.stage3 = StageModule(in_channels=hidden_dim * 1, hidden_dimension=hidden_dim * 1, layers=layers[2],
                                  downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)


        # ----------------------------------------------------------------------------------------

        self.stage6 = StageModule_up(in_channels=hidden_dim * 1, hidden_dimension=hidden_dim * 1,
                                     layers=layers[2], upscaling_factor=downscaling_factors[2], num_heads=heads[2],
                                     head_dim=head_dim,
                                     window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.stage7 = StageModule_up(in_channels=hidden_dim * 1, hidden_dimension=hidden_dim * 1,
                                     layers=layers[1], upscaling_factor=downscaling_factors[1], num_heads=heads[1],
                                     head_dim=head_dim,
                                     window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.stage8 = StageModule_up_final(in_channels=hidden_dim * 1, hidden_dimension=input_channel,
                                     layers=layers[0], upscaling_factor=downscaling_factors[0], num_heads=heads[0],
                                     head_dim=head_dim,
                                     window_size=window_size, relative_pos_embedding=relative_pos_embedding)

    def forward(self, x):
        inp1 = x[0]
        inp2 = x[1]
        inp3 = x[2]
        inp4 = x[3]
        x1 = self.stage1(inp1)     # (4, 96, 72, 72)
        x1 = x1 + inp2
        x2 = self.stage2(x1)    # (4, 192, 36, 36)
        x2 = x2 + inp3
        x3 = self.stage3(x2)    # (4, 384, 18, 18)
        #x3 = x3 + inp4
        #x4 = self.stage4(x3)    # (4, 768, 9, 9)

        #x5 = self.stage5(x4, x3)    # (4, 768, 18, 18)
        x6 = self.stage6(x3, x2)    # (4, 384, 36, 36)
        x7 = self.stage7(x6, x1)    # (4, 192, 72, 72)
        x8 = self.stage8(x7)        # (4, 9, 288, 288)

        return [x8,x7,x6,x3]

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CA(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(CA, self).__init__()
        self.input_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        middle_channel = input_channels // reduction_ratio
        if middle_channel < 10:
            middle_channel = input_channels
        self.MLP1 = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, middle_channel),
            nn.ReLU(),
            nn.Linear(middle_channel, input_channels)
        )
        self.MLP2 = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, middle_channel),
            nn.ReLU(),
            nn.Linear(middle_channel, input_channels)
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        avg_values = self.avg_pool(x)
        max_values = self.max_pool(x)
        out = self.MLP1(avg_values) + self.MLP2(max_values)
        scale = x * torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
        scale = scale.permute(0, 2, 3, 1)
        return scale

class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask


def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

        self.ca = CA(dim)

    def forward(self, x):
        if self.shifted:
            # 左上角移动
            x = self.cyclic_shift(x)

        b, n_h, n_w, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)

        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)

        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)

        out = self.ca(out)

        return out

# dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
# shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding
class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x



class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = torch.reshape(x, (b, c, new_h, self.downscaling_factor, new_w, self.downscaling_factor))
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = torch.reshape(x, (b, c * (self.downscaling_factor ** 2), new_h, new_w)).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x

class PatchExpanding(nn.Module):
    def __init__(self, in_channels, out_channels, upscaling_factor):
        super().__init__()
        self.upscaling_factor = upscaling_factor
        self.linear = nn.Linear(in_channels // (upscaling_factor ** 2), out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h * self.upscaling_factor, w * self.upscaling_factor
        new_c = int(c // (self.upscaling_factor ** 2))
        x = torch.reshape(x, (b, new_c, self.upscaling_factor, self.upscaling_factor, h, w))
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = torch.reshape(x, (b, new_c, new_h, new_w)).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, mid_channel=None):
        super(DoubleConv, self).__init__()
        if not mid_channel:
            mid_channel = out_channel
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(True),
            nn.Conv2d(mid_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                            downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                DoubleConv(hidden_dimension, hidden_dimension)
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shifted_block, cnn in self.layers:
            x = x.permute(0, 3, 1, 2)
            local_x = cnn(x)
            x = x.permute(0, 2, 3, 1)
            local_x = local_x.permute(0, 2, 3, 1)
            global_x = regular_block(x)
            global_x = shifted_block(global_x)
            x = local_x + global_x
        return x.permute(0, 3, 1, 2)

class StageModule_up(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, upscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchExpanding(in_channels=in_channels, out_channels=hidden_dimension,
                                              upscaling_factor=upscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            ]))

    def forward(self, x, x2):
        x = self.patch_partition(x)
        x2 = x2.permute(0, 2, 3, 1)
        x = x + x2
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 3, 1, 2)

class StageModule_up_vgg(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, upscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchExpanding(in_channels=in_channels, out_channels=hidden_dimension,
                                              upscaling_factor=upscaling_factor)

        self.layers = nn.ModuleList([])
        self.convs = []
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            ]))


    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 3, 1, 2)

class StageModule_up_final(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, upscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchExpanding(in_channels=in_channels, out_channels=hidden_dimension,
                                              upscaling_factor=upscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 3, 1, 2)


class SwinTransformer(nn.Module):
    def __init__(self, *, hidden_dim, layers, heads, channels=3, num_classes=1000, head_dim=32, window_size=7,
                 downscaling_factors=(4, 2, 2, 2), relative_pos_embedding=True):
        super().__init__()

        self.stage1 = StageModule(in_channels=channels, hidden_dimension=hidden_dim, layers=layers[0],
                                  downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.stage2 = StageModule(in_channels=hidden_dim, hidden_dimension=hidden_dim * 2, layers=layers[1],
                                  downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.stage3 = StageModule(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2],
                                  downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.stage4 = StageModule(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 8, layers=layers[3],
                                  downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 8),
            nn.Linear(hidden_dim * 8, num_classes)
        )

    def forward(self, img):
        x = self.stage1(img)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = x.mean(dim=[2, 3])
        return self.mlp_head(x)
