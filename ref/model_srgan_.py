# SRGAN

'''
    メモ
    基本的な用語については、ゼミの資料を参照

    非線形な活性化関数 f(x)=tanh(x), f(x)=(1+e^-x)-1が利用されていたが、ReLUを利用することで学習を高速化できる
    勾配消失問題(学習は予測値と実際の値の茣蓙を最小にする過程で進が、活性化関数の勾配がゼロに近づくことで学習が進まなくなる)

-----
    https://arxiv.org/abs/1609.04802, https://arxiv.org/pdf/1609.04802.pdf - original SRGAN
    https://qiita.com/yu4u/items/7e93c454c9410c4b5427#relu
    https://kotobank.jp/word/%E5%8B%BE%E9%85%8D%E6%B6%88%E5%A4%B1%E5%95%8F%E9%A1%8C-2132547
    https://www.youtube.com/watch?v=7FO9qDOhRCc - reference
    https://ichi.pro/seiseiteki-tekitaiteki-nettowa-ku-to-cho-kaizo-gan-srgan-249484521242726
    https://www.nttpc.co.jp/gpu/article/technical02.html

    https://confit.atlas.jp/guide/event-img/jsai2018/3A1-03/public/pdf?type=in
    http://cvlab.cs.miyazaki-u.ac.jp/laboratory/2018/oka_honbun.pdf, http://cvlab.cs.miyazaki-u.ac.jp/laboratory/2018/oka_presen.pdf - SRGANの3次元モデル超解像への拡張

-----

    conv - bn - relu - conv - bn - ...

    conv: convolution
    bn  : batch normalization
    relu: ReLU

'''

import torch
from torch import nn
from torch.nn.modules import padding

class ConvBlock(nn.Module):
    def __init__ (
        self,
        in_channels,
        out_channels,
        descriminator=False,
        use_act=True,
        use_bn=True,
        **kwargs, #key wards arguments
    ):
        super().__init__()
        self.use_act=use_act
        self.cnn=nn.Conv2d(in_channels, out_channels, **kwargs, bias=not use_bn) # CNNの初期化
        self.bn=nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act=( # activation
            nn.LeakyReLU(0.2, inplace=True)
            if descriminator
            else nn.PReLU(num_parameters=out_channels)
        )

    def forward(self, x):
        return self.act(self.bn(self.cnn(x))) if self.use_act else self.bn(self.cnn(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_c, scale_factor):
        super().__init__()
        self.conv=nn.Conv2d(in_c, in_c * scale_factor ** 2, 3, 1, 1) # -> use pixel shuffle
        self.ps=nn.PixelShuffle(scale_factor) # in_c * 4, H, W --> in_c, H*2, W*2
        self.act=nn.PReLU(num_parameters=in_c)

    def forward(self, x):
        return self.act(self.ps(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block1=ConvBlock(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.block2=ConvBlock(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_act=False
        )
    
    def forward(self, x):
        out=self.block1(x)
        out=self.block2(out)
        return out + x

class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=16):
        super().__init__()
        # Input -> Conv -> PReLU (k9n64s1)
        self.initial=ConvBlock(in_channels, num_channels, kernel_size=9, stride=1, padding=4, use_bn=False)
        # Conv -> BN -> PReLU -> ... (k3n64s1)
        self.residuals=nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        # Conv -> BN -> Elementwise Sum (k3n64s1)
        self.convblock=ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding=1, use_act=False)
        # Conv -> PixelShuffler x2 -> PReLU (k3n256s1)
        self.upsamples=nn.Sequential(UpsampleBlock(num_channels, scale_factor=2), UpsampleBlock(num_channels, scale_factor=2))
        # Final Conv
        self.final=nn.Conv2d(num_channels, in_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x = self.convblock(x) + initial
        x = self.upsamples(x)
        return torch.tanh(self.final(x))

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super().__init__()
        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels,
                    feature,
                    kernel_size=3,
                    stride=1 + idx % 2, # 1, 2, 1, 2,...
                    descriminator=True,
                    padding=1,
                    use_act=True,
                    use_bn=False if idx == 0 else True,
                )
            )
            in_channels = feature

        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)), # 適応平均プーリング 96x96 -> 128 -> 192
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.blocks(x)
        return self.classifier(x)

def test():
    # low_resolution = 24 # 96x96 -> 24x24
    low_resolution = 100
    with torch.cuda.amp.autocast():
        x = torch.randn((5, 3, low_resolution, low_resolution))
        gen = Generator()
        gen_out = gen(x)
        disc = Discriminator()
        disc_out = disc(gen_out)

        print(gen_out.shape)
        print(disc_out.shape)

if __name__ == "__main__":
    test()

