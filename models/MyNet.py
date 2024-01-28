import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding


class Model(nn.Module):
    '''
    CrossGNN
    '''
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.fgn = FGN(configs)
    def forward(self,x,x_mark_enc):
        fgn = self.fgn(x,x_mark_enc)  # 32*96*7
        return fgn
class FGN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_size = args.embed_size  # 128  一个batch包含的样本个数
        self.hidden_size = args.hidden_size  #  256 隐藏层大小
        self.number_frequency = 1
        self.pre_length = args.pred_len  # 改  # 预测长度  12  --- 改成96
        # self.feature_size = args.feature_size  # 140 特征个数 特征的维度或特征向量的长度
        self.seq_length = args.seq_len  # 12 输入序列的一条数据的长度 或者维度  ---- 改成96
        self.frequency_size = self.embed_size // self.number_frequency
        self.hidden_size_factor = args.hidden_size_factor
        self.sparsity_threshold = args.sparsity_threshold
        self.hard_thresholding_fraction = args.hard_thresholding_fraction
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))  # 1*128
        # self.layer_norm = nn.LayerNorm(args.d_model)  # 归一化 提高稳定和泛化
        self.layer = args.e_layers
        self.w1 = nn.Parameter( # 2*128*128
            self.scale * torch.randn(2, self.frequency_size, self.frequency_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor)) # # 2*128
        self.w2 = nn.Parameter( # 2*128*128
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor, self.frequency_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size))# 2*128
        self.w3 = nn.Parameter(# 2*128*128
            self.scale * torch.randn(2, self.frequency_size,
                                     self.frequency_size * self.hidden_size_factor))
        self.b3 = nn.Parameter(# 2*128
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.embeddings_10 = nn.Parameter(torch.randn(self.seq_length, 8)) # 12*8
        self.fc = nn.Sequential(
            nn.Linear(self.embed_size * 8, 64), # 1024*64
            nn.LeakyReLU(),
            nn.Linear(64, self.hidden_size), # 64*256
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length) #256*12
        )

        self.projection = nn.Linear(
            args.d_model, args.c_out, bias=True)


        self.predict_linear = nn.Linear(
            self.seq_length, self.pre_length + self.seq_length)

        self.enc_embedding = DataEmbedding(args.enc_in, args.d_model, args.embed, args.freq,
                                           args.dropout)


        self.to('cuda:0')

    def tokenEmb(self, x):
        x = x.unsqueeze(2)
        y = self.embeddings
        return x * y

    # FourierGNN
    def fourierGC(self, x, B, N, L):

        o1_real = F.relu(# 32 337 128
            torch.einsum('bli,ii->bli', x.real, self.w1[0]) - \
            torch.einsum('bli,ii->bli', x.imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag = F.relu(# 32 337 128
            torch.einsum('bli,ii->bli', x.imag, self.w1[0]) + \
            torch.einsum('bli,ii->bli', x.real, self.w1[1]) + \
            self.b1[1]
        )

        # 1 layer
        y = torch.stack([o1_real, o1_imag], dim=-1) # 32 337 128 2
        y = F.softshrink(y, lambd=self.sparsity_threshold)

        o2_real = F.relu(# 32 337 128
            torch.einsum('bli,ii->bli', o1_real, self.w2[0]) - \
            torch.einsum('bli,ii->bli', o1_imag, self.w2[1]) + \
            self.b2[0]
        )

        o2_imag = F.relu(# 32 337 128
            torch.einsum('bli,ii->bli', o1_imag, self.w2[0]) + \
            torch.einsum('bli,ii->bli', o1_real, self.w2[1]) + \
            self.b2[1]
        )

        # 2 layer
        x = torch.stack([o2_real, o2_imag], dim=-1)# 32 337 128 2
        x = F.softshrink(x, lambd=self.sparsity_threshold)# 32 337 128 2
        x = x + y

        o3_real = F.relu(# 32 337 128
                torch.einsum('bli,ii->bli', o2_real, self.w3[0]) - \
                torch.einsum('bli,ii->bli', o2_imag, self.w3[1]) + \
                self.b3[0]
        )

        o3_imag = F.relu(# 32 337 128
                torch.einsum('bli,ii->bli', o2_imag, self.w3[0]) + \
                torch.einsum('bli,ii->bli', o2_real, self.w3[1]) + \
                self.b3[1]
        )

        # 3 layer
        z = torch.stack([o3_real, o3_imag], dim=-1)# 32 337 128 2
        z = F.softshrink(z, lambd=self.sparsity_threshold)# 32 337 128 2
        z = z + x
        z = torch.view_as_complex(z)# 32 337 128
        return z

    def forward(self, x,x_mark_enc):

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev

        x = self.enc_embedding(x, x_mark_enc)
        x = x.permute(0, 2, 1).contiguous() # 32*7*96

        B, N, L = x.shape
        # B*N*L ==> B*NL
        x = x.reshape(B, -1)  # 32*672
        # embedding B*NL ==> B*NL*D
        x = self.tokenEmb(x) # 32*672*128

        # FFT B*NL*D ==> B*NT/2*D
        x = torch.fft.rfft(x, dim=1, norm='ortho')  # 32*337*128

        x = x.reshape(B, (N*L)//2+1, self.frequency_size)  # 32*337*128

        bias = x

        # FourierGNN
        x = self.fourierGC(x, B, N, L)

        x = x + bias # 32*337*128  残差

        x = x.reshape(B, (N*L)//2+1, self.embed_size) # 32*337*128

        # ifft
        x = torch.fft.irfft(x, n=N*L, dim=1, norm="ortho") # 32*672*128

        x = x.reshape(B, N, L, self.embed_size) # 32 *7*96*128
        x = x.permute(0, 1, 3, 2)  # B, N, D, L 32*7*128*96

        # projection
        x = torch.matmul(x, self.embeddings_10)   # 32*7*128*8
        x = x.reshape(B, N, -1)  # 32*7*1024
        x = self.fc(x)

        x = x.permute(0, 2, 1).contiguous()  # 32 * 96 *7

        x = self.predict_linear(x.permute(0, 2, 1)).permute(  0, 2, 1)  # 7*192*16

        # 归一
        # for i in range(self.layer):
        #     x = self.layer_norm(x)
        x = self.projection(x)

        # De-Normalization from Non-stationary Transformer
        t = stdev[:, 0, :]
        stest = stdev[:, 0, :].unsqueeze(1).repeat(1, self.pre_length + self.seq_length, 1)

        x = x * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pre_length + self.seq_length, 1))
        x = x + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pre_length + self.seq_length, 1))

        return x





