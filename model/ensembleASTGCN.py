import numpy as np
from scipy.sparse.linalg import eigs

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_tensor_type(torch.DoubleTensor)

def scaled_Laplacian(W):
    D = np.diag(np.sum(W, axis = 1))
    L = D - W
    lambda_max = eigs(L, k = 1, which = 'LR')[0].real
    return (2 * L) / lambda_max - np.identity(W.shape[0])
    
def cheb_polynomial(L_tilde, K):
    N = L_tilde.shape[0]
    cheb_polynomials = [np.identity(N), L_tilde.copy()]
    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
    return cheb_polynomials

class TAT(nn.Module):
    def __init__(self, inputs, vertices, timesteps):
        super(TAT, self).__init__()
        self.U1 = nn.Parameter(torch.DoubleTensor(vertices))
        self.U2 = nn.Parameter(torch.DoubleTensor(inputs, vertices))
        self.U3 = nn.Parameter(torch.DoubleTensor(inputs))
        self.be = nn.Parameter(torch.DoubleTensor(1, timesteps, timesteps))
        self.Ve = nn.Parameter(torch.DoubleTensor(timesteps, timesteps))

    def forward(self, x):
        print('Begin temporal attention')
        inner = torch.matmul(x.permute(0, 3, 2, 1), self.U1)
        print('Done with inner :)')
        lhs = torch.matmul(inner, self.U2)
        print('Multiplication :)')
        rhs = torch.matmul(self.U3, x)
        product = torch.matmul(lhs, rhs)
        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))
        E_normalized = F.softmax(E, dim=1)
        return E_normalized

class SAT(nn.Module):
    def __init__(self, inputs, vertices, timesteps):
        super(SAT, self).__init__()
        self.W1 = nn.Parameter(torch.DoubleTensor(timesteps))
        self.W2 = nn.Parameter(torch.DoubleTensor(inputs, timesteps))
        self.W3 = nn.Parameter(torch.DoubleTensor(inputs))
        self.bs = nn.Parameter(torch.DoubleTensor(1, vertices, vertices))
        self.Vs = nn.Parameter(torch.DoubleTensor(vertices, vertices))
        
    def forward(self, x):
        print('Begin spatial attention')
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)
        product = torch.matmul(lhs, rhs)
        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))
        S_normalized = F.softmax(S, dim=1)

        return S_normalized

class SAT_Conv(nn.Module):
    def __init__(self, K, cheb_polynomials, inputs, outputs):
        super(SAT_Conv, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.inputs = inputs
        self.outputs = outputs
        self.Theta = nn.ParameterList([nn.Parameter(torch.DoubleTensor(inputs, outputs)) for _ in range(K)])
        
    def forward(self, x, sat):
        batch_size, vertices, inputs, timesteps = x.shape

        final_outputs = []

        for t in range(timesteps):

            graph_signal = x[:, :, :, t]
            output = torch.zeros(batch_size, vertices, self.outputs)

            for k in range(self.K):
                T_k = self.cheb_polynomials[k]
                T_k_with_at = T_k.mul(sat)
                theta_k = self.Theta[k]
                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)
                output = output + rhs.matmul(theta_k)

            final_outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(final_outputs, dim=-1))

class ASTGCN_Block(nn.Module):
    def __init__(self, inputs, K, chev_filters, time_filters, strides, cheb_polynomials, vertices, timesteps):
        super(ASTGCN_Block, self).__init__()
        self.TAt = TAT(inputs, vertices, timesteps)
        self.SAt = SAT(inputs, vertices, timesteps)
        self.SAt_conv = SAT_Conv(K, cheb_polynomials, inputs, chev_filters)
        self.t_conv = nn.Conv2d(chev_filters, time_filters, kernel_size = (1, 3), stride = (1, strides), padding = (0, 1))
        self.residual_conv = nn.Conv2d(inputs, time_filters, kernel_size = (1, 1), stride = (1, strides))
        self.ln = nn.LayerNorm(time_filters)

    def forward(self, x):
        print('Begin block')
        batch_size, vertices, features, timestamps = x.shape
        tat = self.TAt(x)
        x_TAt = torch.matmul(x.reshape(batch_size, -1, timesteps), tat).reshape(batch_size, vertices, features, timesteps)
        sat = self.SAt(x_TAt)
        print('Finished attention')
        gcn = self.SAt_conv(x, sat)
        t_conv_output = self.t_conv(gcn.permute(0, 2, 1, 3))
        print('finished convolution')
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))
        x_residual = self.ln(F.relu(x_residual + t_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        return x_residual

class ASTGCN_Module(nn.Module):
    def __init__(self, blocks, traffic_features, K, chev_filters, time_filters, strides, cheb_polynomials, outputs, vertices):
        super(ASTGCN_Module, self).__init__()

        self.BlockList = nn.ModuleList([ASTGCN_Block(traffic_features, K, chev_filters, time_filters, strides, cheb_polynomials, vertices, strides * outputs)])

        self.BlockList.extend([ASTGCN_Block(time_filters, K, chev_filters, time_filters, 1, cheb_polynomials, vertices, outputs) for _ in range(blocks - 1)])

        self.final_conv = nn.Conv2d(outputs, outputs, kernel_size = (1, time_filters))
        
    def forward(self, x):
        print('Begin submodule')
        for block in self.BlockList:
            x = block(x)
        print('Begin final conv')
        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        return output

class ASTGCN(nn.Module):
    def __init__(self, blocks, traffic_features, K, chev_filters, time_filters, cheb_polynomials, outputs, hours, days, weeks, vertices):
        super(ASTGCN, self).__init__()
        self.hourly = ASTGCN_Module(blocks, traffic_features, K, chev_filters, time_filters, hours, cheb_polynomials, outputs, vertices)
#        self.daily = ASTGCN_Module(blocks, traffic_features, K, chev_filters, time_filters, days, cheb_polynomials, outputs, vertices)
#        self.weekly = ASTGCN_Module(blocks, traffic_features, K, chev_filters, time_filters, weeks, cheb_polynomials, outputs, vertices)
        self.W_h = nn.Parameter(torch.DoubleTensor(vertices, outputs))
#        self.W_d = nn.Parameter(torch.DoubleTensor(vertices, outputs))
#        self.W_w = nn.Parameter(torch.DoubleTensor(vertices, outputs))
        self.b = nn.Parameter(torch.DoubleTensor(vertices, outputs))
        
    def forward(self, X_h, X_d, X_w, W):
        h_out = self.hourly(X_h)
        print('Done with hourly.')
#        d_out = self.daily(X_d)
#        w_out = self.weekly(X_w)
#        fusion = self.W_h.mul(h_out) + self.W_d.mul(d_out) + self.W_w.mul(w_out) + self.b
        fusion = self.W_h.mul(h_out) + self.b
        
        return fusion
        

def get_model(blocks, traffic_features, K, gcn_filters, t_filters, pred_window, hours, days, weeks, vertices, A):
    L_tilde = scaled_Laplacian(A)
    cheb_polynomials = [torch.from_numpy(i).type(torch.DoubleTensor) for i in cheb_polynomial(L_tilde, K)]
    return ASTGCN(blocks, traffic_features, K, gcn_filters, t_filters, cheb_polynomials, pred_window, hours, days, weeks, vertices)
