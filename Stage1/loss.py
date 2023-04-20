import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import *
import numpy

import torch.autograd
from solvers import *

# This function is modified from https://github.com/HobbitLong/SupContrast/blob/master/losses.py
class LossFunction(nn.Module):
    def __init__(self, init_w=10.0, init_b=-5.0, **kwargs): # No temp param
        super(LossFunction, self).__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))

    def forward(self, features):
        batch_size = features.shape[0]
        mask = torch.eye(batch_size, dtype=torch.float32).to(torch.device('cuda'))
        count = features.shape[1]
        feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        dot_feature  = F.cosine_similarity(feature.unsqueeze(-1),feature.unsqueeze(-1).transpose(0,2))
        torch.clamp(self.w, 1e-6)
        dot_feature = dot_feature * self.w + self.b # We add this from angle protocol loss. 
        logits_max, _ = torch.max(dot_feature, dim=1, keepdim=True)
        logits = dot_feature - logits_max.detach()
        mask = mask.repeat(count, count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * count).view(-1, 1).to(torch.device('cuda')),
            0
        )
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        loss = -(mask * log_prob).sum(1) / mask.sum(1)
        loss = loss.view(count, batch_size).mean()
        n         = batch_size * 2
        label     = torch.from_numpy(numpy.asarray(list(range(batch_size - 1,batch_size*2 - 1)) + list(range(0,batch_size)))).cuda()
        logits    = logits.flatten()[1:].view(n-1, n+1)[:,:-1].reshape(n, n-1)
        prec1, _  = accuracy(logits.detach().cpu(), label.detach().cpu(), topk=(1, 2)) # Compute the training acc
        
        return loss, prec1
    
def compute_kernel_new(X,Y,gamma=0.1):
    gamma = 1./float(gamma)
    distances = -gamma*(2-2.*torch.mm(X,Y.T))
    kernel = torch.exp(distances)
    return kernel

class MMCL_inv(nn.Module):
    def __init__(self, sigma=0.07, batch_size=256, anchor_count=2, C=1.0):
        super(MMCL_inv, self).__init__()
        self.sigma = sigma
        self.C = C
        
        nn = batch_size - 1
        bs = batch_size
        self.mask, self.logits_mask = self.get_mask(batch_size, anchor_count)
        self.eye = torch.eye(anchor_count*batch_size).cuda()
        
        self.pos_mask = self.mask[:bs, bs:].bool()
        neg_mask=(self.mask*self.logits_mask+1)%2; 
        self.neg_mask = neg_mask-self.eye
        self.neg_mask = self.neg_mask[:bs, bs:].bool()
        
        self.kmask = torch.ones(batch_size,).bool().cuda()
        self.kmask.requires_grad = False

        self.oneone = (torch.ones(bs, bs) + torch.eye(bs)*0.1).cuda()
        self.one_bs = torch.ones(batch_size, nn, 1).cuda()
        self.one = torch.ones(nn,).cuda()
        self.KMASK = self.get_kmask(bs)
        self.block = torch.zeros(bs,2*bs).bool().cuda(); self.block[:bs,:bs] = True
        self.block12 = torch.zeros(bs,2*bs).bool().cuda(); self.block12[:bs,bs:] = True
        self.no_diag = (1-torch.eye(bs)).bool().cuda()
        
    def get_kmask(self, bs):
        KMASK = torch.ones(bs, bs, bs).bool().cuda()
        for t in range(bs):
            KMASK[t,t,:] = False
            KMASK[t,:,t] = False
        return KMASK.detach()
        
    def get_mask(self, batch_size, anchor_count):
        mask = torch.eye(batch_size, dtype=torch.float32).cuda()
        
        mask = mask.repeat(anchor_count, anchor_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask
        return mask, logits_mask
        
    def forward(self, features, labels=None, mask=None):
        bs = features.shape[0]
        nn = bs - 1
        
        F = torch.cat(torch.unbind(features, dim=1), dim=0)
        K = compute_kernel_new(F[:nn+1], F, gamma=self.sigma)
        
        
        with torch.no_grad():
            KK = torch.masked_select(K.detach(), self.block).reshape(bs, bs)
        
            KK_d0 = KK*self.no_diag
            KXY = -KK_d0.unsqueeze(1).repeat(1,bs,1)
            KXY = KXY + KXY.transpose(2,1)                
            Delta = (self.oneone + KK).unsqueeze(0) + KXY
            
            DD = torch.masked_select(Delta, self.KMASK).reshape(bs, nn, nn)
            
            alpha_y, _ = torch.solve(2*self.one_bs, DD)
            alpha_y = alpha_y.squeeze(2)

            if self.C == -1:
                alpha_y = torch.relu(alpha_y).detach()
            else:
                alpha_y = torch.relu(alpha_y).clamp(min=0, max=self.C).detach()

            alpha_x = alpha_y.sum(1)
            
        Ks = torch.masked_select(K, self.block12).reshape(bs, bs)
        Kn = torch.masked_select(Ks.T, self.neg_mask).reshape(bs,nn).T
        
        pos_loss = (alpha_x*(Ks*self.pos_mask).sum(1)).mean()
        neg_loss = (alpha_y.T*Kn).sum()/bs
        loss = neg_loss - pos_loss

        return loss, 0.
