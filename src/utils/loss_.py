import torch
from torch import nn
import torch.nn.functional as F

class VicLoss(nn.modules.loss._Loss):
    def __init__(self, λ: float = 25., μ: float = 25., ν: float = 1.,
                 γ: float = 1., ϵ: float = 1e-4):
        super(VicLoss,self).__init__()
        self.lambd = λ
        self.mu = μ
        self.nu = ν
        self.gamma = γ
        self.eps = ϵ

    def _off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
    def loss_fn(self, z1, z2, λ, μ, ν, γ, ϵ):
        # Get batch size and dim of rep
        N,D = z1.shape
            
        # invariance loss
        sim_loss = F.mse_loss(z1, z2)
            
        # variance loss
        std_z1 = torch.sqrt(z1.var(dim=0) + ϵ)
        std_z2 = torch.sqrt(z2.var(dim=0) + ϵ)
        std_loss = torch.relu(γ - std_z1).mean() / 2  + torch.relu(γ - std_z2).mean() / 2

        # covariance loss
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)
        cov_z1 = (z1.T @ z1) / (N-1)
        cov_z2 = (z2.T @ z2) / (N-1)
        cov_loss = (self._off_diagonal(cov_z1).pow_(2).sum() + self._off_diagonal(cov_z2).pow_(2).sum()) / D

        return λ*sim_loss + μ*std_loss + ν*cov_loss, (sim_loss,std_loss,cov_loss)

    def forward(self,z1,z2):
        return self.loss_fn(z1, z2, self.lambd, self.mu, self.nu, self.gamma, self.eps)

class TemporalVicLoss(VicLoss):
    def __init__(self, λ: float = 25., μ: float = 25., ν: float = 1.,
                 γ: float = 1., ϵ: float = 1e-4, t: float = 1.):
        super(TemporalVicLoss,self).__init__(λ, μ, ν, γ, ϵ)
        self.t = t

    def timeloss(self,t1,t2,diff):
        return self.t*F.mse_loss(abs(t1-t2), diff)

    def forward(self,z1,z2,t1,t2,diff):
        if self.lambd or self.mu or self.nu:
            cssl_loss, ind_loss = self.loss_fn(z1, z2, self.lambd, self.mu, self.nu, self.gamma, self.eps) #calculate vicreg loss when we actually use them
        else:
            cssl_loss, ind_loss = 0, (0,)
        time_loss = self.timeloss(t1,t2,diff)

        ind_loss = ind_loss + (time_loss,)
        return  cssl_loss+time_loss, ind_loss

class TINCLoss(VicLoss):
    def __init__(self, λ: float = 25., μ: float = 25., ν: float = 1.,
                 γ: float = 1., ϵ: float = 1e-4, insensitive = "one"):
        super(TINCLoss,self).__init__(λ, μ, ν, γ, ϵ)
        self.insensitive = insensitive

    def loss_fn(self, z1, z2, λ, μ, ν, γ, ϵ, time_label):
        # Get batch size and dim of rep
        N,D = z1.shape
            
        # invariance loss
        # original insensitive;
        # margin = F.mae_loss(z1, z2, reduction='none').mean(axis=1)-time_label
        margin = F.mse_loss(z1, z2, reduction='none').mean(axis=1)-time_label #mean is the across the representation dimension
        #margin = ((z1-z2)**2).mean(axis=1)-time_label
        if self.insensitive == "two": margin = margin**2
        sim_loss = F.relu(margin).mean() # Reduction needs to be done because margin is specific to each example
            
        # variance loss
        std_z1 = torch.sqrt(z1.var(dim=0) + ϵ)
        std_z2 = torch.sqrt(z2.var(dim=0) + ϵ)
        std_loss = torch.relu(γ - std_z1).mean() / 2  + torch.relu(γ - std_z2).mean() / 2
            
        # covariance loss
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)
        cov_z1 = (z1.T @ z1) / (N-1)
        cov_z2 = (z2.T @ z2) / (N-1)
        cov_loss = (self._off_diagonal(cov_z1).pow_(2).sum() + self._off_diagonal(cov_z2).pow_(2).sum()) / D

        return λ*sim_loss + μ*std_loss + ν*cov_loss, (sim_loss,std_loss,cov_loss)

    def forward(self,z1,z2,time_label):
        cssl_loss, ind_loss = self.loss_fn(z1, z2, self.lambd, self.mu, self.nu, self.gamma, self.eps, time_label) #calculate vicreg loss when we actually use them

        return  cssl_loss, ind_loss

class NtXent(nn.modules.loss._Loss):
    def __init__(self,temperature, return_logits=False):
        super(NtXent, self).__init__()
        self.temperature = temperature
        self.INF = 1e8
        self.return_logits = return_logits

    def forward(self, z_i, z_j):
        N = len(z_i)
        z_i = F.normalize(z_i, p=2, dim=-1) # dim [N, D]
        z_j = F.normalize(z_j, p=2, dim=-1) # dim [N, D]
        sim_zii= (z_i @ z_i.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zij = (z_i @ z_j.T) / self.temperature # dim [N, N] => the diag contains the correct pairs (i,j) (x transforms via T_i and T_j)
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)
        correct_pairs = torch.arange(N, device=z_i.device).long()
        loss_i = F.cross_entropy(torch.cat([sim_zij, sim_zii], dim=1), correct_pairs)
        loss_j = F.cross_entropy(torch.cat([sim_zij.T, sim_zjj], dim=1), correct_pairs)

        if self.return_logits:
            return (loss_i + loss_j), sim_zij, correct_pairs

        return (loss_i + loss_j)

class BarlowLoss(nn.modules.loss._Loss):
    def __init__(self, λ: float = 0.0051):
        super(BarlowLoss,self).__init__()
        self.lambd = λ

    def _off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
    def loss_fn(self, z1, z2):
        # cross-correlation matrix
        c = (z1.T @ z2) / z1.shape[0]

        on_diag = torch.diagonal(c).add_(-1.0).pow_(2).sum()
        off_diag = self._off_diagonal(c).pow_(2).sum()
            
        # finall loss
        loss = on_diag + self.lambd * off_diag

        return loss, (on_diag,off_diag)

    def forward(self,z1,z2):
        return self.loss_fn(z1, z2)

class TemporalBarlowLoss(BarlowLoss):
    # this is a prototype
    def __init__(self, λ: float = 0.0051, t: float = 0.1):
        super(TemporalVicLoss,self).__init__(λ)
        self.t = t

    def timeloss(self,t1,diff):
        return self.t*F.mse_loss(t1, diff)

    def forward(self,z1,z2,t1,diff):
        cssl_loss, ind_loss = self.loss_fn(z1, z2, self.lambd, self.mu, self.nu, self.gamma, self.eps) #calculate vicreg loss when we actually use them
        time_loss = self.timeloss(t1,diff)

        ind_loss = ind_loss + (time_loss,)
        return  cssl_loss+time_loss, ind_loss
