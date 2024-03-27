from cProfile import label
from math import gamma
from tkinter.tix import Y_REGION
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# from mdistiller.mdistiller.distillers.SP import similarity_loss
class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t,target = None):
        p_s = F.log_softmax(y_s / self.T, dim=-1)
        p_t = F.softmax(y_t / self.T, dim=-1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T ** 2) / y_s.shape[0]
        return loss
    

        
class SWD(nn.Module):
    
    def __init__(self,
                 T : float = 4.0, 
                 alpha:float = 1.0,
                 beta: float = 1.0,
                 reduction: str = 'mean',
                 ignore_index: int = -100
                 ) -> None:
        """Constructor.

        Args:
            T (float, Optional): Temperature parameter for KL Divergence.
                Defaults to 4.
            alpha (float, optional): A constant, a parameter for KL Divergence.
                Defaults to 1.0.
            beta (float, optional): A constant, a parameter for Cross-entropy.
                Defaults to 1.0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean','sum','none'):
            raise ValueError(
                'Reduction must be one of :"mean", "sum", "none".')
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.kl_loss = nn.KLDivLoss(reduction="none")
        self.nll_loss = nn.NLLLoss(reduction='none', ignore_index=ignore_index)

        # self.sim = nn.CosineSimilarity()
    def __repr__(self):
        arg_keys = ['T','alpha','beta', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'
        
    def forward(self, y_s: Tensor, y_t: Tensor, target) -> Tensor:
        
        if y_s.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = y_s.shape[1]
            y_s = y_s.permute(0, *range(2, y_s.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            target = target.view(-1)
        unignored_mask = target != self.ignore_index
        target = target[unignored_mask]
        if len(target) == 0:
            return torch.tensor(0.)
        y_s = y_s[unignored_mask]
        
        # compute softmaxs 
        log_p_s = F.log_softmax(y_s / self.T, dim=-1)
        log_p = F.log_softmax(y_s,dim = -1)
        log_t = F.log_softmax(y_t,dim = -1)
        p_t_temp = F.softmax(y_t / self.T, dim=-1)
        # p_s = F.softmax(y_t,dim=-1)
        # p_t = F.softmax(y_t,dim=-1)
        
        # get true class column from each row
        all_rows = torch.arange(len(y_s))
        log_pt_s = log_p[all_rows, target]
        log_pt_t = log_t[all_rows, target]
        pt_s = log_pt_s.exp()
        pt_t = log_pt_t.exp()
        #compute similarity between the logits
        
        # Compute L2-normalized logits
        logits_s_normalized = F.normalize(y_s, p=2, dim=1)
        logits_t_normalized = F.normalize(y_t, p=2, dim=1)

        # Compute cosine similarity between normalized logits
        similarity = F.cosine_similarity(logits_s_normalized, logits_t_normalized, dim=-1)

        # compute the factor term
        factor = torch.exp(-self.beta * (1- pt_s ))
        # factor = torch.exp(-(self.beta/pt_s) * torch.abs(pt_s - 1))
        
        # compute the weight: [(1 - similarity) * exp(alpha * similarity) * factor]
        weight = (1 - torch.exp(-self.beta * torch.abs(pt_s - pt_t))) * (1 - similarity) * torch.exp(self.alpha * similarity) 
        
        # weight = (1 - torch.exp(-(self.beta/pt_t) * torch.abs(pt_s - pt_t))) * torch.exp((self.alpha * pt_s)/(similarity + self.beta))/self.beta 
        
        # CE loss
        # ce = self.ce_loss(y_s,target)
        ce = self.nll_loss(log_p, target)


         #compute KL Divergence
        kl = self.kl_loss(log_p_s,p_t_temp).sum(-1)*(self.T ** 2)
        
        # weighted losses
        weighted_ce = (1-factor) * ce
        weighted_kl = (weight*kl)* factor
        
        # the full loss: focal_term * kl_loss
        if self.reduction == 'mean':
            weighted_ce = weighted_ce.mean()
            weighted_kl = weighted_kl.mean()
        elif self.reduction == 'sum':
            weighted_ce = weighted_ce.sum()
            weighted_kl = weighted_kl.sum()
            
        loss = weighted_ce + weighted_kl
        return loss


class DKD(nn.Module):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, opt):
        super(DKD, self).__init__()
        self.alpha = opt.ALPHA
        self.beta = opt.BETA
        self.temperature = opt.kd_T

    def _get_gt_mask(self,logits, target):
        target = target.reshape(-1)
        mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
        return mask


    def _get_other_mask(self,logits, target):
        target = target.reshape(-1)
        mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
        return mask


    def cat_mask(self,t, mask1, mask2):
        t1 = (t * mask1).sum(dim=1, keepdims=True)
        t2 = (t * mask2).sum(1, keepdims=True)
        rt = torch.cat([t1, t2], dim=1)
        return rt
    
    def forward(self, logits_student, logits_teacher, target):
        
        gt_mask = self._get_gt_mask(logits_student, target)
        other_mask = self._get_other_mask(logits_student, target)
        pred_student = F.softmax(logits_student / self.temperature, dim=-1)
        pred_teacher = F.softmax(logits_teacher / self.temperature, dim=-1)
        pred_student = self.cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = self.cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student)
        tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, size_average=False)
            * (self.temperature**2)
            / target.shape[0]
        )
        pred_teacher_part2 = F.softmax(
            logits_teacher / self.temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / self.temperature - 1000.0 * gt_mask, dim=1
        )
        nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
            * (self.temperature**2)
            / target.shape[0]
        )
        loss_dkd = self.alpha * tckd_loss + self.beta * nckd_loss
        return loss_dkd

      