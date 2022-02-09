from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from torch.autograd import Variable
from sklearn.svm import OneClassSVM as ssvm

class EDM(nn.Module):
    def __init__(self):
        super(EDM, self).__init__()
        
        self.Softmax = nn.Softmax(dim=1)
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        
    def forward(self, outputs1,outputs2,outputs3,outputs4):
        
        outputs1 = self.Softmax(outputs1)
        outputs2 = self.Softmax(outputs2)
        outputs3 = self.Softmax(outputs3)
        outputs4 = self.Softmax(outputs4)
        
        preds = torch.cat((outputs1.unsqueeze(0), outputs2.unsqueeze(0), outputs3.unsqueeze(0),outputs4.unsqueeze(0)),0)
        n_models = len(preds)
        n_test = len(preds[0])
        for i in range(n_test):
            cur_preds = [preds[k][i] for k in range(n_models)]
            for j in range(len(cur_preds)):
                for k in range(j+1, len(cur_preds)):    
                    cost = self.cos(cur_preds[j],cur_preds[k]).unsqueeze(0)
                    if j ==0 and k == 1:
                        d = cost
                    else:
                        d = torch.cat((d,cost),0)
            d = torch.logsumexp(d, dim=0).unsqueeze(0)
            if i==0:
                loss = d
            else:
                loss = torch.cat((loss,d),0)
        final_loss = torch.mean(loss)
        return final_loss
    
    
class RIDE(nn.Module):
    def __init__(self): # We do not need to consider the test time, so Routing diversified experts are not included.
        super(RIDE, self).__init__()
        
        cls_num_list = [705, 717, 281, 4772, 1982, 1290, 2524] # RAF-DB Data distribution： 705 717 281 4772 1982 1290 2524 
        
        cls_num_list = np.array(cls_num_list) / np.sum(cls_num_list)
        C = len(cls_num_list)
        per_cls_weights = C * cls_num_list * 0.015 + 1 - 0.015
        per_cls_weights = per_cls_weights / np.max(per_cls_weights)
        per_cls_weights_enabled_diversity = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False).cuda()
        self.diversity_temperature = 1.5 * per_cls_weights_enabled_diversity.view((1, -1))
        
        
    def forward(self, outputs1,outputs2,outputs3,outputs4,targets):
        
        preds = torch.cat((outputs1.unsqueeze(0), outputs2.unsqueeze(0), outputs3.unsqueeze(0), outputs4.unsqueeze(0)),0)
        n_models = len(preds)
        n_test = len(preds[0])
        for i in range(n_test):
            cur_preds = [preds[k][i] for k in range(n_models)]
            cost = 0
            for j in range(1,len(cur_preds)):
                KD_loss = nn.KLDivLoss()(F.log_softmax(cur_preds[j].unsqueeze(0)/self.diversity_temperature,dim=1),\
                    F.softmax(cur_preds[0].unsqueeze(0)/self.diversity_temperature,dim=1))\
                               * self.diversity_temperature * self.diversity_temperature
                cost = cost + KD_loss.mean(1) 
            d = cost / 3
            if i==0:
                loss = d
            else:
                loss = torch.cat((loss,d),0)
        final_loss = torch.mean(loss)
        return final_loss
    
    
    
def USTE(inputs,targets):
    
    inputs1 = None
    inputs2 = None
    inputs3 = None
    inputs4 = None
    targets1 = None
    targets2 = None
    targets3 = None
    targets4 = None
    # Teacher1: 6,1,2
    # Teacher2: 0,1,3,5
    # Teacher3: 4,0,3,6
    # Teacher4: 2,4,5
    for i in range(len(targets)):
        
        ip = inputs[i].unsqueeze(0)
        tg = targets[i].unsqueeze(0)
        
        if targets[i]==0:# 2,3
            if inputs2 !=None:
                inputs2 = torch.cat((inputs2,ip),0)
                targets2 = torch.cat((targets2,tg),0)
            else:
                inputs2 = ip
                targets2 = tg
            if inputs3 !=None:
                inputs3 = torch.cat((inputs3,ip),0)
                targets3 = torch.cat((targets3,tg),0)
            else:
                inputs3 = ip
                targets3 = tg   
        elif targets[i]==1:# 1,2
            if inputs1 !=None:
                inputs1 = torch.cat((inputs1,ip),0)
                targets1 = torch.cat((targets1,tg),0)
            else:
                inputs1 = ip
                targets1 = tg
            if inputs2 !=None:
                inputs2 = torch.cat((inputs2,ip),0)
                targets2 = torch.cat((targets2,tg),0)
            else:
                inputs2 = ip
                targets2 = tg
                
        elif targets[i]==2:# 1,4
            if inputs1 !=None:
                inputs1 = torch.cat((inputs1,ip),0)
                targets1 = torch.cat((targets1,tg),0)
            else:
                inputs1 = ip
                targets1 = tg
            if inputs4 !=None:
                inputs4 = torch.cat((inputs4,ip),0)
                targets4 = torch.cat((targets4,tg),0)
            else:
                inputs4 = ip
                targets4 = tg
        elif targets[i]==3: # 2,3
            if inputs2 !=None:
                inputs2 = torch.cat((inputs2,ip),0)
                targets2 = torch.cat((targets2,tg),0)
            else:
                inputs2 = ip
                targets2 = tg
            if inputs3 !=None:
                inputs3 = torch.cat((inputs3,ip),0)
                targets3 = torch.cat((targets3,tg),0)
            else:
                inputs3 = ip
                targets3 = tg
        elif targets[i]==4: # 3,4
            if inputs3 !=None:
                inputs3 = torch.cat((inputs3,ip),0)
                targets3 = torch.cat((targets3,tg),0)
            else:
                inputs3 = ip
                targets3 = tg
            if inputs4 !=None:
                inputs4 = torch.cat((inputs4,ip),0)
                targets4 = torch.cat((targets4,tg),0)
            else:
                inputs4 = ip
                targets4 = tg
        elif targets[i]==5: # 2,4
            if inputs2 !=None:
                inputs2 = torch.cat((inputs2,ip),0)
                targets2 = torch.cat((targets2,tg),0)
            else:
                inputs2 = ip
                targets2 = tg
            if inputs4 !=None:
                inputs4 = torch.cat((inputs4,ip),0)
                targets4 = torch.cat((targets4,tg),0)
            else:
                inputs4 = ip
                targets4 = tg
        elif targets[i]==6: # 1,3
            if inputs1 !=None:
                inputs1 = torch.cat((inputs1,ip),0)
                targets1 = torch.cat((targets1,tg),0)
            else:
                inputs1 = ip
                targets1 = tg
            if inputs3 !=None:
                inputs3 = torch.cat((inputs3,ip),0)
                targets3 = torch.cat((targets3,tg),0)
            else:
                inputs3 = ip
                targets3 = tg
        else:
            raise Exception('Invalid ...')
            
    return inputs1,targets1,inputs2,targets2,inputs3,targets3,inputs4,targets4
     
    
class USTE_prediction(nn.Module):
    def __init__(self, temperature = 1):
        super(USTE_prediction, self).__init__()
        self.Softmax = nn.Softmax(dim=1)
        self.T = temperature
        
    def forward(self, outputs1,outputs2,outputs3,outputs4):
        outputs1 = self.Softmax(outputs1/self.T)
        outputs2 = self.Softmax(outputs2/self.T)
        outputs3 = self.Softmax(outputs3/self.T)
        outputs4 = self.Softmax(outputs4/self.T)
        
        outputs = (outputs1+outputs2+outputs3+outputs4)/4
        
        return outputs
    
class KL_divergence(nn.Module):
    def __init__(self, temperature = 1):
        super(KL_divergence, self).__init__()
        self.T = temperature
    def forward(self, teacher_logit, student_logit):

        KD_loss = nn.KLDivLoss()(F.log_softmax(student_logit/self.T,dim=1), F.softmax(teacher_logit/self.T,dim=1)) * self.T * self.T

        return KD_loss
    
    
    
class OKDDip(nn.Module):
    def __init__(self, in_dim=7,out_dim=7,temperature = 3):
        super(OKDDip, self).__init__()
        self.layerL = nn.Linear(in_dim, out_dim)
        self.layerE = nn.Linear(in_dim, out_dim)
        self.Softmax = nn.Softmax(dim=0)
        self.T = temperature
 
    def forward(self, outputs1,outputs2,outputs3,outputs4):
        weightsL1 = self.layerL(outputs1)
        weightsE1 = self.layerE(outputs1)
        weightsL2 = self.layerL(outputs2)
        weightsE2 = self.layerE(outputs2)
        weightsL3 = self.layerL(outputs3)
        weightsE3 = self.layerE(outputs3)
        
        weightsLs = self.layerL(outputs4)
        weightsEs = self.layerE(outputs4)
        
        w1 = torch.cat((torch.mm(weightsL1.t(),weightsE1).squeeze(0), torch.mm(weightsL1.t(),weightsE2).squeeze(0),  \
                        torch.mm(weightsL1.t(),weightsE3).squeeze(0)),0)
        Alpha1 = self.Softmax(w1)
        w2 = torch.cat((torch.mm(weightsL2.t(),weightsE1).squeeze(0), torch.mm(weightsL2.t(),weightsE2).squeeze(0),  \
                        torch.mm(weightsL2.t(),weightsE3).squeeze(0)),0)
        Alpha2 = self.Softmax(w2)
        w3 = torch.cat((torch.mm(weightsL3.t(),weightsE1).squeeze(0), torch.mm(weightsL3.t(),weightsE2).squeeze(0),  \
                        torch.mm(weightsL3.t(),weightsE3).squeeze(0)),0)
        Alpha3 = self.Softmax(w3)
        
        ws = torch.cat((torch.mm(weightsLs.t(),weightsE1).squeeze(0), torch.mm(weightsLs.t(),weightsE2).squeeze(0),  \
                        torch.mm(weightsLs.t(),weightsE3).squeeze(0)),0)
        AlphaS = self.Softmax(ws)
        
        t1 = Alpha1[0]*self.Softmax(outputs1/self.T)+Alpha1[1]*self.Softmax(outputs2/self.T)+ \
                         Alpha1[2]*self.Softmax(outputs3/self.T)
        t2 = Alpha2[0]*self.Softmax(outputs1/self.T)+Alpha2[1]*self.Softmax(outputs2/self.T)+ \
                         Alpha2[2]*self.Softmax(outputs3/self.T)+Alpha2[3]
        t3 = Alpha3[0]*self.Softmax(outputs1/self.T)+Alpha3[1]*self.Softmax(outputs2/self.T)+ \
                         Alpha3[2]*self.Softmax(outputs3/self.T)
        
        tS = AlphaS[0]*self.Softmax(outputs1/self.T)+AlphaS[1]*self.Softmax(outputs2/self.T)+ \
                         AlphaS[2]*self.Softmax(outputs3/self.T)
        
        L1 = nn.KLDivLoss()(F.log_softmax(outputs1/self.T,dim=1), F.softmax(t1,dim=1)) * self.T * self.T
        L2 = nn.KLDivLoss()(F.log_softmax(outputs2/self.T,dim=1), F.softmax(t2,dim=1)) * self.T * self.T
        L3 = nn.KLDivLoss()(F.log_softmax(outputs3/self.T,dim=1), F.softmax(t3,dim=1)) * self.T * self.T
        
        LS = nn.KLDivLoss()(F.log_softmax(outputs4/self.T,dim=1), F.softmax(tS,dim=1)) * self.T * self.T
        
        loss = L1+L2+L3+ LS
        return loss
    
    
class OKDDip_Student(nn.Module):
    def __init__(self, in_dim=7,out_dim=7,temperature = 3):
        super(OKDDip_Student, self).__init__()
        self.layerL = nn.Linear(in_dim, out_dim)
        self.layerE = nn.Linear(in_dim, out_dim)
        self.Softmax = nn.Softmax(dim=0)
        self.T = temperature
 
    def forward(self, outputs1,outputs2,outputs3,outputs4,out_s):
        weightsL1 = self.layerL(outputs1)
        weightsE1 = self.layerE(outputs1)
        weightsL2 = self.layerL(outputs2)
        weightsE2 = self.layerE(outputs2)
        weightsL3 = self.layerL(outputs3)
        weightsE3 = self.layerE(outputs3)
        weightsL4 = self.layerL(outputs4)
        weightsE4 = self.layerE(outputs4)
        
        weightsLs = self.layerL(out_s)
        weightsEs = self.layerE(out_s)
        
        w1 = torch.cat((torch.mm(weightsL1.t(),weightsE1).squeeze(0), torch.mm(weightsL1.t(),weightsE2).squeeze(0),  \
                        torch.mm(weightsL1.t(),weightsE3).squeeze(0),torch.mm(weightsL1.t(),weightsE4).squeeze(0)),0)
        Alpha1 = self.Softmax(w1)
        w2 = torch.cat((torch.mm(weightsL2.t(),weightsE1).squeeze(0), torch.mm(weightsL2.t(),weightsE2).squeeze(0),  \
                        torch.mm(weightsL2.t(),weightsE3).squeeze(0),torch.mm(weightsL2.t(),weightsE4).squeeze(0)),0)
        Alpha2 = self.Softmax(w2)
        w3 = torch.cat((torch.mm(weightsL3.t(),weightsE1).squeeze(0), torch.mm(weightsL3.t(),weightsE2).squeeze(0),  \
                        torch.mm(weightsL3.t(),weightsE3).squeeze(0),torch.mm(weightsL3.t(),weightsE4).squeeze(0)),0)
        Alpha3 = self.Softmax(w3)
        w4 = torch.cat((torch.mm(weightsL4.t(),weightsE1).squeeze(0), torch.mm(weightsL4.t(),weightsE2).squeeze(0),  \
                        torch.mm(weightsL4.t(),weightsE3).squeeze(0),torch.mm(weightsL4.t(),weightsE4).squeeze(0)),0)
        Alpha4 = self.Softmax(w4)
        
        ws = torch.cat((torch.mm(weightsLs.t(),weightsE1).squeeze(0), torch.mm(weightsLs.t(),weightsE2).squeeze(0),  \
                        torch.mm(weightsLs.t(),weightsE3).squeeze(0),torch.mm(weightsLs.t(),weightsE4).squeeze(0)),0)
        AlphaS = self.Softmax(ws)
        
        t1 = Alpha1[0]*self.Softmax(outputs1/self.T)+Alpha1[1]*self.Softmax(outputs2/self.T)+ \
                         Alpha1[2]*self.Softmax(outputs3/self.T)+Alpha1[3]*self.Softmax(outputs4/self.T)
        t2 = Alpha2[0]*self.Softmax(outputs1/self.T)+Alpha2[1]*self.Softmax(outputs2/self.T)+ \
                         Alpha2[2]*self.Softmax(outputs3/self.T)+Alpha2[3]*self.Softmax(outputs4/self.T)
        t3 = Alpha3[0]*self.Softmax(outputs1/self.T)+Alpha3[1]*self.Softmax(outputs2/self.T)+ \
                         Alpha3[2]*self.Softmax(outputs3/self.T)+Alpha3[3]*self.Softmax(outputs4/self.T)
        t4 = Alpha4[0]*self.Softmax(outputs1/self.T)+Alpha4[1]*self.Softmax(outputs2/self.T)+ \
                         Alpha4[2]*self.Softmax(outputs3/self.T)+Alpha4[3]*self.Softmax(outputs4/self.T)
        
        tS = AlphaS[0]*self.Softmax(outputs1/self.T)+AlphaS[1]*self.Softmax(outputs2/self.T)+ \
                         AlphaS[2]*self.Softmax(outputs3/self.T)+AlphaS[3]*self.Softmax(outputs4/self.T)
        
        L1 = nn.KLDivLoss()(F.log_softmax(outputs1/self.T,dim=1), F.softmax(t1,dim=1)) * self.T * self.T
        L2 = nn.KLDivLoss()(F.log_softmax(outputs2/self.T,dim=1), F.softmax(t2,dim=1)) * self.T * self.T
        L3 = nn.KLDivLoss()(F.log_softmax(outputs3/self.T,dim=1), F.softmax(t3,dim=1)) * self.T * self.T
        L4 = nn.KLDivLoss()(F.log_softmax(outputs4/self.T,dim=1), F.softmax(t4,dim=1)) * self.T * self.T
        
        LS = nn.KLDivLoss()(F.log_softmax(out_s/self.T,dim=1), F.softmax(tS,dim=1)) * self.T * self.T
        
        loss = L1+L2+L3+L4 + LS
        return loss
    
    
class OKDDip_Online(nn.Module):
    def __init__(self, in_dim=7,out_dim=7,temperature = 3):
        super(OKDDip_Online, self).__init__()
        self.layerL = nn.Linear(in_dim, out_dim)
        self.layerE = nn.Linear(in_dim, out_dim)
        self.Softmax = nn.Softmax(dim=0)
        self.T = temperature
 
    def forward(self, outputs1,outputs2,outputs3,outputs4,out_s):
        weightsL1 = self.layerL(outputs1)
        weightsE1 = self.layerE(outputs1)
        weightsL2 = self.layerL(outputs2)
        weightsE2 = self.layerE(outputs2)
        weightsL3 = self.layerL(outputs3)
        weightsE3 = self.layerE(outputs3)
        weightsL4 = self.layerL(outputs4)
        weightsE4 = self.layerE(outputs4)
        
        weightsLs = self.layerL(out_s)
        weightsEs = self.layerE(out_s)
        
        w1 = torch.cat((torch.mm(weightsL1.t(),weightsE1).squeeze(0), torch.mm(weightsL1.t(),weightsE2).squeeze(0),  \
                        torch.mm(weightsL1.t(),weightsE3).squeeze(0),torch.mm(weightsL1.t(),weightsE4).squeeze(0)),0)
        Alpha1 = self.Softmax(w1)
        w2 = torch.cat((torch.mm(weightsL2.t(),weightsE1).squeeze(0), torch.mm(weightsL2.t(),weightsE2).squeeze(0),  \
                        torch.mm(weightsL2.t(),weightsE3).squeeze(0),torch.mm(weightsL2.t(),weightsE4).squeeze(0)),0)
        Alpha2 = self.Softmax(w2)
        w3 = torch.cat((torch.mm(weightsL3.t(),weightsE1).squeeze(0), torch.mm(weightsL3.t(),weightsE2).squeeze(0),  \
                        torch.mm(weightsL3.t(),weightsE3).squeeze(0),torch.mm(weightsL3.t(),weightsE4).squeeze(0)),0)
        Alpha3 = self.Softmax(w3)
        w4 = torch.cat((torch.mm(weightsL4.t(),weightsE1).squeeze(0), torch.mm(weightsL4.t(),weightsE2).squeeze(0),  \
                        torch.mm(weightsL4.t(),weightsE3).squeeze(0),torch.mm(weightsL4.t(),weightsE4).squeeze(0)),0)
        Alpha4 = self.Softmax(w4)
        
        ws = torch.cat((torch.mm(weightsLs.t(),weightsE1).squeeze(0), torch.mm(weightsLs.t(),weightsE2).squeeze(0),  \
                        torch.mm(weightsLs.t(),weightsE3).squeeze(0),torch.mm(weightsLs.t(),weightsE4).squeeze(0)),0)
        AlphaS = self.Softmax(ws)
        
        t1 = Alpha1[0]*self.Softmax(outputs1/self.T)+Alpha1[1]*self.Softmax(outputs2/self.T)+ \
                         Alpha1[2]*self.Softmax(outputs3/self.T)+Alpha1[3]*self.Softmax(outputs4/self.T)
        t2 = Alpha2[0]*self.Softmax(outputs1/self.T)+Alpha2[1]*self.Softmax(outputs2/self.T)+ \
                         Alpha2[2]*self.Softmax(outputs3/self.T)+Alpha2[3]*self.Softmax(outputs4/self.T)
        t3 = Alpha3[0]*self.Softmax(outputs1/self.T)+Alpha3[1]*self.Softmax(outputs2/self.T)+ \
                         Alpha3[2]*self.Softmax(outputs3/self.T)+Alpha3[3]*self.Softmax(outputs4/self.T)
        t4 = Alpha4[0]*self.Softmax(outputs1/self.T)+Alpha4[1]*self.Softmax(outputs2/self.T)+ \
                         Alpha4[2]*self.Softmax(outputs3/self.T)+Alpha4[3]*self.Softmax(outputs4/self.T)
        
        tS = AlphaS[0]*self.Softmax(outputs1/self.T)+AlphaS[1]*self.Softmax(outputs2/self.T)+ \
                         AlphaS[2]*self.Softmax(outputs3/self.T)+AlphaS[3]*self.Softmax(outputs4/self.T)
        
        L1 = nn.KLDivLoss()(F.log_softmax(outputs1/self.T,dim=1), F.softmax(t1,dim=1)) * self.T * self.T
        L2 = nn.KLDivLoss()(F.log_softmax(outputs2/self.T,dim=1), F.softmax(t2,dim=1)) * self.T * self.T
        L3 = nn.KLDivLoss()(F.log_softmax(outputs3/self.T,dim=1), F.softmax(t3,dim=1)) * self.T * self.T
        L4 = nn.KLDivLoss()(F.log_softmax(outputs4/self.T,dim=1), F.softmax(t4,dim=1)) * self.T * self.T
        
        LS = nn.KLDivLoss()(F.log_softmax(out_s/self.T,dim=1), F.softmax(tS,dim=1)) * self.T * self.T
        
        loss = L1+L2+L3+L4 + LS
        return loss
    
     
    
class KDCL(nn.Module): # KDCL-MinLogit,T=2,λ=1
    def __init__(self):
        super(KDCL, self).__init__()
        
        self.KD = KL_divergence(temperature = 2).cuda()
 
    def forward(self, outputs1,outputs2,outputs3,outputs4,targets):
        
        n_test = len(outputs1)
        outputs = None
        for i in range(n_test):
            out1 = outputs1[i]-outputs1[i][targets[i]]
            out2 = outputs2[i]-outputs2[i][targets[i]]
            out3 = outputs3[i]-outputs3[i][targets[i]]
            out4 = outputs4[i]-outputs4[i][targets[i]]
            out = torch.cat((out1.unsqueeze(0),out2.unsqueeze(0),out3.unsqueeze(0),out4.unsqueeze(0)),0)
            out,_ = torch.min(out,dim=0)
            if outputs != None:
                outputs = torch.cat((outputs,out.unsqueeze(0)),0)
            else:
                outputs = out.unsqueeze(0)
                
        loss = self.KD(outputs,outputs1)+self.KD(outputs,outputs2)+self.KD(outputs,outputs3)+self.KD(outputs,outputs4)
        
        return loss,outputs
    
    
def find_optimal_svm(vecs, method="ssvm-precomputed", nu=-1, gpu_id=0, is_norm=True):
    m = vecs.shape[0]
    vec_tmp = vecs.reshape(vecs.shape[0], vecs.shape[1], -1)
    vec_mean = torch.mean(vec_tmp, dim=1)
    vec_norm = vec_mean.norm(dim=1, keepdim=True)
    if is_norm:
        vec_mean = vec_mean * (1 / vec_norm)
    G = torch.matmul(vec_mean, vec_mean.permute(1, 0))

    if nu == -1:
        nu = 1 / m
    elif nu > 1:
        nu = 1
    elif nu < 1 / m:
        nu = 1 / m
    ret = np.zeros(m)

    if method == "ssvm-precomputed":
        if G.is_cuda:
            G_cpu = G.cpu()
        else:
            G_cpu = G
        G_cpu = G_cpu.detach().numpy()
        svm = ssvm(kernel="precomputed", nu=nu, tol=1e-6)
        svm.fit(G_cpu)
    else:
        raise NotImplementedError

    if is_norm:
        if vec_norm.is_cuda:
            vec_norm = vec_norm.cpu()
        vec_norm = vec_norm.squeeze().detach().numpy()

    ret[svm.support_] = svm.dual_coef_ / (m * nu)
    if is_norm:
        ret_normalize = ret * (1 / vec_norm)
        ret_normalize = ret_normalize / np.sum(ret_normalize)
        ret_final = torch.from_numpy(ret_normalize).float()
    else:
        ret_final = torch.from_numpy(ret).float()

    return ret_final

    
class AEKD(nn.Module): 
    def __init__(self):
        super(AEKD, self).__init__()
        
        self.KD = KL_divergence(temperature = 4).cuda()
 
    def forward(self, out_t1,out_t2,out_t3,out_t4, out_s):
        
        grads = []
        out_s.register_hook(lambda grad: grads.append(
            Variable(grad.data.clone(), requires_grad=False)))
        loss1 = self.KD(out_t1, out_s)
        loss1.backward(retain_graph=True)
        loss2 = self.KD(out_t2, out_s)
        loss2.backward(retain_graph=True)
        loss3 = self.KD(out_t3, out_s)
        loss3.backward(retain_graph=True)
        loss4 = self.KD(out_t4, out_s)
        loss4.backward(retain_graph=True)
        loss_div_list = torch.cat((loss1.unsqueeze(0),loss2.unsqueeze(0),loss3.unsqueeze(0),loss4.unsqueeze(0)),0)
        
        nu = 1 / (0.6 * 4)
        scale = find_optimal_svm(torch.stack(grads),nu=nu,gpu_id=0,is_norm=False)
        loss_div = torch.dot(scale.cuda(), loss_div_list.cuda())
        
        return loss_div
    
    
class AEKD_Teacher(nn.Module):
    def __init__(self):
        super(AEKD_Teacher, self).__init__()
        
    def forward(self, outputs1,outputs2,outputs3,outputs4):
        
        preds = torch.cat((outputs1.unsqueeze(0), outputs2.unsqueeze(0), outputs3.unsqueeze(0),outputs4.unsqueeze(0)),0)
        argmax_preds = preds.argmax(2)
        
        n_models = len(argmax_preds)
        n_test = len(argmax_preds[0])
        
        for i in range(n_test):
            cur_preds = [argmax_preds[k][i].cpu() for k in range(n_models)]
            classes, counts = np.unique(cur_preds, return_counts=True)
            
            if (counts == max(counts)).sum() > 1:
                score = (preds[0][i]+preds[1][i]+preds[2][i]+preds[3][i])/4
                score = torch.unsqueeze(score, 0)
                if i==0:
                    final_preds = score
                else:
                    final_preds = torch.cat((final_preds,score),0)
            else:
                maxlabel = [i for i,a in enumerate(cur_preds) if a==max(cur_preds,key=cur_preds.count)]
                number = np.random.randint(0, len(maxlabel), size=None, dtype='l')
                score = preds[maxlabel[number]][i]
                score = torch.unsqueeze(score, 0)
                if i==0:
                    final_preds = score
                else:
                    final_preds = torch.cat((final_preds,score),0)
        return final_preds 

    
def del_tensor_ele(arr,index):
        arr1 = arr[0:index]
        arr2 = arr[index+1:]
        return torch.cat((arr1,arr2),dim=0)
    
class AED(nn.Module): 
    def __init__(self):
        super(AED, self).__init__()
        self.T = 3
        self.Cls_crit = nn.CrossEntropyLoss().cuda()
 
    def forward(self, outputs1,outputs2,outputs3,outputs4,targets):
        n_test = targets.shape[0] 
        mimic = (outputs1+outputs2+outputs3+outputs4)/4
        for i in range(n_test):
            t1 = 0.8*outputs1[i] + 0.2*mimic[i]
            CE_loss1 = self.Cls_crit(outputs1[i].unsqueeze(0), targets[i].unsqueeze(0))
            KD_loss1 = nn.KLDivLoss()(F.log_softmax(outputs1[i]/self.T,dim=0),\
                                     F.softmax(t1/self.T,dim=0)) * self.T * self.T
            
            t2 = 0.8*outputs2[i] + 0.2*mimic[i]
            CE_loss2 = self.Cls_crit(outputs2[i].unsqueeze(0), targets[i].unsqueeze(0))
            KD_loss2 = nn.KLDivLoss()(F.log_softmax(outputs2[i]/self.T,dim=0),\
                                     F.softmax(t2/self.T,dim=0)) * self.T * self.T
            
            t3 = 0.8*outputs3[i] + 0.2*mimic[i]
            CE_loss3 = self.Cls_crit(outputs3[i].unsqueeze(0), targets[i].unsqueeze(0))
            KD_loss3 = nn.KLDivLoss()(F.log_softmax(outputs3[i]/self.T,dim=0),\
                                     F.softmax(t3/self.T,dim=0)) * self.T * self.T
            
            t4 = 0.8*outputs4[i] + 0.2*mimic[i]
            CE_loss4 = self.Cls_crit(outputs4[i].unsqueeze(0), targets[i].unsqueeze(0))
            KD_loss4 = nn.KLDivLoss()(F.log_softmax(outputs4[i]/self.T,dim=0),\
                                     F.softmax(t4/self.T,dim=0)) * self.T * self.T
            
            Loss_IRM = CE_loss1+KD_loss1+CE_loss2+KD_loss2+CE_loss3+KD_loss3+CE_loss4+KD_loss4
            
            min_CE_num = torch.argmin(torch.cat((CE_loss1.unsqueeze(0),CE_loss2.unsqueeze(0),CE_loss3.unsqueeze(0), \
                                             CE_loss4.unsqueeze(0)),0),dim=0)
            if min_CE_num ==0:
                min_CE = CE_loss1
            elif min_CE_num ==1:
                min_CE = CE_loss2
            elif min_CE_num ==2:
                min_CE = CE_loss3
            elif min_CE_num ==3:
                min_CE = CE_loss4
            else:
                raise Exception('Invalid ...')
            
            out = torch.cat((del_tensor_ele(outputs1[i],targets[i]).unsqueeze(0),del_tensor_ele(outputs2[i],targets[i]).unsqueeze(0),\
                         del_tensor_ele(outputs3[i],targets[i]).unsqueeze(0),del_tensor_ele(outputs4[i],targets[i]).unsqueeze(0)),0)
            Loss_DAL = 0
            for j in range(len(out)):
                for k in range(j+1, len(out)):
                    cost = torch.sum(out[j] * out[k])/(torch.sqrt(torch.sum(out[j] ** 2)) * torch.sqrt(torch.sum(out[k] ** 2)))
                    Loss_DAL = Loss_DAL + cost
            
            L = Loss_IRM + 0.7*min_CE+0.3*Loss_DAL
            if i==0:
                loss = L.unsqueeze(0)
            else:
                loss = torch.cat((loss,L.unsqueeze(0)),0)
        return torch.mean(loss), mimic
    
    
class FFMCD(nn.Module): 
    def __init__(self):
        super(FFMCD, self).__init__()
        self.T = 3
        self.Cls_crit = nn.CrossEntropyLoss().cuda()
        self.Softmax = nn.Softmax(dim=2)
 
    def forward(self, outputs1,outputs2,outputs3,outputs4,targets):
        
        CL1 = self.Cls_crit(outputs1,targets)
        CL2 = self.Cls_crit(outputs2,targets)
        CL3 = self.Cls_crit(outputs3,targets)
        CL4 = self.Cls_crit(outputs4,targets)
        
        outputs = torch.cat((outputs1.unsqueeze(0),outputs2.unsqueeze(0),outputs3.unsqueeze(0),outputs4.unsqueeze(0)),0)
        outputs = self.Softmax(outputs)
        outputs = outputs[0]*outputs1+outputs[1]*outputs2+outputs[2]*outputs3+outputs[3]*outputs4
        
        KD_loss1 = nn.KLDivLoss()(F.log_softmax(outputs1/self.T,dim=1),\
                                     F.softmax(outputs/self.T,dim=1)) * self.T * self.T
        KD_loss2 = nn.KLDivLoss()(F.log_softmax(outputs2/self.T,dim=1),\
                                     F.softmax(outputs/self.T,dim=1)) * self.T * self.T
        KD_loss3 = nn.KLDivLoss()(F.log_softmax(outputs3/self.T,dim=1),\
                                     F.softmax(outputs/self.T,dim=1)) * self.T * self.T
        KD_loss4 = nn.KLDivLoss()(F.log_softmax(outputs4/self.T,dim=1),\
                                     F.softmax(outputs/self.T,dim=1)) * self.T * self.T
        loss = CL1 + CL2 + CL3 + CL4 + KD_loss1 + KD_loss2 + KD_loss3 + KD_loss4
        
        return loss    
    
    
class CD(nn.Module): 
    def __init__(self):
        super(CD, self).__init__()
 
    def forward(self, net1,net2,net3,net4):
        
        for name,parameters in net1.named_parameters():
            if name == 'fc.weight':
                parm1 =parameters.detach()
        for name,parameters in net2.named_parameters():
            if name == 'fc.weight':
                parm2 =parameters.detach()        
        for name,parameters in net3.named_parameters():
            if name == 'fc.weight':
                parm3 =parameters.detach()
        for name,parameters in net4.named_parameters():
            if name == 'fc.weight':
                parm4 =parameters.detach()
                
        cd1= torch.norm(torch.mm(parm1.t(),parm2))
        cd2= torch.norm(torch.mm(parm1.t(),parm3))
        cd3= torch.norm(torch.mm(parm1.t(),parm4))
        cd4= torch.norm(torch.mm(parm2.t(),parm3))
        cd5= torch.norm(torch.mm(parm2.t(),parm4))
        cd6= torch.norm(torch.mm(parm3.t(),parm4))
        loss_CD = cd1+cd2+cd3+cd4+cd5+cd6
        return loss_CD   
    
    
class CTR(nn.Module): 
    def __init__(self, dim=7):
        super(CTR, self).__init__()
        self.layer1 = nn.Linear(dim, dim)
        self.layer2 = nn.Linear(dim, dim)
        self.layer3 = nn.Linear(dim, dim)
        self.layer4 = nn.Linear(dim, dim)
        self.Softmax = nn.Softmax(dim=0)
 
    def forward(self, outputs1,outputs2,outputs3,outputs4):
        T1 = self.layer1(outputs1)
        T2 = self.layer2(outputs2)
        T3 = self.layer3(outputs3)
        T4 = self.layer4(outputs4)
        outputs = torch.cat((T1.unsqueeze(0),T2.unsqueeze(0),T3.unsqueeze(0),T4.unsqueeze(0)),0)
        outputs = self.Softmax(outputs)
        mimic = outputs[0]*outputs1 + outputs[1]*outputs2 + outputs[2]*outputs3 + outputs[3]*outputs4
        return mimic   
    
    
class ENDD(nn.Module): # https://github.com/lennelov/endd-reproduce/blob/master/code/utils/losses.py
    def __init__(self, epsilon=1e-8, ensemble_epsilon=1e-3):
        super(ENDD, self).__init__()
        self.temp=2.5
        self.smooth_val = epsilon
        self.tp_scaling = 1 - ensemble_epsilon
        self.Softmax = nn.Softmax(dim=2)
 
    def forward(self, outputs1,outputs2,outputs3,outputs4,logits):
        ensemble_logits = torch.cat((outputs1.unsqueeze(1),outputs2.unsqueeze(1),outputs3.unsqueeze(1),outputs4.unsqueeze(1)),1)
        alphas = torch.exp(logits / self.temp)
        precision = torch.sum(alphas, axis=1)  #sum over classes
        ensemble_probs = self.Softmax(ensemble_logits / self.temp)  #softmax over classes
        # Smooth for num. stability:
        probs_mean = 1 / ensemble_probs.shape[2]  #divide by nr of classes
        # Subtract mean, scale down, add mean back)
        ensemble_probs = self.tp_scaling * (ensemble_probs - probs_mean) + probs_mean
        
        log_ensemble_probs_geo_mean = torch.sum(torch.log(ensemble_probs + self.smooth_val),
                                                  axis=1)  #mean over ensembles
        
        target_independent_term = torch.sum(torch.lgamma(alphas + self.smooth_val), axis=1) - torch.lgamma(
            precision + self.smooth_val)  #sum over lgammma of classes - lgamma(precision)
        
        target_dependent_term = -torch.sum(
            (alphas - 1.) * log_ensemble_probs_geo_mean, axis=1)  # -sum over classes
        
        cost = target_dependent_term + target_independent_term
        
        loss = torch.sum(cost) * (self.temp**2)  #mean of all batches
        return loss    
    
    
    
class FewShot(nn.Module): # https://github.com/dvornikita/fewshot_ensemble
    def __init__(self):
        super(FewShot, self).__init__()
        self.T=1
 
    def forward(self, outputs1,outputs2,outputs3,outputs4,labels):
        logits = torch.cat((outputs1.unsqueeze(2),outputs2.unsqueeze(2),outputs3.unsqueeze(2),outputs4.unsqueeze(2)),2)
        batch_size, n_cats, n_heads = logits.shape
        all_probs = torch.softmax(logits / self.T, dim=1) 
        label_inds = torch.ones(batch_size, n_cats).cuda()
        label_inds[range(batch_size), labels] = 0
        
        # removing the gt prob
        probs = all_probs * label_inds.unsqueeze(-1).detach()
        # re-normalize such that probs sum to 1
        probs /= (all_probs.sum(dim=1, keepdim=True) + 1e-8)
        
        probs = probs / torch.sqrt(((
            all_probs ** 2).sum(dim=1, keepdim=True) + 1e-8))    # l2 normed
        cov_mat = torch.einsum('ijk,ijl->ikl', probs, probs)
        pairwise_inds = 1 - torch.eye(n_heads).cuda()
        den = batch_size * (n_heads - 1) * n_heads
        loss = (cov_mat * pairwise_inds).sum() / den
        
        return loss 
    
def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res
    

class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        # N x C
        # N x N x C

        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
        return loss


class RkdDistance(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d>0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d>0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
        return loss
    
    
class AMTML_KD(nn.Module): #https://github.com/FLHonker/AMTML-KD-code/blob/master/LR_adaptive_learning.ipynb
    def __init__(self):
        super(AMTML_KD, self).__init__()
        self.angle_criterion = RKdAngle().cuda()
        self.dist_criterion = RkdDistance().cuda()
 
    def forward(self, y, labels, weighted_logits, T, alpha):
        kd_loss = nn.KLDivLoss()(F.log_softmax(y/T,dim=1), F.softmax(weighted_logits/T,dim=1)) * T * T * alpha + F.cross_entropy(y, labels)
        angle_loss = self.angle_criterion(y, weighted_logits)
        dist_loss = self.dist_criterion(y, weighted_logits)
        loss = kd_loss + angle_loss + dist_loss
        return loss       
    
    
    
class EKD(nn.Module): 
    def __init__(self):
        super(EKD, self).__init__()
        self.T = 10
        self.MSE = nn.MSELoss().cuda()
 
    def forward(self, out_t1,out_t2,out_t3,out_t4,mimic,out_s):
        
        KD_loss = nn.KLDivLoss()(F.log_softmax(out_s,dim=1), F.softmax(mimic/self.T,dim=1)) * self.T * self.T
        KD_loss1 = nn.KLDivLoss()(F.log_softmax(out_s,dim=1), F.softmax(out_t1/self.T,dim=1)) * self.T * self.T
        KD_loss2 = nn.KLDivLoss()(F.log_softmax(out_s,dim=1), F.softmax(out_t2/self.T,dim=1)) * self.T * self.T
        KD_loss3 = nn.KLDivLoss()(F.log_softmax(out_s,dim=1), F.softmax(out_t3/self.T,dim=1)) * self.T * self.T
        KD_loss4 = nn.KLDivLoss()(F.log_softmax(out_s,dim=1), F.softmax(out_t4/self.T,dim=1)) * self.T * self.T
        
        MSE = self.MSE(out_s,mimic)
        MSE1 = self.MSE(out_s,out_t1)
        MSE2 = self.MSE(out_s,out_t2)
        MSE3 = self.MSE(out_s,out_t3)
        MSE4 = self.MSE(out_s,out_t4)
        
        loss = KD_loss + KD_loss1 + KD_loss2 + KD_loss3 + KD_loss4 + MSE + MSE1 + MSE2 + MSE3 + MSE4
        
        return loss 
    
class EKD_Teacher(nn.Module): 
    def __init__(self):
        super(EKD_Teacher, self).__init__()
        self.T = 10
        self.MSE = nn.MSELoss().cuda()
 
    def forward(self, out_t1,out_t2,out_t3,out_t4,mimic):
        
        KD_loss1 = nn.KLDivLoss()(F.log_softmax(mimic/self.T,dim=1), F.softmax(out_t1,dim=1)) * self.T * self.T
        KD_loss2 = nn.KLDivLoss()(F.log_softmax(mimic/self.T,dim=1), F.softmax(out_t2,dim=1)) * self.T * self.T
        KD_loss3 = nn.KLDivLoss()(F.log_softmax(mimic/self.T,dim=1), F.softmax(out_t3,dim=1)) * self.T * self.T
        KD_loss4 = nn.KLDivLoss()(F.log_softmax(mimic/self.T,dim=1), F.softmax(out_t4,dim=1)) * self.T * self.T
        
        MSE1 = self.MSE(mimic,out_t1)
        MSE2 = self.MSE(mimic,out_t2)
        MSE3 = self.MSE(mimic,out_t3)
        MSE4 = self.MSE(mimic,out_t4)
        
        loss = KD_loss1 + KD_loss2 + KD_loss3 + KD_loss4 + MSE1 + MSE2 + MSE3 + MSE4
        
        return loss 
    
    
class PCL(nn.Module): 
    def __init__(self,out_dim):
        super(PCL, self).__init__()
        self.layer = nn.Linear(4*272, out_dim)
 
    def forward(self, mimic_t1,mimic_t2,mimic_t3,mimic_t4):
        mimic = torch.cat((mimic_t1.unsqueeze(2), mimic_t2.unsqueeze(2), mimic_t3.unsqueeze(2), mimic_t4.unsqueeze(2)),2).view(-1,4*272)
        mimic = self.layer(mimic)
        return mimic     
    

    
def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))  
    
    
    
class ONE(nn.Module):
    def __init__(self, in_dim=7,out_dim=7):
        super(ONE, self).__init__()
        self.layer = nn.Linear(in_dim, out_dim)
        self.norm = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)
 
    def forward(self, outputs1,outputs2,outputs3,outputs4):
        
        outputs1 = self.relu(self.norm(self.layer(outputs1)))
        outputs2 = self.relu(self.norm(self.layer(outputs2)))
        outputs3 = self.relu(self.norm(self.layer(outputs3)))
        outputs4 = self.relu(self.norm(self.layer(outputs4)))
        
        mimic = torch.cat((outputs1.unsqueeze(0), outputs2.unsqueeze(0), outputs3.unsqueeze(0), outputs4.unsqueeze(0)),0)
        g = F.softmax(mimic,dim=0)
        mimic = g[0]*outputs1 + g[1]*outputs2 + g[2]*outputs3 + g[3]*outputs4
        
        return mimic    
    
    
    
    
    
def colorferet_USTE(inputs,targets):
    
    inputs1 = None
    inputs2 = None
    inputs3 = None
    inputs4 = None
    targets1 = None
    targets2 = None
    targets3 = None
    targets4 = None
    
    for i in range(len(targets)):
        
        ip = inputs[i].unsqueeze(0)
        tg = targets[i].unsqueeze(0)
        if targets[i] <= 248:
            if inputs1 !=None:
                inputs1 = torch.cat((inputs1,ip),0)
                targets1 = torch.cat((targets1,tg),0)
            else:
                inputs1 = ip
                targets1 = tg   
        elif 248 < targets[i] <= 496:
            if inputs2 !=None:
                inputs2 = torch.cat((inputs2,ip),0)
                targets2 = torch.cat((targets2,tg),0)
            else:
                inputs2 = ip
                targets2 = tg  
        elif 496 < targets[i] <= 744:
            if inputs3 !=None:
                inputs3 = torch.cat((inputs3,ip),0)
                targets3 = torch.cat((targets3,tg),0)
            else:
                inputs3 = ip
                targets3 = tg
        else:
            if inputs4 !=None:
                inputs4 = torch.cat((inputs4,ip),0)
                targets4 = torch.cat((targets4,tg),0)
            else:
                inputs4 = ip
                targets4 = tg
            
    return inputs1,targets1,inputs2,targets2,inputs3,targets3,inputs4,targets4  
    
    
    
    
def PET_USTE(inputs,targets):
    
    inputs1 = None
    inputs2 = None
    inputs3 = None
    inputs4 = None
    targets1 = None
    targets2 = None
    targets3 = None
    targets4 = None
    
    for i in range(len(targets)):
        
        ip = inputs[i].unsqueeze(0)
        tg = targets[i].unsqueeze(0)
        if targets[i] <= 15:
            if inputs1 !=None:
                inputs1 = torch.cat((inputs1,ip),0)
                targets1 = torch.cat((targets1,tg),0)
            else:
                inputs1 = ip
                targets1 = tg   
        elif 10 < targets[i] <= 25:
            if inputs2 !=None:
                inputs2 = torch.cat((inputs2,ip),0)
                targets2 = torch.cat((targets2,tg),0)
            else:
                inputs2 = ip
                targets2 = tg  
        elif 20 < targets[i] <= 30:
            if inputs3 !=None:
                inputs3 = torch.cat((inputs3,ip),0)
                targets3 = torch.cat((targets3,tg),0)
            else:
                inputs3 = ip
                targets3 = tg
        elif 25 < targets[i] <= 37:
            if inputs4 !=None:
                inputs4 = torch.cat((inputs4,ip),0)
                targets4 = torch.cat((targets4,tg),0)
            else:
                inputs4 = ip
                targets4 = tg
                
        else:
            Exception('Invalid targets[i]...')
            
    return inputs1,targets1,inputs2,targets2,inputs3,targets3,inputs4,targets4     
    