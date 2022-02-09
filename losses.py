from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def adjust_lr(optimizer, epoch, args_lr):
	scale   = 0.1
	lr_list =  [args_lr] * 100
	lr_list += [args_lr*scale] * 50
	lr_list += [args_lr*scale*scale] * 50

	lr = lr_list[epoch-1]
	print ('Epoch: {}  lr: {:.3f}'.format(epoch, lr))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def confusion_matrix(preds, y, NUM_CLASSES=7):
    """ Returns confusion matrix """
    assert preds.shape[0] == y.shape[0], "1 dim of predictions and labels must be equal"
    rounded_preds = torch.argmax(preds,1)
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for i in range(rounded_preds.shape[0]):
        predicted_class = rounded_preds[i]
        correct_class = y[i]
        conf_mat[correct_class][predicted_class] += 1
    return conf_mat

class KL_divergence(nn.Module):
    def __init__(self, temperature = 1):
        super(KL_divergence, self).__init__()
        self.T = temperature
        self.Cls_crit = nn.CrossEntropyLoss(reduction='none').cuda()
    def forward(self, teacher_logit, student_logit, targets, weights1, weights2, n_test):
        
        CE_loss = self.Cls_crit(student_logit, targets)
#         KD_loss = - torch.sum(F.softmax(teacher_logit/self.T,dim=1) * F.log_softmax(student_logit/self.T,dim=1), 1, keepdim=False)* self.T * self.T
        KD_loss = nn.KLDivLoss(reduction='none')(F.log_softmax(student_logit/self.T,dim=1), F.softmax(teacher_logit/self.T,dim=1)) * self.T * self.T
        loss = weights1*CE_loss + weights2*KD_loss.sum(1)
        return loss

def KW_Variance(outputs1,outputs2,outputs3,outputs4,targets,KW_Variance):
    
    total = targets.size(0)
    
    outputs1 = torch.argmax(outputs1,1)
    outputs2 = torch.argmax(outputs2,1)
    outputs3 = torch.argmax(outputs3,1)
    outputs4 = torch.argmax(outputs4,1)
    outputs = torch.cat([outputs1.unsqueeze(0), outputs2.unsqueeze(0),outputs3.unsqueeze(0), outputs4.unsqueeze(0)], 0)
    outputs = torch.transpose(outputs,1,0)
    
    for i in range(0, total):
        Correct_number = outputs[i].eq(targets[i].data).cpu().sum() #  Correct number of teachers
        KW_Variance.append(Correct_number)
    return KW_Variance
    

class Diversity(nn.Module):
    def __init__(self, direction, variance, temperature = 20):
        super(Diversity, self).__init__()
        self.direction = direction
        self.variance = variance
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.T = temperature
        self.Softmax = nn.Softmax(dim=2)
        
    def forward(self, outputs1,outputs2,outputs3,outputs4,targets):
        
        preds = torch.cat((outputs1.unsqueeze(0), outputs2.unsqueeze(0), outputs3.unsqueeze(0), outputs4.unsqueeze(0)),0)
        n_models = len(preds)
        n_test = len(preds[0])
        
        d = 0
        for j in range(n_models):
            for k in range(j+1, n_models):
                vx = preds[j] - torch.mean(preds[j], dim=1, keepdim=True)
                vy = preds[k] - torch.mean(preds[k], dim=1, keepdim=True)
                pearson = self.cos(vx, vy)
                d = d + pearson  

        preds = self.Softmax(preds/self.T)      
        Var = torch.cat((torch.var(preds[:,:,0],dim=0).unsqueeze(1),torch.var(preds[:,:,1],dim=0).unsqueeze(1),
                         torch.var(preds[:,:,2],dim=0).unsqueeze(1),torch.var(preds[:,:,3],dim=0).unsqueeze(1),
                         torch.var(preds[:,:,4],dim=0).unsqueeze(1),torch.var(preds[:,:,5],dim=0).unsqueeze(1),
                         torch.var(preds[:,:,6],dim=0).unsqueeze(1)),1)
        loss = self.direction*d - self.variance*Var.sum(1)
        final_loss = torch.mean(loss)
        return final_loss
    
    
    

class Calculate_dynamic_weights(nn.Module):
    def __init__(self):
        super(Calculate_dynamic_weights, self).__init__()
        
    def forward(self, outputs, targets, n_test, threshold):
        
        weights = [outputs[i][targets[i]] for i in range(n_test)]
        weights = torch.as_tensor(weights).cuda()/threshold
        
        return 1-weights, weights
    
    
class Threshold_weights(nn.Module):
    def __init__(self, temperature = 2):
        super(Threshold_weights, self).__init__()
        self.Softmax = nn.Softmax(dim=1)
        self.T = temperature
        
    def forward(self, outputs1,outputs2,outputs3,outputs4,mimic,targets,n_test):
        
        d1, _ = torch.sort(outputs1, dim=1, descending=True)
        d1_targets = torch.as_tensor([outputs1[i][targets[i]] for i in range(n_test)]).cuda()
        d1 = torch.where(d1[:,0] == d1_targets, d1[:,0] - d1[:,1], torch.zeros(1).cuda())
        
        d2, _ = torch.sort(outputs2, dim=1, descending=True)
        d2_targets = torch.as_tensor([outputs2[i][targets[i]] for i in range(n_test)]).cuda()
        d2 = torch.where(d2[:,0] == d2_targets, d2[:,0] - d2[:,1], torch.zeros(1).cuda())
        
        d3, _ = torch.sort(outputs3, dim=1, descending=True)
        d3_targets = torch.as_tensor([outputs3[i][targets[i]] for i in range(n_test)]).cuda()
        d3 = torch.where(d3[:,0] == d3_targets, d3[:,0] - d3[:,1], torch.zeros(1).cuda())
        
        d4, _ = torch.sort(outputs4, dim=1, descending=True)
        d4_targets = torch.as_tensor([outputs4[i][targets[i]] for i in range(n_test)]).cuda()
        d4 = torch.where(d4[:,0] == d4_targets, d4[:,0] - d4[:,1], torch.zeros(1).cuda())
        
        d5, _ = torch.sort(mimic, dim=1, descending=True)
        d5_targets = torch.as_tensor([mimic[i][targets[i]] for i in range(n_test)]).cuda()
        d5 = torch.where(d5[:,0] == d5_targets, d5[:,0] - d5[:,1], torch.zeros(1).cuda())
        
        preds = torch.cat((d1.unsqueeze(1),d2.unsqueeze(1),d3.unsqueeze(1),d4.unsqueeze(1),d5.unsqueeze(1)),1)
        out_threshold = self.Softmax(preds/self.T)
        
        max_preds = torch.cat((outputs1.unsqueeze(0), outputs2.unsqueeze(0), outputs3.unsqueeze(0),outputs4.unsqueeze(0)),0)
        max_preds = torch.max(max_preds)
                
        return max_preds, out_threshold



class Dynamic_MultiTeacher(nn.Module):
    def __init__(self):
        super(Dynamic_MultiTeacher, self).__init__()
        self.calculate_dynamic_weights = Calculate_dynamic_weights().cuda()
        self.threshold_weights = Threshold_weights(temperature = 6).cuda()
        
        self.CE_KD = KL_divergence(temperature = 20).cuda()
        
    def forward(self, outputs1,outputs2,outputs3,outputs4,out_s,targets):
        
        n_test = targets.shape[0] 
        mimic = (outputs1+outputs2+outputs3+outputs4)/4
        
        Cweights1 = min([outputs1[i][targets[i]] for i in range(n_test)])
        Cweights2 = min([outputs2[i][targets[i]] for i in range(n_test)])
        Cweights3 = min([outputs3[i][targets[i]] for i in range(n_test)])
        Cweights4 = min([outputs4[i][targets[i]] for i in range(n_test)])
        C = min(Cweights1, Cweights2, Cweights3, Cweights4)
        if C < 0:
            C = -C*torch.ones(outputs1.shape).cuda()
            Coutputs1 = C + outputs1 + 0.00001
            Coutputs2 = C + outputs2 + 0.00001
            Coutputs3 = C + outputs3 + 0.00001
            Coutputs4 = C + outputs4 + 0.00001
            Cmimic = C + mimic + 0.00001
            
            max_preds, out_threshold = self.threshold_weights(Coutputs1,Coutputs2,Coutputs3,Coutputs4,Cmimic,targets,n_test)
            weights11, weights12 = self.calculate_dynamic_weights(Coutputs1, targets, n_test, max_preds)
            weights21, weights22 = self.calculate_dynamic_weights(Coutputs2, targets, n_test, max_preds)
            weights31, weights32 = self.calculate_dynamic_weights(Coutputs3, targets, n_test, max_preds)
            weights41, weights42 = self.calculate_dynamic_weights(Coutputs4, targets, n_test, max_preds)
            weights61, weights62 = self.calculate_dynamic_weights(Cmimic, targets, n_test, max_preds)
        else:
            max_preds, out_threshold = self.threshold_weights(outputs1,outputs2,outputs3,outputs4,mimic,targets,n_test)
            weights11, weights12 = self.calculate_dynamic_weights(outputs1, targets, n_test, max_preds)
            weights21, weights22 = self.calculate_dynamic_weights(outputs2, targets, n_test, max_preds)
            weights31, weights32 = self.calculate_dynamic_weights(outputs3, targets, n_test, max_preds)
            weights41, weights42 = self.calculate_dynamic_weights(outputs4, targets, n_test, max_preds)
            weights61, weights62 = self.calculate_dynamic_weights(mimic, targets, n_test, max_preds)
            
        loss1 = self.CE_KD(outputs1, out_s, targets, weights11, weights12, n_test)
        loss2 = self.CE_KD(outputs2, out_s, targets, weights21, weights22, n_test)
        loss3 = self.CE_KD(outputs3, out_s, targets, weights31, weights32, n_test)
        loss4 = self.CE_KD(outputs4, out_s, targets, weights41, weights42, n_test)
        loss6 = self.CE_KD(mimic, out_s, targets, weights61, weights62, n_test)
        
        loss = (out_threshold[:,0] * loss1 + out_threshold[:,1] * loss2 + out_threshold[:,2] * loss3 + out_threshold[:,3] * loss4 + out_threshold[:,4] * loss6).mean()
        
        return loss


