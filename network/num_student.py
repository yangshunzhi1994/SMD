from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class KL_divergence(nn.Module):
    def __init__(self, temperature = 1):
        super(KL_divergence, self).__init__()
        self.T = temperature
        self.Cls_crit = nn.CrossEntropyLoss().cuda()
    def forward(self, teacher_logit, student_logit, targets, weights1, weights2, n_test):
        
        for i in range(n_test):
            CE_loss = self.Cls_crit(student_logit[i].unsqueeze(0), targets[i].unsqueeze(0))
            KD_loss = - torch.sum(F.softmax(teacher_logit[i]/self.T,dim=0) * F.log_softmax(student_logit[i]/self.T,dim=0), 0, keepdim=False)* self.T * self.T
            loss = weights1[i]*CE_loss + weights2[i]*KD_loss
            if i==0:
                CE_KD_Loss = torch.unsqueeze(loss, 0)
            else:
                loss = torch.unsqueeze(loss, 0)
                CE_KD_Loss = torch.cat((CE_KD_Loss,loss),0)
        return CE_KD_Loss  

class Calculate_dynamic_weights(nn.Module):
    def __init__(self):
        super(Calculate_dynamic_weights, self).__init__()
        
    def forward(self, outputs, targets, n_test, threshold):
        for i in range(n_test):
            if i==0:
                weights = torch.unsqueeze(outputs[i][targets[i]]/threshold, 0)
            else:
                w = torch.unsqueeze(outputs[i][targets[i]]/threshold, 0)
                weights = torch.cat((weights,w),0)
        return 1-0.8*weights, 0.8*weights
    
    
class Threshold_weights2(nn.Module):
    def __init__(self, temperature = 2):
        super(Threshold_weights2, self).__init__()
        self.Softmax = nn.Softmax(dim=0)
        self.T = temperature   
    def forward(self, outputs1,outputs2,mimic,targets,n_test):
        
        for i in range(n_test):
            d1, _ = torch.sort(outputs1[i], descending=True)
            if outputs1[i][targets[i]] == d1[0]:
                d1 = d1[0] - d1[1]
            else:
                d1 = torch.zeros(1).cuda().squeeze(0)
                
            d2, _ = torch.sort(outputs2[i], descending=True)
            if outputs2[i][targets[i]] == d2[0]:
                d2 = d2[0] - d2[1]
            else:
                d2 = torch.zeros(1).cuda().squeeze(0)
                
            m, _ = torch.sort(mimic[i], descending=True)
            if mimic[i][targets[i]] == m[0]:
                m = m[0] - m[1]
            else:
                m = torch.zeros(1).cuda().squeeze(0)
                
            preds = torch.cat((d1.unsqueeze(0),d2.unsqueeze(0),m.unsqueeze(0)),0)
            preds = self.Softmax(preds/self.T)
            if i==0:
                out_threshold = preds.unsqueeze(0)
            else:
                out_threshold = torch.cat((out_threshold,preds.unsqueeze(0)),0)
        
        max_preds = torch.cat((outputs1.unsqueeze(0), outputs2.unsqueeze(0)),0)
        max_preds = torch.max(max_preds)
        return max_preds, out_threshold
    
class Dynamic_MultiTeacher2(nn.Module):
    def __init__(self):
        super(Dynamic_MultiTeacher2, self).__init__()
        self.calculate_dynamic_weights = Calculate_dynamic_weights().cuda()
        self.threshold_weights = Threshold_weights2().cuda()
        
        self.CE_KD = KL_divergence(temperature = 20).cuda()
        
    def forward(self, outputs1,outputs2,out_s,targets):
        
        n_test = targets.shape[0] 
        mimic = (outputs1+outputs2)/2
        max_preds, out_threshold = self.threshold_weights(outputs1,outputs2,mimic,targets,n_test)
        
        weights11, weights12 = self.calculate_dynamic_weights(outputs1, targets, n_test, max_preds)
        loss1 = self.CE_KD(outputs1, out_s, targets, weights11, weights12, n_test)
        weights21, weights22 = self.calculate_dynamic_weights(outputs2, targets, n_test, max_preds)
        loss2 = self.CE_KD(outputs2, out_s, targets, weights21, weights22, n_test)
        
        weightsM1, weightsM2 = self.calculate_dynamic_weights(mimic, targets, n_test, max_preds)
        lossM = self.CE_KD(mimic, out_s, targets, weightsM1, weightsM2, n_test)
        
        for i in range(n_test):
            N_loss = out_threshold[i][0]*loss1[i] + out_threshold[i][1]*loss2[i] + out_threshold[i][2]*lossM[i] 
            if i==0:
                loss = N_loss.unsqueeze(0)
            else:
                loss = torch.cat((loss,N_loss.unsqueeze(0)),0)
        return loss.mean()



    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
class Threshold_weights3(nn.Module):
    def __init__(self, temperature = 2):
        super(Threshold_weights3, self).__init__()
        self.Softmax = nn.Softmax(dim=0)
        self.T = temperature   
    def forward(self, outputs1,outputs2,outputs3,mimic,targets,n_test):
        
        for i in range(n_test):
            d1, _ = torch.sort(outputs1[i], descending=True)
            if outputs1[i][targets[i]] == d1[0]:
                d1 = d1[0] - d1[1]
            else:
                d1 = torch.zeros(1).cuda().squeeze(0)
                
            d2, _ = torch.sort(outputs2[i], descending=True)
            if outputs2[i][targets[i]] == d2[0]:
                d2 = d2[0] - d2[1]
            else:
                d2 = torch.zeros(1).cuda().squeeze(0)
                
            d3, _ = torch.sort(outputs3[i], descending=True)
            if outputs3[i][targets[i]] == d3[0]:
                d3 = d3[0] - d3[1]
            else:
                d3 = torch.zeros(1).cuda().squeeze(0)
                
            m, _ = torch.sort(mimic[i], descending=True)
            if mimic[i][targets[i]] == m[0]:
                m = m[0] - m[1]
            else:
                m = torch.zeros(1).cuda().squeeze(0)
                
            preds = torch.cat((d1.unsqueeze(0),d2.unsqueeze(0),d3.unsqueeze(0),m.unsqueeze(0)),0)
            preds = self.Softmax(preds/self.T)
            if i==0:
                out_threshold = preds.unsqueeze(0)
            else:
                out_threshold = torch.cat((out_threshold,preds.unsqueeze(0)),0)
        
        max_preds = torch.cat((outputs1.unsqueeze(0), outputs2.unsqueeze(0), outputs3.unsqueeze(0)),0)
        max_preds = torch.max(max_preds)
        return max_preds, out_threshold
    
class Dynamic_MultiTeacher3(nn.Module):
    def __init__(self):
        super(Dynamic_MultiTeacher3, self).__init__()
        self.calculate_dynamic_weights = Calculate_dynamic_weights().cuda()
        self.threshold_weights = Threshold_weights3().cuda()
        
        self.CE_KD = KL_divergence(temperature = 20).cuda()
        
    def forward(self, outputs1,outputs2,outputs3,out_s,targets):
        
        n_test = targets.shape[0] 
        mimic = (outputs1+outputs2+outputs3)/3
        max_preds, out_threshold = self.threshold_weights(outputs1,outputs2,outputs3,mimic,targets,n_test)
        
        weights11, weights12 = self.calculate_dynamic_weights(outputs1, targets, n_test, max_preds)
        loss1 = self.CE_KD(outputs1, out_s, targets, weights11, weights12, n_test)
        weights21, weights22 = self.calculate_dynamic_weights(outputs2, targets, n_test, max_preds)
        loss2 = self.CE_KD(outputs2, out_s, targets, weights21, weights22, n_test)
        weights31, weights32 = self.calculate_dynamic_weights(outputs3, targets, n_test, max_preds)
        loss3 = self.CE_KD(outputs3, out_s, targets, weights31, weights32, n_test)
        
        weightsM1, weightsM2 = self.calculate_dynamic_weights(mimic, targets, n_test, max_preds)
        lossM = self.CE_KD(mimic, out_s, targets, weightsM1, weightsM2, n_test)
        
        for i in range(n_test):
            N_loss = out_threshold[i][0]*loss1[i] + out_threshold[i][1]*loss2[i] + out_threshold[i][2]*loss3[i] + out_threshold[i][3]*lossM[i] 
            if i==0:
                loss = N_loss.unsqueeze(0)
            else:
                loss = torch.cat((loss,N_loss.unsqueeze(0)),0)
        return loss.mean()
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
class Threshold_weights4(nn.Module):
    def __init__(self, temperature = 2):
        super(Threshold_weights4, self).__init__()
        self.Softmax = nn.Softmax(dim=0)
        self.T = temperature   
    def forward(self, outputs1,outputs2,outputs3,outputs4,mimic,targets,n_test):
        
        for i in range(n_test):
            d1, _ = torch.sort(outputs1[i], descending=True)
            if outputs1[i][targets[i]] == d1[0]:
                d1 = d1[0] - d1[1]
            else:
                d1 = torch.zeros(1).cuda().squeeze(0)
                
            d2, _ = torch.sort(outputs2[i], descending=True)
            if outputs2[i][targets[i]] == d2[0]:
                d2 = d2[0] - d2[1]
            else:
                d2 = torch.zeros(1).cuda().squeeze(0)
                
            d3, _ = torch.sort(outputs3[i], descending=True)
            if outputs3[i][targets[i]] == d3[0]:
                d3 = d3[0] - d3[1]
            else:
                d3 = torch.zeros(1).cuda().squeeze(0)
                
            d4, _ = torch.sort(outputs4[i], descending=True)
            if outputs4[i][targets[i]] == d4[0]:
                d4 = d4[0] - d4[1]
            else:
                d4 = torch.zeros(1).cuda().squeeze(0)
                
            m, _ = torch.sort(mimic[i], descending=True)
            if mimic[i][targets[i]] == m[0]:
                m = m[0] - m[1]
            else:
                m = torch.zeros(1).cuda().squeeze(0)
                
            preds = torch.cat((d1.unsqueeze(0),d2.unsqueeze(0),d3.unsqueeze(0),d4.unsqueeze(0),m.unsqueeze(0)),0)
            preds = self.Softmax(preds/self.T)
            if i==0:
                out_threshold = preds.unsqueeze(0)
            else:
                out_threshold = torch.cat((out_threshold,preds.unsqueeze(0)),0)
        
        max_preds = torch.cat((outputs1.unsqueeze(0), outputs2.unsqueeze(0), outputs3.unsqueeze(0), outputs4.unsqueeze(0)),0)
        max_preds = torch.max(max_preds)
        return max_preds, out_threshold
    
class Dynamic_MultiTeacher4(nn.Module):
    def __init__(self):
        super(Dynamic_MultiTeacher4, self).__init__()
        self.calculate_dynamic_weights = Calculate_dynamic_weights().cuda()
        self.threshold_weights = Threshold_weights4().cuda()
        
        self.CE_KD = KL_divergence(temperature = 20).cuda()
        
    def forward(self, outputs1,outputs2,outputs3,outputs4,out_s,targets):
        
        n_test = targets.shape[0] 
        mimic = (outputs1+outputs2+outputs3+outputs4)/4
        max_preds, out_threshold = self.threshold_weights(outputs1,outputs2,outputs3,outputs4,mimic,targets,n_test)
        
        weights11, weights12 = self.calculate_dynamic_weights(outputs1, targets, n_test, max_preds)
        loss1 = self.CE_KD(outputs1, out_s, targets, weights11, weights12, n_test)
        weights21, weights22 = self.calculate_dynamic_weights(outputs2, targets, n_test, max_preds)
        loss2 = self.CE_KD(outputs2, out_s, targets, weights21, weights22, n_test)
        weights31, weights32 = self.calculate_dynamic_weights(outputs3, targets, n_test, max_preds)
        loss3 = self.CE_KD(outputs3, out_s, targets, weights31, weights32, n_test)
        weights41, weights42 = self.calculate_dynamic_weights(outputs4, targets, n_test, max_preds)
        loss4 = self.CE_KD(outputs4, out_s, targets, weights41, weights42, n_test)
        
        weightsM1, weightsM2 = self.calculate_dynamic_weights(mimic, targets, n_test, max_preds)
        lossM = self.CE_KD(mimic, out_s, targets, weightsM1, weightsM2, n_test)
        
        for i in range(n_test):
            N_loss = out_threshold[i][0]*loss1[i] + out_threshold[i][1]*loss2[i] + out_threshold[i][2]*loss3[i] +  \
                    out_threshold[i][3]*loss4[i] + out_threshold[i][4]*lossM[i] 
            if i==0:
                loss = N_loss.unsqueeze(0)
            else:
                loss = torch.cat((loss,N_loss.unsqueeze(0)),0)
        return loss.mean()  
    
    
    
    
    
    
    
    
    
    
    
class Threshold_weights5(nn.Module):
    def __init__(self, temperature = 2):
        super(Threshold_weights5, self).__init__()
        self.Softmax = nn.Softmax(dim=0)
        self.T = temperature   
    def forward(self, outputs1,outputs2,outputs3,outputs4,outputs5,mimic,targets,n_test):
        
        for i in range(n_test):
            d1, _ = torch.sort(outputs1[i], descending=True)
            if outputs1[i][targets[i]] == d1[0]:
                d1 = d1[0] - d1[1]
            else:
                d1 = torch.zeros(1).cuda().squeeze(0)
                
            d2, _ = torch.sort(outputs2[i], descending=True)
            if outputs2[i][targets[i]] == d2[0]:
                d2 = d2[0] - d2[1]
            else:
                d2 = torch.zeros(1).cuda().squeeze(0)
                
            d3, _ = torch.sort(outputs3[i], descending=True)
            if outputs3[i][targets[i]] == d3[0]:
                d3 = d3[0] - d3[1]
            else:
                d3 = torch.zeros(1).cuda().squeeze(0)
                
            d4, _ = torch.sort(outputs4[i], descending=True)
            if outputs4[i][targets[i]] == d4[0]:
                d4 = d4[0] - d4[1]
            else:
                d4 = torch.zeros(1).cuda().squeeze(0)
                
            d5, _ = torch.sort(outputs5[i], descending=True)
            if outputs5[i][targets[i]] == d5[0]:
                d5 = d5[0] - d5[1]
            else:
                d5 = torch.zeros(1).cuda().squeeze(0)
                
            m, _ = torch.sort(mimic[i], descending=True)
            if mimic[i][targets[i]] == m[0]:
                m = m[0] - m[1]
            else:
                m = torch.zeros(1).cuda().squeeze(0)
                
            preds = torch.cat((d1.unsqueeze(0),d2.unsqueeze(0),d3.unsqueeze(0),d4.unsqueeze(0),d5.unsqueeze(0),m.unsqueeze(0)),0)
            preds = self.Softmax(preds/self.T)
            if i==0:
                out_threshold = preds.unsqueeze(0)
            else:
                out_threshold = torch.cat((out_threshold,preds.unsqueeze(0)),0)
        
        max_preds = torch.cat((outputs1.unsqueeze(0), outputs2.unsqueeze(0), outputs3.unsqueeze(0), \
                               outputs4.unsqueeze(0), outputs5.unsqueeze(0)),0)
        max_preds = torch.max(max_preds)
        return max_preds, out_threshold
    
class Dynamic_MultiTeacher5(nn.Module):
    def __init__(self):
        super(Dynamic_MultiTeacher5, self).__init__()
        self.calculate_dynamic_weights = Calculate_dynamic_weights().cuda()
        self.threshold_weights = Threshold_weights5().cuda()
        
        self.CE_KD = KL_divergence(temperature = 20).cuda()
        
    def forward(self, outputs1,outputs2,outputs3,outputs4,outputs5,out_s,targets):
        
        n_test = targets.shape[0] 
        mimic = (outputs1+outputs2+outputs3+outputs4+outputs5)/5
        max_preds, out_threshold = self.threshold_weights(outputs1,outputs2,outputs3,outputs4,outputs5,mimic,targets,n_test)
        
        weights11, weights12 = self.calculate_dynamic_weights(outputs1, targets, n_test, max_preds)
        loss1 = self.CE_KD(outputs1, out_s, targets, weights11, weights12, n_test)
        weights21, weights22 = self.calculate_dynamic_weights(outputs2, targets, n_test, max_preds)
        loss2 = self.CE_KD(outputs2, out_s, targets, weights21, weights22, n_test)
        weights31, weights32 = self.calculate_dynamic_weights(outputs3, targets, n_test, max_preds)
        loss3 = self.CE_KD(outputs3, out_s, targets, weights31, weights32, n_test)
        weights41, weights42 = self.calculate_dynamic_weights(outputs4, targets, n_test, max_preds)
        loss4 = self.CE_KD(outputs4, out_s, targets, weights41, weights42, n_test)
        weights51, weights52 = self.calculate_dynamic_weights(outputs5, targets, n_test, max_preds)
        loss5 = self.CE_KD(outputs5, out_s, targets, weights51, weights52, n_test)
        
        weightsM1, weightsM2 = self.calculate_dynamic_weights(mimic, targets, n_test, max_preds)
        lossM = self.CE_KD(mimic, out_s, targets, weightsM1, weightsM2, n_test)
        
        for i in range(n_test):
            N_loss = out_threshold[i][0]*loss1[i] + out_threshold[i][1]*loss2[i] + out_threshold[i][2]*loss3[i] +  \
                    out_threshold[i][3]*loss4[i] + out_threshold[i][4]*loss5[i] + out_threshold[i][5]*lossM[i] 
            if i==0:
                loss = N_loss.unsqueeze(0)
            else:
                loss = torch.cat((loss,N_loss.unsqueeze(0)),0)
        return loss.mean()
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
class Threshold_weights6(nn.Module):
    def __init__(self, temperature = 2):
        super(Threshold_weights6, self).__init__()
        self.Softmax = nn.Softmax(dim=0)
        self.T = temperature   
    def forward(self, outputs1,outputs2,outputs3,outputs4,outputs5,outputs6,mimic,targets,n_test):
        
        for i in range(n_test):
            d1, _ = torch.sort(outputs1[i], descending=True)
            if outputs1[i][targets[i]] == d1[0]:
                d1 = d1[0] - d1[1]
            else:
                d1 = torch.zeros(1).cuda().squeeze(0)
                
            d2, _ = torch.sort(outputs2[i], descending=True)
            if outputs2[i][targets[i]] == d2[0]:
                d2 = d2[0] - d2[1]
            else:
                d2 = torch.zeros(1).cuda().squeeze(0)
                
            d3, _ = torch.sort(outputs3[i], descending=True)
            if outputs3[i][targets[i]] == d3[0]:
                d3 = d3[0] - d3[1]
            else:
                d3 = torch.zeros(1).cuda().squeeze(0)
                
            d4, _ = torch.sort(outputs4[i], descending=True)
            if outputs4[i][targets[i]] == d4[0]:
                d4 = d4[0] - d4[1]
            else:
                d4 = torch.zeros(1).cuda().squeeze(0)
                
            d5, _ = torch.sort(outputs5[i], descending=True)
            if outputs5[i][targets[i]] == d5[0]:
                d5 = d5[0] - d5[1]
            else:
                d5 = torch.zeros(1).cuda().squeeze(0)
                
            d6, _ = torch.sort(outputs6[i], descending=True)
            if outputs6[i][targets[i]] == d6[0]:
                d6 = d6[0] - d6[1]
            else:
                d6 = torch.zeros(1).cuda().squeeze(0)
                
            m, _ = torch.sort(mimic[i], descending=True)
            if mimic[i][targets[i]] == m[0]:
                m = m[0] - m[1]
            else:
                m = torch.zeros(1).cuda().squeeze(0)
                
            preds = torch.cat((d1.unsqueeze(0),d2.unsqueeze(0),d3.unsqueeze(0),d4.unsqueeze(0),d5.unsqueeze(0), \
                               d6.unsqueeze(0),m.unsqueeze(0)),0)
            preds = self.Softmax(preds/self.T)
            if i==0:
                out_threshold = preds.unsqueeze(0)
            else:
                out_threshold = torch.cat((out_threshold,preds.unsqueeze(0)),0)
        
        max_preds = torch.cat((outputs1.unsqueeze(0), outputs2.unsqueeze(0), outputs3.unsqueeze(0),\
                               outputs4.unsqueeze(0), outputs5.unsqueeze(0), outputs6.unsqueeze(0)),0)
        max_preds = torch.max(max_preds)
        return max_preds, out_threshold
    
class Dynamic_MultiTeacher6(nn.Module):
    def __init__(self):
        super(Dynamic_MultiTeacher6, self).__init__()
        self.calculate_dynamic_weights = Calculate_dynamic_weights().cuda()
        self.threshold_weights = Threshold_weights6().cuda()
        
        self.CE_KD = KL_divergence(temperature = 20).cuda()
        
    def forward(self, outputs1,outputs2,outputs3,outputs4,outputs5,outputs6,out_s,targets):
        
        n_test = targets.shape[0] 
        mimic = (outputs1+outputs2+outputs3+outputs4+outputs5+outputs6)/6
        max_preds, out_threshold = self.threshold_weights(outputs1,outputs2,outputs3,outputs4,outputs5,outputs6,mimic,targets,n_test)
        
        weights11, weights12 = self.calculate_dynamic_weights(outputs1, targets, n_test, max_preds)
        loss1 = self.CE_KD(outputs1, out_s, targets, weights11, weights12, n_test)
        weights21, weights22 = self.calculate_dynamic_weights(outputs2, targets, n_test, max_preds)
        loss2 = self.CE_KD(outputs2, out_s, targets, weights21, weights22, n_test)
        weights31, weights32 = self.calculate_dynamic_weights(outputs3, targets, n_test, max_preds)
        loss3 = self.CE_KD(outputs3, out_s, targets, weights31, weights32, n_test)
        weights41, weights42 = self.calculate_dynamic_weights(outputs4, targets, n_test, max_preds)
        loss4 = self.CE_KD(outputs4, out_s, targets, weights41, weights42, n_test)
        weights51, weights52 = self.calculate_dynamic_weights(outputs5, targets, n_test, max_preds)
        loss5 = self.CE_KD(outputs5, out_s, targets, weights51, weights52, n_test)
        weights61, weights62 = self.calculate_dynamic_weights(outputs6, targets, n_test, max_preds)
        loss6 = self.CE_KD(outputs6, out_s, targets, weights61, weights62, n_test)
        
        weightsM1, weightsM2 = self.calculate_dynamic_weights(mimic, targets, n_test, max_preds)
        lossM = self.CE_KD(mimic, out_s, targets, weightsM1, weightsM2, n_test)
        
        for i in range(n_test):
            N_loss = out_threshold[i][0]*loss1[i] + out_threshold[i][1]*loss2[i] + out_threshold[i][2]*loss3[i] +  \
                    out_threshold[i][3]*loss4[i] + out_threshold[i][4]*loss5[i] + out_threshold[i][5]*loss6[i] + \
                    out_threshold[i][6]*lossM[i] 
            if i==0:
                loss = N_loss.unsqueeze(0)
            else:
                loss = torch.cat((loss,N_loss.unsqueeze(0)),0)
        return loss.mean() 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
class Threshold_weights7(nn.Module):
    def __init__(self, temperature = 2):
        super(Threshold_weights7, self).__init__()
        self.Softmax = nn.Softmax(dim=0)
        self.T = temperature   
    def forward(self, outputs1,outputs2,outputs3,outputs4,outputs5,outputs6,outputs7,mimic,targets,n_test):
        
        for i in range(n_test):
            d1, _ = torch.sort(outputs1[i], descending=True)
            if outputs1[i][targets[i]] == d1[0]:
                d1 = d1[0] - d1[1]
            else:
                d1 = torch.zeros(1).cuda().squeeze(0)
                
            d2, _ = torch.sort(outputs2[i], descending=True)
            if outputs2[i][targets[i]] == d2[0]:
                d2 = d2[0] - d2[1]
            else:
                d2 = torch.zeros(1).cuda().squeeze(0)
                
            d3, _ = torch.sort(outputs3[i], descending=True)
            if outputs3[i][targets[i]] == d3[0]:
                d3 = d3[0] - d3[1]
            else:
                d3 = torch.zeros(1).cuda().squeeze(0)
                
            d4, _ = torch.sort(outputs4[i], descending=True)
            if outputs4[i][targets[i]] == d4[0]:
                d4 = d4[0] - d4[1]
            else:
                d4 = torch.zeros(1).cuda().squeeze(0)
                
            d5, _ = torch.sort(outputs5[i], descending=True)
            if outputs5[i][targets[i]] == d5[0]:
                d5 = d5[0] - d5[1]
            else:
                d5 = torch.zeros(1).cuda().squeeze(0)
                
            d6, _ = torch.sort(outputs6[i], descending=True)
            if outputs6[i][targets[i]] == d6[0]:
                d6 = d6[0] - d6[1]
            else:
                d6 = torch.zeros(1).cuda().squeeze(0)
                
            d7, _ = torch.sort(outputs7[i], descending=True)
            if outputs7[i][targets[i]] == d7[0]:
                d7 = d7[0] - d7[1]
            else:
                d7 = torch.zeros(1).cuda().squeeze(0)
                
            m, _ = torch.sort(mimic[i], descending=True)
            if mimic[i][targets[i]] == m[0]:
                m = m[0] - m[1]
            else:
                m = torch.zeros(1).cuda().squeeze(0)
                
            preds = torch.cat((d1.unsqueeze(0),d2.unsqueeze(0),d3.unsqueeze(0),d4.unsqueeze(0),d5.unsqueeze(0), \
                               d6.unsqueeze(0),d7.unsqueeze(0),m.unsqueeze(0)),0)
            preds = self.Softmax(preds/self.T)
            if i==0:
                out_threshold = preds.unsqueeze(0)
            else:
                out_threshold = torch.cat((out_threshold,preds.unsqueeze(0)),0)
        
        max_preds = torch.cat((outputs1.unsqueeze(0), outputs2.unsqueeze(0), outputs3.unsqueeze(0),\
                               outputs4.unsqueeze(0), outputs5.unsqueeze(0), outputs6.unsqueeze(0), outputs7.unsqueeze(0)),0)
        max_preds = torch.max(max_preds)
        return max_preds, out_threshold
    
class Dynamic_MultiTeacher7(nn.Module):
    def __init__(self):
        super(Dynamic_MultiTeacher7, self).__init__()
        self.calculate_dynamic_weights = Calculate_dynamic_weights().cuda()
        self.threshold_weights = Threshold_weights7().cuda()
        
        self.CE_KD = KL_divergence(temperature = 20).cuda()
        
    def forward(self, outputs1,outputs2,outputs3,outputs4,outputs5,outputs6,outputs7,out_s,targets):
        
        n_test = targets.shape[0] 
        mimic = (outputs1+outputs2+outputs3+outputs4+outputs5+outputs6+outputs7)/7
        max_preds, out_threshold = self.threshold_weights(outputs1,outputs2,outputs3,outputs4,outputs5,outputs6,outputs7,mimic,targets,n_test)
        
        weights11, weights12 = self.calculate_dynamic_weights(outputs1, targets, n_test, max_preds)
        loss1 = self.CE_KD(outputs1, out_s, targets, weights11, weights12, n_test)
        weights21, weights22 = self.calculate_dynamic_weights(outputs2, targets, n_test, max_preds)
        loss2 = self.CE_KD(outputs2, out_s, targets, weights21, weights22, n_test)
        weights31, weights32 = self.calculate_dynamic_weights(outputs3, targets, n_test, max_preds)
        loss3 = self.CE_KD(outputs3, out_s, targets, weights31, weights32, n_test)
        weights41, weights42 = self.calculate_dynamic_weights(outputs4, targets, n_test, max_preds)
        loss4 = self.CE_KD(outputs4, out_s, targets, weights41, weights42, n_test)
        weights51, weights52 = self.calculate_dynamic_weights(outputs5, targets, n_test, max_preds)
        loss5 = self.CE_KD(outputs5, out_s, targets, weights51, weights52, n_test)
        weights61, weights62 = self.calculate_dynamic_weights(outputs6, targets, n_test, max_preds)
        loss6 = self.CE_KD(outputs6, out_s, targets, weights61, weights62, n_test)
        weights71, weights72 = self.calculate_dynamic_weights(outputs7, targets, n_test, max_preds)
        loss7 = self.CE_KD(outputs7, out_s, targets, weights71, weights72, n_test)
        
        weightsM1, weightsM2 = self.calculate_dynamic_weights(mimic, targets, n_test, max_preds)
        lossM = self.CE_KD(mimic, out_s, targets, weightsM1, weightsM2, n_test)
        
        for i in range(n_test):
            N_loss = out_threshold[i][0]*loss1[i] + out_threshold[i][1]*loss2[i] + out_threshold[i][2]*loss3[i] +  \
                    out_threshold[i][3]*loss4[i] + out_threshold[i][4]*loss5[i] + out_threshold[i][5]*loss6[i] + \
                    out_threshold[i][6]*loss7[i] + out_threshold[i][7]*lossM[i] 
            if i==0:
                loss = N_loss.unsqueeze(0)
            else:
                loss = torch.cat((loss,N_loss.unsqueeze(0)),0)
        return loss.mean() 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
class Threshold_weights8(nn.Module):
    def __init__(self, temperature = 2):
        super(Threshold_weights8, self).__init__()
        self.Softmax = nn.Softmax(dim=0)
        self.T = temperature   
    def forward(self, outputs1,outputs2,outputs3,outputs4,outputs5,outputs6,outputs7,outputs8,mimic,targets,n_test):
        
        for i in range(n_test):
            d1, _ = torch.sort(outputs1[i], descending=True)
            if outputs1[i][targets[i]] == d1[0]:
                d1 = d1[0] - d1[1]
            else:
                d1 = torch.zeros(1).cuda().squeeze(0)
                
            d2, _ = torch.sort(outputs2[i], descending=True)
            if outputs2[i][targets[i]] == d2[0]:
                d2 = d2[0] - d2[1]
            else:
                d2 = torch.zeros(1).cuda().squeeze(0)
                
            d3, _ = torch.sort(outputs3[i], descending=True)
            if outputs3[i][targets[i]] == d3[0]:
                d3 = d3[0] - d3[1]
            else:
                d3 = torch.zeros(1).cuda().squeeze(0)
                
            d4, _ = torch.sort(outputs4[i], descending=True)
            if outputs4[i][targets[i]] == d4[0]:
                d4 = d4[0] - d4[1]
            else:
                d4 = torch.zeros(1).cuda().squeeze(0)
                
            d5, _ = torch.sort(outputs5[i], descending=True)
            if outputs5[i][targets[i]] == d5[0]:
                d5 = d5[0] - d5[1]
            else:
                d5 = torch.zeros(1).cuda().squeeze(0)
                
            d6, _ = torch.sort(outputs6[i], descending=True)
            if outputs6[i][targets[i]] == d6[0]:
                d6 = d6[0] - d6[1]
            else:
                d6 = torch.zeros(1).cuda().squeeze(0)
                
            d7, _ = torch.sort(outputs7[i], descending=True)
            if outputs7[i][targets[i]] == d7[0]:
                d7 = d7[0] - d7[1]
            else:
                d7 = torch.zeros(1).cuda().squeeze(0)
                
            d8, _ = torch.sort(outputs8[i], descending=True)
            if outputs8[i][targets[i]] == d8[0]:
                d8 = d8[0] - d8[1]
            else:
                d8 = torch.zeros(1).cuda().squeeze(0)
                
            m, _ = torch.sort(mimic[i], descending=True)
            if mimic[i][targets[i]] == m[0]:
                m = m[0] - m[1]
            else:
                m = torch.zeros(1).cuda().squeeze(0)
                
            preds = torch.cat((d1.unsqueeze(0),d2.unsqueeze(0),d3.unsqueeze(0),d4.unsqueeze(0),d5.unsqueeze(0), \
                               d6.unsqueeze(0),d7.unsqueeze(0),d8.unsqueeze(0),m.unsqueeze(0)),0)
            preds = self.Softmax(preds/self.T)
            if i==0:
                out_threshold = preds.unsqueeze(0)
            else:
                out_threshold = torch.cat((out_threshold,preds.unsqueeze(0)),0)
        
        max_preds = torch.cat((outputs1.unsqueeze(0), outputs2.unsqueeze(0), outputs3.unsqueeze(0),outputs4.unsqueeze(0), \
                               outputs5.unsqueeze(0), outputs6.unsqueeze(0), outputs7.unsqueeze(0), outputs8.unsqueeze(0)),0)
        max_preds = torch.max(max_preds)
        return max_preds, out_threshold
    
class Dynamic_MultiTeacher8(nn.Module):
    def __init__(self):
        super(Dynamic_MultiTeacher8, self).__init__()
        self.calculate_dynamic_weights = Calculate_dynamic_weights().cuda()
        self.threshold_weights = Threshold_weights8().cuda()
        
        self.CE_KD = KL_divergence(temperature = 20).cuda()
        
    def forward(self, outputs1,outputs2,outputs3,outputs4,outputs5,outputs6,outputs7,outputs8,out_s,targets):
        
        n_test = targets.shape[0] 
        mimic = (outputs1+outputs2+outputs3+outputs4+outputs5+outputs6+outputs7+outputs8)/8
        max_preds, out_threshold = self.threshold_weights(outputs1,outputs2,outputs3,outputs4,outputs5,outputs6,\
                                                          outputs7,outputs8,mimic,targets,n_test)
        
        weights11, weights12 = self.calculate_dynamic_weights(outputs1, targets, n_test, max_preds)
        loss1 = self.CE_KD(outputs1, out_s, targets, weights11, weights12, n_test)
        weights21, weights22 = self.calculate_dynamic_weights(outputs2, targets, n_test, max_preds)
        loss2 = self.CE_KD(outputs2, out_s, targets, weights21, weights22, n_test)
        weights31, weights32 = self.calculate_dynamic_weights(outputs3, targets, n_test, max_preds)
        loss3 = self.CE_KD(outputs3, out_s, targets, weights31, weights32, n_test)
        weights41, weights42 = self.calculate_dynamic_weights(outputs4, targets, n_test, max_preds)
        loss4 = self.CE_KD(outputs4, out_s, targets, weights41, weights42, n_test)
        weights51, weights52 = self.calculate_dynamic_weights(outputs5, targets, n_test, max_preds)
        loss5 = self.CE_KD(outputs5, out_s, targets, weights51, weights52, n_test)
        weights61, weights62 = self.calculate_dynamic_weights(outputs6, targets, n_test, max_preds)
        loss6 = self.CE_KD(outputs6, out_s, targets, weights61, weights62, n_test)
        weights71, weights72 = self.calculate_dynamic_weights(outputs7, targets, n_test, max_preds)
        loss7 = self.CE_KD(outputs7, out_s, targets, weights71, weights72, n_test)
        weights81, weights82 = self.calculate_dynamic_weights(outputs8, targets, n_test, max_preds)
        loss8 = self.CE_KD(outputs8, out_s, targets, weights81, weights82, n_test)
        
        weightsM1, weightsM2 = self.calculate_dynamic_weights(mimic, targets, n_test, max_preds)
        lossM = self.CE_KD(mimic, out_s, targets, weightsM1, weightsM2, n_test)
        
        for i in range(n_test):
            N_loss = out_threshold[i][0]*loss1[i] + out_threshold[i][1]*loss2[i] + out_threshold[i][2]*loss3[i] +  \
                    out_threshold[i][3]*loss4[i] + out_threshold[i][4]*loss5[i] + out_threshold[i][5]*loss6[i] + \
                    out_threshold[i][6]*loss7[i] + out_threshold[i][7]*loss8[i] + out_threshold[i][8]*lossM[i] 
            if i==0:
                loss = N_loss.unsqueeze(0)
            else:
                loss = torch.cat((loss,N_loss.unsqueeze(0)),0)
        return loss.mean() 