from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def KW_Variance2(outputs1,outputs2,targets,KW_Variance):
    
    total = targets.size(0)
    outputs1 = torch.argmax(outputs1,1)
    outputs2 = torch.argmax(outputs2,1)
    outputs = torch.cat([outputs1.unsqueeze(0), outputs2.unsqueeze(0)], 0)
    outputs = torch.transpose(outputs,1,0)
    for i in range(0, total):
        Correct_number = outputs[i].eq(targets[i].data).cpu().sum() #  Correct number of teachers
        KW_Variance.append(Correct_number)
    return KW_Variance
class Diversity2(nn.Module):
    def __init__(self):
        super(Diversity2, self).__init__()
        self.T = 20
        self.Softmax = nn.Softmax(dim=1)
    def forward(self, outputs1,outputs2,targets):
        
        outputs1 = self.Softmax(outputs1/self.T)
        outputs2 = self.Softmax(outputs2/self.T)
        preds = torch.cat((outputs1.unsqueeze(0), outputs2.unsqueeze(0)),0)
        n_models = len(preds)
        n_test = len(preds[0])
        for i in range(n_test):
            cur_preds = [preds[k][i] for k in range(n_models)]
            d = 0
            for j in range(len(cur_preds)):
                for k in range(j+1, len(cur_preds)):    
                    vx = cur_preds[j] - torch.mean(cur_preds[j])
                    vy = cur_preds[k] - torch.mean(cur_preds[k])
                    cost = torch.sum(vx * vy)/(torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
                    d = d + cost
            v = 0.3*d
            if i==0:
                loss = v.unsqueeze(0)
            else:
                loss = torch.cat((loss,v.unsqueeze(0)),0)
        final_loss = torch.mean(loss)
        return final_loss
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

def KW_Variance3(outputs1,outputs2,outputs3,targets,KW_Variance):
    
    total = targets.size(0)
    outputs1 = torch.argmax(outputs1,1)
    outputs2 = torch.argmax(outputs2,1)
    outputs3 = torch.argmax(outputs3,1)
    outputs = torch.cat([outputs1.unsqueeze(0), outputs2.unsqueeze(0),outputs3.unsqueeze(0)], 0)
    outputs = torch.transpose(outputs,1,0)
    for i in range(0, total):
        Correct_number = outputs[i].eq(targets[i].data).cpu().sum() #  Correct number of teachers
        KW_Variance.append(Correct_number)
    return KW_Variance
class Diversity3(nn.Module):
    def __init__(self):
        super(Diversity3, self).__init__()
        self.T = 20
        self.Softmax = nn.Softmax(dim=1)
        
    def forward(self, outputs1,outputs2,outputs3,targets):
        
        outputs1 = self.Softmax(outputs1/self.T)
        outputs2 = self.Softmax(outputs2/self.T)
        outputs3 = self.Softmax(outputs3/self.T)
        
        preds = torch.cat((outputs1.unsqueeze(0), outputs2.unsqueeze(0), outputs3.unsqueeze(0)),0)
        n_models = len(preds)
        n_test = len(preds[0])
        for i in range(n_test):
            cur_preds = [preds[k][i] for k in range(n_models)]
            d = 0
            for j in range(len(cur_preds)):
                for k in range(j+1, len(cur_preds)):    
                    vx = cur_preds[j] - torch.mean(cur_preds[j])
                    vy = cur_preds[k] - torch.mean(cur_preds[k])
                    cost = torch.sum(vx * vy)/(torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
                    d = d + cost
            v = 0.3*d
            if i==0:
                loss = v.unsqueeze(0)
            else:
                loss = torch.cat((loss,v.unsqueeze(0)),0)
                
        final_loss = torch.mean(loss)
        return final_loss
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def KW_Variance4(outputs1,outputs2,outputs3,outputs4,targets,KW_Variance):
    
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
class Diversity4(nn.Module):
    def __init__(self):
        super(Diversity4, self).__init__()
        self.T = 20
        self.Softmax = nn.Softmax(dim=1)
        
    def forward(self, outputs1,outputs2,outputs3,outputs4,targets):
        
        outputs1 = self.Softmax(outputs1/self.T)
        outputs2 = self.Softmax(outputs2/self.T)
        outputs3 = self.Softmax(outputs3/self.T)
        outputs4 = self.Softmax(outputs4/self.T)
        
        preds = torch.cat((outputs1.unsqueeze(0), outputs2.unsqueeze(0), outputs3.unsqueeze(0),outputs4.unsqueeze(0)),0)
        n_models = len(preds)
        n_test = len(preds[0])
        for i in range(n_test):
            cur_preds = [preds[k][i] for k in range(n_models)]
            d = 0
            for j in range(len(cur_preds)):
                for k in range(j+1, len(cur_preds)):    
                    vx = cur_preds[j] - torch.mean(cur_preds[j])
                    vy = cur_preds[k] - torch.mean(cur_preds[k])
                    cost = torch.sum(vx * vy)/(torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
                    d = d + cost
            v = 0.3*d
            if i==0:
                loss = v.unsqueeze(0)
            else:
                loss = torch.cat((loss,v.unsqueeze(0)),0)
                
        final_loss = torch.mean(loss)
        return final_loss
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
def KW_Variance5(outputs1,outputs2,outputs3,outputs4,outputs5,targets,KW_Variance):
    
    total = targets.size(0)
    outputs1 = torch.argmax(outputs1,1)
    outputs2 = torch.argmax(outputs2,1)
    outputs3 = torch.argmax(outputs3,1)
    outputs4 = torch.argmax(outputs4,1)
    outputs5 = torch.argmax(outputs5,1)
    outputs = torch.cat([outputs1.unsqueeze(0), outputs2.unsqueeze(0),outputs3.unsqueeze(0), outputs4.unsqueeze(0), outputs5.unsqueeze(0)], 0)
    outputs = torch.transpose(outputs,1,0)
    for i in range(0, total):
        Correct_number = outputs[i].eq(targets[i].data).cpu().sum() #  Correct number of teachers
        KW_Variance.append(Correct_number)
    return KW_Variance
class Diversity5(nn.Module):
    def __init__(self):
        super(Diversity5, self).__init__()
        self.T = 20
        self.Softmax = nn.Softmax(dim=1)
        
    def forward(self, outputs1,outputs2,outputs3,outputs4,outputs5,targets):
        
        outputs1 = self.Softmax(outputs1/self.T)
        outputs2 = self.Softmax(outputs2/self.T)
        outputs3 = self.Softmax(outputs3/self.T)
        outputs4 = self.Softmax(outputs4/self.T)
        outputs5 = self.Softmax(outputs5/self.T)
        
        preds = torch.cat((outputs1.unsqueeze(0), outputs2.unsqueeze(0), outputs3.unsqueeze(0),outputs4.unsqueeze(0),outputs5.unsqueeze(0)),0)
        n_models = len(preds)
        n_test = len(preds[0])
        for i in range(n_test):
            cur_preds = [preds[k][i] for k in range(n_models)]
            d = 0
            for j in range(len(cur_preds)):
                for k in range(j+1, len(cur_preds)):    
                    vx = cur_preds[j] - torch.mean(cur_preds[j])
                    vy = cur_preds[k] - torch.mean(cur_preds[k])
                    cost = torch.sum(vx * vy)/(torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
                    d = d + cost
            v = 0.3*d
            if i==0:
                loss = v.unsqueeze(0)
            else:
                loss = torch.cat((loss,v.unsqueeze(0)),0)
                
        final_loss = torch.mean(loss)
        return final_loss
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def KW_Variance6(outputs1,outputs2,outputs3,outputs4,outputs5,outputs6,targets,KW_Variance):
    
    total = targets.size(0)
    outputs1 = torch.argmax(outputs1,1)
    outputs2 = torch.argmax(outputs2,1)
    outputs3 = torch.argmax(outputs3,1)
    outputs4 = torch.argmax(outputs4,1)
    outputs5 = torch.argmax(outputs5,1)
    outputs6 = torch.argmax(outputs6,1)
    outputs = torch.cat([outputs1.unsqueeze(0), outputs2.unsqueeze(0), 
                         outputs3.unsqueeze(0), outputs4.unsqueeze(0), outputs5.unsqueeze(0), outputs6.unsqueeze(0)], 0)
    outputs = torch.transpose(outputs,1,0)
    for i in range(0, total):
        Correct_number = outputs[i].eq(targets[i].data).cpu().sum() #  Correct number of teachers
        KW_Variance.append(Correct_number)
    return KW_Variance
class Diversity6(nn.Module):
    def __init__(self):
        super(Diversity6, self).__init__()
        self.T = 20
        self.Softmax = nn.Softmax(dim=1)
        
    def forward(self, outputs1,outputs2,outputs3,outputs4,outputs5,outputs6,targets):
        
        outputs1 = self.Softmax(outputs1/self.T)
        outputs2 = self.Softmax(outputs2/self.T)
        outputs3 = self.Softmax(outputs3/self.T)
        outputs4 = self.Softmax(outputs4/self.T)
        outputs5 = self.Softmax(outputs5/self.T)
        outputs6 = self.Softmax(outputs6/self.T)
        
        preds = torch.cat((outputs1.unsqueeze(0), outputs2.unsqueeze(0), outputs3.unsqueeze(0), \
                        outputs4.unsqueeze(0), outputs5.unsqueeze(0), outputs6.unsqueeze(0)),0)
        n_models = len(preds)
        n_test = len(preds[0])
        for i in range(n_test):
            cur_preds = [preds[k][i] for k in range(n_models)]
            d = 0
            for j in range(len(cur_preds)):
                for k in range(j+1, len(cur_preds)):    
                    vx = cur_preds[j] - torch.mean(cur_preds[j])
                    vy = cur_preds[k] - torch.mean(cur_preds[k])
                    cost = torch.sum(vx * vy)/(torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
                    d = d + cost
            v = 0.3*d
            if i==0:
                loss = v.unsqueeze(0)
            else:
                loss = torch.cat((loss,v.unsqueeze(0)),0)
                
        final_loss = torch.mean(loss)
        return final_loss
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def KW_Variance7(outputs1,outputs2,outputs3,outputs4,outputs5,outputs6,outputs7,targets,KW_Variance):
    
    total = targets.size(0)
    outputs1 = torch.argmax(outputs1,1)
    outputs2 = torch.argmax(outputs2,1)
    outputs3 = torch.argmax(outputs3,1)
    outputs4 = torch.argmax(outputs4,1)
    outputs5 = torch.argmax(outputs5,1)
    outputs6 = torch.argmax(outputs6,1)
    outputs7 = torch.argmax(outputs7,1)
    outputs = torch.cat([outputs1.unsqueeze(0), outputs2.unsqueeze(0),outputs3.unsqueeze(0), outputs4.unsqueeze(0),\
                         outputs5.unsqueeze(0), outputs6.unsqueeze(0), outputs7.unsqueeze(0)], 0)
    outputs = torch.transpose(outputs,1,0)
    for i in range(0, total):
        Correct_number = outputs[i].eq(targets[i].data).cpu().sum() #  Correct number of teachers
        KW_Variance.append(Correct_number)
    return KW_Variance
class Diversity7(nn.Module):
    def __init__(self):
        super(Diversity7, self).__init__()
        self.T = 20
        self.Softmax = nn.Softmax(dim=1)
        
    def forward(self, outputs1,outputs2,outputs3,outputs4,outputs5,outputs6,outputs7,targets):
        
        outputs1 = self.Softmax(outputs1/self.T)
        outputs2 = self.Softmax(outputs2/self.T)
        outputs3 = self.Softmax(outputs3/self.T)
        outputs4 = self.Softmax(outputs4/self.T)
        outputs5 = self.Softmax(outputs5/self.T)
        outputs6 = self.Softmax(outputs6/self.T)
        outputs7 = self.Softmax(outputs7/self.T)
        
        preds = torch.cat((outputs1.unsqueeze(0), outputs2.unsqueeze(0), outputs3.unsqueeze(0), \
                        outputs4.unsqueeze(0), outputs5.unsqueeze(0), outputs6.unsqueeze(0), outputs7.unsqueeze(0)),0)
        n_models = len(preds)
        n_test = len(preds[0])
        for i in range(n_test):
            cur_preds = [preds[k][i] for k in range(n_models)]
            d = 0
            for j in range(len(cur_preds)):
                for k in range(j+1, len(cur_preds)):    
                    vx = cur_preds[j] - torch.mean(cur_preds[j])
                    vy = cur_preds[k] - torch.mean(cur_preds[k])
                    cost = torch.sum(vx * vy)/(torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
                    d = d + cost
            v = 0.3*d
            if i==0:
                loss = v.unsqueeze(0)
            else:
                loss = torch.cat((loss,v.unsqueeze(0)),0)
                
        final_loss = torch.mean(loss)
        return final_loss
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   
    
    
    
    
    
def KW_Variance8(outputs1,outputs2,outputs3,outputs4,outputs5,outputs6,outputs7,outputs8,targets,KW_Variance):
    
    total = targets.size(0)
    outputs1 = torch.argmax(outputs1,1)
    outputs2 = torch.argmax(outputs2,1)
    outputs3 = torch.argmax(outputs3,1)
    outputs4 = torch.argmax(outputs4,1)
    outputs5 = torch.argmax(outputs5,1)
    outputs6 = torch.argmax(outputs6,1)
    outputs7 = torch.argmax(outputs7,1)
    outputs8 = torch.argmax(outputs8,1)
    outputs = torch.cat([outputs1.unsqueeze(0), outputs2.unsqueeze(0),outputs3.unsqueeze(0), outputs4.unsqueeze(0),\
                         outputs5.unsqueeze(0), outputs6.unsqueeze(0), outputs7.unsqueeze(0), outputs8.unsqueeze(0)], 0)
    outputs = torch.transpose(outputs,1,0)
    for i in range(0, total):
        Correct_number = outputs[i].eq(targets[i].data).cpu().sum() #  Correct number of teachers
        KW_Variance.append(Correct_number)
    return KW_Variance
class Diversity8(nn.Module):
    def __init__(self):
        super(Diversity8, self).__init__()
        self.T = 20
        self.Softmax = nn.Softmax(dim=1)
        
    def forward(self, outputs1,outputs2,outputs3,outputs4,outputs5,outputs6,outputs7,outputs8,targets):
        
        outputs1 = self.Softmax(outputs1/self.T)
        outputs2 = self.Softmax(outputs2/self.T)
        outputs3 = self.Softmax(outputs3/self.T)
        outputs4 = self.Softmax(outputs4/self.T)
        outputs5 = self.Softmax(outputs5/self.T)
        outputs6 = self.Softmax(outputs6/self.T)
        outputs7 = self.Softmax(outputs7/self.T)
        outputs8 = self.Softmax(outputs8/self.T)
        
        preds = torch.cat((outputs1.unsqueeze(0), outputs2.unsqueeze(0), outputs3.unsqueeze(0),outputs4.unsqueeze(0), \
                           outputs5.unsqueeze(0), outputs6.unsqueeze(0), outputs7.unsqueeze(0), outputs8.unsqueeze(0)),0)
        n_models = len(preds)
        n_test = len(preds[0])
        for i in range(n_test):
            cur_preds = [preds[k][i] for k in range(n_models)]
            d = 0
            for j in range(len(cur_preds)):
                for k in range(j+1, len(cur_preds)):    
                    vx = cur_preds[j] - torch.mean(cur_preds[j])
                    vy = cur_preds[k] - torch.mean(cur_preds[k])
                    cost = torch.sum(vx * vy)/(torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
                    d = d + cost
            v = 0.3*d
            if i==0:
                loss = v.unsqueeze(0)
            else:
                loss = torch.cat((loss,v.unsqueeze(0)),0)
                
        final_loss = torch.mean(loss)
        return final_loss