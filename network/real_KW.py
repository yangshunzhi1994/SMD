from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class real_KW2(nn.Module):
    def __init__(self):
        super(real_KW2, self).__init__()
        self.T = 20
        self.Softmax = nn.Softmax(dim=1)
        
    def forward(self, outputs1,outputs2,real_KW):
        
        mimic = (outputs1+outputs2)/2
        
        outputs1 = self.Softmax(outputs1/self.T)
        outputs2 = self.Softmax(outputs2/self.T)
        mimic = self.Softmax(mimic/self.T)
        
        total = len(outputs1)
        for i in range(total):
            d1 = torch.abs(outputs1[i] - mimic[i]).sum()
            d2 = torch.abs(outputs2[i] - mimic[i]).sum()
            KW = (d1+d2)/2
            
            if real_KW is None:
                real_KW = KW.unsqueeze(0)
            else:
                real_KW = torch.cat((real_KW,KW.unsqueeze(0)),0)
                
        return real_KW
    
    
    
    
    
    
    
class real_KW3(nn.Module):
    def __init__(self):
        super(real_KW3, self).__init__()
        self.T = 20
        self.Softmax = nn.Softmax(dim=1)
        
    def forward(self, outputs1,outputs2,outputs3,real_KW):
        
        mimic = (outputs1+outputs2+outputs3)/3
        
        outputs1 = self.Softmax(outputs1/self.T)
        outputs2 = self.Softmax(outputs2/self.T)
        outputs3 = self.Softmax(outputs3/self.T)
        mimic = self.Softmax(mimic/self.T)
        
        total = len(outputs1)
        for i in range(total):
            d1 = torch.abs(outputs1[i] - mimic[i]).sum()
            d2 = torch.abs(outputs2[i] - mimic[i]).sum()
            d3 = torch.abs(outputs3[i] - mimic[i]).sum()
            KW = (d1+d2+d3)/3
            
            if real_KW is None:
                real_KW = KW.unsqueeze(0)
            else:
                real_KW = torch.cat((real_KW,KW.unsqueeze(0)),0)
                
        return real_KW
    
    

class real_KW4(nn.Module):
    def __init__(self):
        super(real_KW4, self).__init__()
        self.T = 20
        self.Softmax = nn.Softmax(dim=1)
        
    def forward(self, outputs1,outputs2,outputs3,outputs4,real_KW):
        
        mimic = (outputs1+outputs2+outputs3+outputs4)/4
        
        outputs1 = self.Softmax(outputs1/self.T)
        outputs2 = self.Softmax(outputs2/self.T)
        outputs3 = self.Softmax(outputs3/self.T)
        outputs4 = self.Softmax(outputs4/self.T)
        mimic = self.Softmax(mimic/self.T)
        
        total = len(outputs1)
        for i in range(total):
            d1 = torch.abs(outputs1[i] - mimic[i]).sum()
            d2 = torch.abs(outputs2[i] - mimic[i]).sum()
            d3 = torch.abs(outputs3[i] - mimic[i]).sum()
            d4 = torch.abs(outputs4[i] - mimic[i]).sum()
            KW = (d1+d2+d3+d4)/4
            
            if real_KW is None:
                real_KW = KW.unsqueeze(0)
            else:
                real_KW = torch.cat((real_KW,KW.unsqueeze(0)),0)
        return real_KW
    
    
class real_KW5(nn.Module):
    def __init__(self):
        super(real_KW5, self).__init__()
        self.T = 20
        self.Softmax = nn.Softmax(dim=1)
        
    def forward(self, outputs1,outputs2,outputs3,outputs4,outputs5,real_KW):
        
        mimic = (outputs1+outputs2+outputs3+outputs4+outputs5)/5
        
        outputs1 = self.Softmax(outputs1/self.T)
        outputs2 = self.Softmax(outputs2/self.T)
        outputs3 = self.Softmax(outputs3/self.T)
        outputs4 = self.Softmax(outputs4/self.T)
        outputs5 = self.Softmax(outputs5/self.T)
        mimic = self.Softmax(mimic/self.T)
        
        total = len(outputs1)
        for i in range(total):
            d1 = torch.abs(outputs1[i] - mimic[i]).sum()
            d2 = torch.abs(outputs2[i] - mimic[i]).sum()
            d3 = torch.abs(outputs3[i] - mimic[i]).sum()
            d4 = torch.abs(outputs4[i] - mimic[i]).sum()
            d5 = torch.abs(outputs5[i] - mimic[i]).sum()
            KW = (d1+d2+d3+d4+d5)/5
            
            if real_KW is None:
                real_KW = KW.unsqueeze(0)
            else:
                real_KW = torch.cat((real_KW,KW.unsqueeze(0)),0)
                
        return real_KW
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
class real_KW6(nn.Module):
    def __init__(self):
        super(real_KW6, self).__init__()
        self.T = 20
        self.Softmax = nn.Softmax(dim=1)
        
    def forward(self, outputs1,outputs2,outputs3,outputs4,outputs5,outputs6,real_KW):
        
        mimic = (outputs1+outputs2+outputs3+outputs4+outputs5+outputs6)/6
        
        outputs1 = self.Softmax(outputs1/self.T)
        outputs2 = self.Softmax(outputs2/self.T)
        outputs3 = self.Softmax(outputs3/self.T)
        outputs4 = self.Softmax(outputs4/self.T)
        outputs5 = self.Softmax(outputs5/self.T)
        outputs6 = self.Softmax(outputs6/self.T)
        mimic = self.Softmax(mimic/self.T)
        
        total = len(outputs1)
        for i in range(total):
            d1 = torch.abs(outputs1[i] - mimic[i]).sum()
            d2 = torch.abs(outputs2[i] - mimic[i]).sum()
            d3 = torch.abs(outputs3[i] - mimic[i]).sum()
            d4 = torch.abs(outputs4[i] - mimic[i]).sum()
            d5 = torch.abs(outputs5[i] - mimic[i]).sum()
            d6 = torch.abs(outputs6[i] - mimic[i]).sum()
            KW = (d1+d2+d3+d4+d5+d6)/6
            
            if real_KW is None:
                real_KW = KW.unsqueeze(0)
            else:
                real_KW = torch.cat((real_KW,KW.unsqueeze(0)),0)
                
        return real_KW
    
    
    
    
    
class real_KW7(nn.Module):
    def __init__(self):
        super(real_KW7, self).__init__()
        self.T = 20
        self.Softmax = nn.Softmax(dim=1)
        
    def forward(self, outputs1,outputs2,outputs3,outputs4,outputs5,outputs6,outputs7,real_KW):
        
        mimic = (outputs1+outputs2+outputs3+outputs4+outputs5+outputs6+outputs7)/7
        
        outputs1 = self.Softmax(outputs1/self.T)
        outputs2 = self.Softmax(outputs2/self.T)
        outputs3 = self.Softmax(outputs3/self.T)
        outputs4 = self.Softmax(outputs4/self.T)
        outputs5 = self.Softmax(outputs5/self.T)
        outputs6 = self.Softmax(outputs6/self.T)
        outputs7 = self.Softmax(outputs7/self.T)
        mimic = self.Softmax(mimic/self.T)
        
        total = len(outputs1)
        for i in range(total):
            d1 = torch.abs(outputs1[i] - mimic[i]).sum()
            d2 = torch.abs(outputs2[i] - mimic[i]).sum()
            d3 = torch.abs(outputs3[i] - mimic[i]).sum()
            d4 = torch.abs(outputs4[i] - mimic[i]).sum()
            d5 = torch.abs(outputs5[i] - mimic[i]).sum()
            d6 = torch.abs(outputs6[i] - mimic[i]).sum()
            d7 = torch.abs(outputs7[i] - mimic[i]).sum()
            KW = (d1+d2+d3+d4+d5+d6+d7)/7
            
            if real_KW is None:
                real_KW = KW.unsqueeze(0)
            else:
                real_KW = torch.cat((real_KW,KW.unsqueeze(0)),0)
                
        return real_KW
    
    
    
    
    
class real_KW8(nn.Module):
    def __init__(self):
        super(real_KW8, self).__init__()
        self.T = 20
        self.Softmax = nn.Softmax(dim=1)
        
    def forward(self, outputs1,outputs2,outputs3,outputs4,outputs5,outputs6,outputs7,outputs8,real_KW):
        
        mimic = (outputs1+outputs2+outputs3+outputs4+outputs5+outputs6+outputs7+outputs8)/8
        
        outputs1 = self.Softmax(outputs1/self.T)
        outputs2 = self.Softmax(outputs2/self.T)
        outputs3 = self.Softmax(outputs3/self.T)
        outputs4 = self.Softmax(outputs4/self.T)
        outputs5 = self.Softmax(outputs5/self.T)
        outputs6 = self.Softmax(outputs6/self.T)
        outputs7 = self.Softmax(outputs7/self.T)
        outputs8 = self.Softmax(outputs8/self.T)
        mimic = self.Softmax(mimic/self.T)
        
        total = len(outputs1)
        for i in range(total):
            d1 = torch.abs(outputs1[i] - mimic[i]).sum()
            d2 = torch.abs(outputs2[i] - mimic[i]).sum()
            d3 = torch.abs(outputs3[i] - mimic[i]).sum()
            d4 = torch.abs(outputs4[i] - mimic[i]).sum()
            d5 = torch.abs(outputs5[i] - mimic[i]).sum()
            d6 = torch.abs(outputs6[i] - mimic[i]).sum()
            d7 = torch.abs(outputs7[i] - mimic[i]).sum()
            d8 = torch.abs(outputs8[i] - mimic[i]).sum()
            KW = (d1+d2+d3+d4+d5+d6+d7+d8)/8
            
            if real_KW is None:
                real_KW = KW.unsqueeze(0)
            else:
                real_KW = torch.cat((real_KW,KW.unsqueeze(0)),0)
                
        return real_KW