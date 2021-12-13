import torch.nn as nn
import timm
import torch
import numpy as np

class x_W(nn.Module):
    def __init__(self):
        super(x_W, self).__init__()
        self.classification_fc_yaw = nn.Linear(1000,120)

class y_W(nn.Module):
    def __init__(self):
        super(y_W, self).__init__()
        self.classification_fc_pitch = nn.Linear(1000,66)

class z_W(nn.Module):
    def __init__(self):
        super(z_W, self).__init__()
        self.classification_fc_roll  = nn.Linear(1000,66)

class add_W(nn.Module):
    def __init__(self):
        super(add_W, self).__init__()
        self.x = x_W()
        self.y = y_W()
        self.z = z_W()
        self.relu = nn.ReLU()
        self.Softmax = nn.Softmax(1)
    
class WHENet(nn.Module):
    def __init__(self):
        super(WHENet, self).__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True)
        self.add = add_W()
        self.idx_tensor = [idx for idx in range(66)]
        self.idx_tensor = np.array(self.idx_tensor, dtype=np.float32)
        self.idx_tensor = torch.from_numpy(self.idx_tensor)
        self.idx_tensor_yaw = [idx for idx in range(120)]
        self.idx_tensor_yaw = np.array(self.idx_tensor_yaw, dtype=np.float32)
        self.idx_tensor_yaw = torch.from_numpy(self.idx_tensor_yaw)


    def forward(self,x):
        x =                    self.add.relu(self.model(x))
        classification_yaw =   self.add.x.classification_fc_yaw(x)
        classification_pitch = self.add.y.classification_fc_pitch(x)
        classification_roll =  self.add.z.classification_fc_roll(x)
        regression_yaw =       self.add.Softmax(classification_yaw)
        regression_pitch =     self.add.Softmax(classification_pitch)
        regression_roll =      self.add.Softmax(classification_roll)
        regression_yaw =       torch.sum(regression_yaw   * self.idx_tensor_yaw, axis=1) * 3 - 180
        regression_pitch =     torch.sum(regression_pitch * self.idx_tensor,     axis=1) * 3 - 99
        regression_roll =      torch.sum(regression_roll  * self.idx_tensor,     axis=1) * 3 - 99
        return classification_yaw, classification_pitch, classification_roll, regression_yaw, regression_pitch, regression_roll