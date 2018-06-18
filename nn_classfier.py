import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from util import config
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 4)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 4)
        self.fc1 = nn.Linear(160, 50)
        self.fc2 = nn.Linear(50,20)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 160)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict(self,x):
        self.eval()
        x=torch.from_numpy(x).float()
        x=x.view(-1,1,config["cep_num"],config["clip_num"]*2)
        out=self(x)
        ret=torch.argmax(out,1)
        return ret.data.tolist()

    def save_file(self,path):
        torch.save(self.state_dict(),path)

    def load_file(self,path):
        self.load_state_dict(torch.load(path))