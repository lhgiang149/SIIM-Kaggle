import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from efficientnet_pytorch import EfficientNet

import os
import tqdm
from utilities import *

params = {'size_input': 224,
         'workers': 0,
dataframe = pd.read_csv('./../data/test.csv')
         'batch_size': 32,
         'num_classes':2}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

IDs = dataframe.image_name.to_list()
                      
test_dataset = PyTorchTestImageDataset(image_list = IDs, size_image = params['size_input'])
test_dataloader = DataLoader(dataset = test_dataset, batch_size = params['batch_size'], shuffle = False, num_workers=params['workers'])

# model = resnet50()
# model.fc = nn.Linear(model.fc.in_features, params['num_classes'])
model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = nn.Linear(model._fc.in_features,2)

weights_path = 'C:/Users/emage/OneDrive/Desktop/SIIM-ISIC-Melanoma-Classification/Toby/result/model/efficientnetb0_ver2_epochs_3_loss_1.0184.pth'
model.load_state_dict(torch.load(weights_path))
model.to(device)

softmax = nn.Softmax()
with torch.no_grad():
    for i, data in enumerate(tqdm.tqdm(test_dataloader, position=0, leave=True)):
        
        data_batch = data.to(device)
                        
        b_size = data_batch.size(0)

        output = model(data_batch)

        prob = softmax(output)
        
        if i == 0:
            torch.cuda.synchronize() 
            valid_prob = np.reshape(prob.cpu().detach().numpy(), (b_size,params['num_classes']))[:,1]
        else:
            torch.cuda.synchronize() 
            temp_prob = np.reshape(prob.cpu().detach().numpy(), (b_size,params['num_classes']))[:,1]
            valid_prob = np.concatenate((valid_prob, temp_prob), axis = 0)
# print(valid_prob.shape)
path = './../data/sample_submission.csv'
df = pd.read_csv(path)
valid_prob = np.round(valid_prob,4)
df.target = valid_prob
df.to_csv('submission.csv', index=False)