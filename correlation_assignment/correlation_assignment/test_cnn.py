import pandas as pd
import numpy as np
from PIL import Image
import tqdm
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# open cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3= nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# read the data
data = pd.read_csv('responses.csv')
image_data = []
images=os.listdir('images')
data = data[data['id'].isin([item.split('.')[0] for item in images])]
data['corr'] = data['corr'].astype(float)
output_data = torch.Tensor(data['corr'].values).unsqueeze(1).cuda()
for item in tqdm.tqdm(data['id']):
    dir='images/'+item+'.png'
    if os.path.exists(dir):
        # let image data into tensor
        img = Image.open(dir).convert('L').resize((128, 128))
        img_array = np.array(img)
        img_array = torch.Tensor(img_array).unsqueeze(0).cuda()
        image_data.append(img_array)
# split the image data into training and testing
# data into training and testing
train_data = torch.stack(image_data[:int(len(image_data)*0.8)])
train_output = output_data[:int(len(image_data)*0.8)]
test_data = torch.stack(image_data[int(len(image_data)*0.8):])
test_output = output_data[int(len(image_data)*0.8):]
# devide the data into different batches
train_data = torch.split(train_data, 1000)
train_output = torch.split(train_output, 1000)
print(len(train_data), len(test_data))
print(len(train_output), len(test_output))
# define the model
model=CNN()
model.cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(5):
    for batch, data in enumerate(zip(train_data)):
        model.train()
        for i, item in enumerate(data):
            optimizer.zero_grad()
            output = model(item)
            loss = criterion(output, train_output[batch][i])
            loss.backward()
            optimizer.step()
    print('Epoch:', epoch, 'Loss:', loss.item())
# testing the model with the testing data and calculate the difference between the prediction and the real value
model.eval()
avg_diff = 0
avg_loss = 0
with torch.no_grad():
    for i, item in enumerate(test_data):
        output = model(item)
        loss=criterion(output, test_output[i])
        diff = torch.abs(output - test_output[i])
        avg_diff += diff
        avg_loss += loss

print('avg_diff: ',avg_diff/len(test_data))
print('avg_loss: ',avg_loss/len(test_data))
# save the model
torch.save(model, 'model.pth')