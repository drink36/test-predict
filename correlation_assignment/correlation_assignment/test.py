import pandas as pd
import numpy as np
from PIL import Image
import tqdm
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
def extract_coordinates(img_path):
    img = Image.open(img_path).convert('L')
    img_array = np.array(img)
    threshold = 1  
    dots = np.column_stack(np.where(img_array < threshold))
    max_height, max_width = img_array.shape
    relative_coords = dots / np.array([max_width, max_height])
    # change the x to y and y to x
    relative_coords[:, 0], relative_coords[:, 1] = relative_coords[:, 1], relative_coords[:, 0].copy()
    relative_coords[:, 1] = 1 - relative_coords[:, 1]
    
    relative_coords = torch.Tensor(relative_coords).cuda()
    return relative_coords
class PointsLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PointsLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, sequence):
        # LSTM layer
        lstm_out, _ = self.lstm(sequence)
        # Take only the output of the last LSTM cell
        last_cell_output = lstm_out[:, -1, :]
        # Linear layer to produce output
        output = self.linear(last_cell_output)
        return output


# read the data
data = pd.read_csv('responses.csv')
image_data = []
target_data = []
images=os.listdir('images')
# drop the rows without in the images
data = data[data['id'].isin([item.split('.')[0] for item in images])]
print(data.shape)
for item in tqdm.tqdm(data['id']):
    dir='images/'+item+'.png'
    if os.path.exists(dir):
        image_data.append(extract_coordinates(dir))
        target_data.append(data[data['id'] == item]['corr'].values[0])

X = torch.nn.utils.rnn.pad_sequence(image_data, batch_first=True)
y = torch.tensor(target_data, dtype=torch.float).view(-1, 1)  # 假设y是一个列向量
print(X.shape, y.shape)
print(X[0], y[0])
train_size = int(0.8 * len(X))
train_dataset = TensorDataset(X[:train_size], y[:train_size])
test_dataset = TensorDataset(X[train_size:], y[train_size:])

batch_size = 10  
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Loss function and optimizer
criterion = nn.MSELoss()
# Model instantiation
input_size = 2  # x and y coordinates
hidden_size = 50  # Can be changed to a different value
output_size = 1  # One output value representing the correlation

model = PointsLSTM(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Training
model.cuda()
num_epochs = 20  
for epoch in range(num_epochs):
    model.train() 
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad() 
        batch_X = batch_X.cuda() 
        batch_y = batch_y.cuda() 
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()  
        optimizer.step() 
    print(f"Epoch {epoch}, Loss: {loss.item()}")
# Testing
model.eval()
avg_diff = 0
avg_loss = 0
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.cuda()
        batch_y = batch_y.cuda()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        print(output, batch_y)
        diff = torch.abs(output - batch_y)
        avg_diff += diff
        avg_loss += loss
print('avg_diff: ', avg_diff / len(test_loader))
print('avg_loss: ', avg_loss / len(test_loader))