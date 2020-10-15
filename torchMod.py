import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.utils.data
import numpy as np
import os
import pandas as pd


class FER2013(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        curr_path = os.getcwd()
        dataframe = pd.read_csv(curr_path + '/fer2013/fer2013.csv')
        data = np.array(dataframe)
        length = 30000
        y = data[:length, 0]
        # y = np.zeros((len(output), 7))
        # for i in range(len(output)):
        #     y[i][output[i]] = 1
        X = data[:length, 1]
        X = [np.fromstring(x, dtype='int', sep=' ') for x in X]
        X = np.array([np.fromstring(x, dtype='int', sep=' ').reshape(2304)
                      for x in data[:length, 1]])
        X = X.reshape(-1, 1, 48, 48)
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        train_set = torch.from_numpy(np.asarray(X)).float(
        ), torch.from_numpy(np.asarray(y)).long()
        return train_set


class testFER2013(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        curr_path = os.getcwd()
        dataframe = pd.read_csv(curr_path + '/fer2013/fer2013.csv')
        data = np.array(dataframe)
        length1 = 30000
        length2 = 34000
        y = data[length1:length2, 0]
        # y = np.zeros((len(output), 7))
        # for i in range(len(output)):
        #     y[i][output[i]] = 1
        X = data[length1:length2, 1]
        X = [np.fromstring(x, dtype='int', sep=' ') for x in X]
        X = np.array([np.fromstring(x, dtype='int', sep=' ').reshape(2304)
                      for x in data[length1:length2, 1]])
        X = X.reshape(-1, 1, 48, 48)
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        train_set = torch.from_numpy(np.asarray(X)).float(
        ), torch.from_numpy(np.asarray(y)).long()
        return train_set


class FerNet(nn.Module):
    def __init__(self):
        super(FerNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, stride=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1)
        self.conv3 = nn.Conv2d(48, 64, 3, stride=1)
        self.conv4 = nn.Conv2d(64, 128, 1, stride=1)
        self.conv5 = nn.Conv2d(32, 48, 3, stride=1, padding=1)
        # self.max_pool = nn.Conv2d()
        self.dropout = nn.Dropout(0.5)
        # self.fc1 = nn.Linear(4 * 4 * 128, 2048)
        self.fc2 = nn.Linear(2 * 2 * 128, 128)
        self.fc3 = nn.Linear(128, 7)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        # print(x.shape)
        x = x.view(-1, 2 * 2 * 128)
        # x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


if __name__ == '__main__':

    trainset = FER2013()
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=0)
    print("Trainset Loaded")
    testset = testFER2013()
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=True, num_workers=0)
    print("Testset Loaded")
    net = FerNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=7e-6, momentum=0.1500101011)

    for epoch in range(50):
        print('Running Epoch', epoch + 1)
        running_loss = 0
        for i, Data in enumerate(trainloader):
            inputs, expected = Data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, expected)
            loss.backward()
            optimizer.step()

        count_correct = 0
        count_incorrect = 0
        for i, Data in enumerate(testloader):
            inputs, expected = Data
            # optimizer.zero_grad()
            outputs = net(inputs)
            out = outputs.detach().numpy()
            exp = expected.detach().numpy()
            for i in range(len(out)):
                count_correct += (np.argmax(out[i]) == exp[i])
                count_incorrect += (np.argmax(out[i]) != exp[i])
                print(np.argmax(out[i]), exp[i])
        print('Correct:', count_correct, 'Incorrect:', count_incorrect)
