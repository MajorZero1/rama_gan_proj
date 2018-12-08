import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from lenet import Net
import csv


batch_size = 10
test_batch_size = 10
device = 'cpu'
num_of_epochs = 10

#data loaders
train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./data', train=True, download=True,
                  transform=transforms.Compose([transforms.ToTensor()])),
            batch_size=batch_size, shuffle=True)
        
test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./data', train=False, 
                  transform=transforms.Compose([transforms.ToTensor()])),
            batch_size=test_batch_size, shuffle=False)
            


model = Net().to(device)
optimizer = optim.Adam(model.parameters())

for epoch in range(1, num_of_epochs):  	
    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(image)
        loss = F.cross_entropy(output,label)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train epoch: %d \t Iter: %d \t Loss: \t %f' %
             (epoch, batch_idx, loss.item()))
               

    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(test_loader):
            image, label = image.to(device), label.to(device)
            output = model(image)
            unused, predicted = output.max(1)
            correct_batch = predicted.eq(label.view_as(predicted)).sum().item()
            correct += correct_batch
            print('Test epoch: %d \t Iter: %d \t Correct: %d/%d' %
               (epoch, batch_idx, test_batch_size))
        
        print('Test epoch: %d \t Num correct: %d / %d' % 
         (epoch, correct, len(test_loader)))
         
torch.save({'state_dict': model.state_dict()},'./lenet.pth' % epoch)
        
        
