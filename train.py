import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from scipy import misc
import csv
from lenet import Net

batch_size = 10
device = 'cuda'
train_log = './train_log.csv'
distribution_log = './distribution_log.csv'

#data loaders
train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./data', train=True, download=True,
                  transform=transforms.Compose([transforms.ToTensor()])),
            batch_size=batch_size, shuffle=True)
        
#test_loader = torch.utils.data.DataLoader(
#    datasets.FashionMNIST('./data', train=False, 
#                  transform=transforms.Compose([transforms.ToTensor()])),
#            batch_size=test_batch_size, shuffle=False)
            

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(100, 64 , kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 128 , kernel_size=5, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2)
            )
        
    #takes random bx1x3x3
    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.net = nn.Sequential(
		    nn.Conv2d(1, 32, kernel_size=4, stride=2),
		    nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, kernel_size=4,stride=2),
		    nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=4,stride=2),
			nn.Sigmoid()
			)

	def forward(self, x):
		return self.net(x)

#load lenet to analyze mode collapse
lenet = Net().to(device)
checkpoint = torch.load('./lenet.pth')
lenet.load_state_dict(checkpoint['state_dict'])        
        
gen = Generator()
gen.to(device)
dis = Discriminator()
dis.to(device)
        
optimizer_gen = optim.Adam(gen.parameters())
optimizer_dis = optim.Adam(dis.parameters())

# y_n*log(x_n) + (1-y_n)*log(1-x_n)
criterion = nn.BCELoss()

train_csv = open(train_log,'w')
fieldnames = ['epoch','batch','loss_gen','loss_dis']
csv_writer = csv.DictWriter(train_csv,fieldnames = fieldnames)
csv_writer.writeheader()

count_csv = open(distribution_log,'w')
csv_dist_writer = csv.writer(count_csv)


for epoch in range(0,10):
    print('training epoch %d' % epoch)
    for i, data in enumerate(train_loader, 0):
    	#train discriminator
        #log(D(x)) + log(1 - D(G(z)))
        dis.zero_grad()
        #train real
        real_data = data[0].to(device)
        real_label = torch.ones(batch_size, device=device)
        real_label_output = dis(real_data).squeeze()
        real_loss = criterion(real_label_output, real_label)

        # train with fake
        noise = torch.randn(batch_size, 100, 1, 1, device=device)
        fake_data = gen(noise)
        fake_label = torch.zeros(batch_size, device=device)
        fake_label_output = dis(fake_data.detach()).squeeze()
        fake_loss = criterion(fake_label_output, fake_label)
        
        loss_dis = fake_loss+real_loss
        loss_dis.backward()
        optimizer_dis.step()
        
        
        #train generator
        #log(D(G(z)))
        gen.zero_grad()
        label = torch.ones(batch_size,device=device)
        noise = torch.randn(batch_size, 100, 1, 1, device=device)
        output = dis(gen(noise)).squeeze()
        loss_gen = criterion(output, label)
        loss_gen.backward()
        optimizer_gen.step()
        csv_writer.writerow({'epoch': epoch, 'batch': i,
              'loss_gen': loss_gen.item(), 'loss_dis': loss_dis.item()})
        if i % 100 == 0:
            print('epoch:%d \t loss_gen: %05f \t loss_dis: %05f \t' %(epoch,
                   loss_gen.item(), loss_dis.item()))
    #after each epoch sample some images from the generator  
    noise = torch.randn(20, 100, 1, 1, device=device)
    fake_data = gen(noise)
    for i in range(0,20):
    	save_name = './images/epoch%dsample%d.jpg' %(epoch,i)
    	image = fake_data[i,:,:,:].squeeze().contiguous().cpu().detach().numpy()
    	misc.imsave(save_name,image)
    #generate some images and classify them with lenet to look at distribution	
    count = torch.zeros(10)
    for i in range(0,100):
        noise = torch.randn(10,100,1,1,device=device)
        fake_data = gen(noise)
        out = lenet(fake_data)
        unused,predicted = out.max(1)
        for j in range(0,len(predicted)):
            count[predicted[j]] +=1
    csv_dist_writer.writerow(count.detach().cpu().numpy())


train_csv.close()
csv_dist_writer.close()

