import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cpu')

# load dataset
batch_size = 32

train_dataset = datasets.MNIST('./data', 
                               train=True, 
                               download=True, 
                               transform=transforms.ToTensor())

validation_dataset = datasets.MNIST('./data', 
                                    train=False, 
                                    transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, 
                                                batch_size=batch_size, 
                                                shuffle=False)


# define a teacher network and a student network 
class Net_T(nn.Module):
    def __init__(self):
        super(Net_T, self).__init__()
        self.fc1 = nn.Linear(28*28, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x, self.fc4(x)
        
        
class Net_S(nn.Module):
    def __init__(self):
        super(Net_S, self).__init__()
        self.fc1 = nn.Linear(28*28, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        return x, self.fc2(x)
        


# define the validate function
def validate(model, loss_vector, accuracy_vector):
    model.eval()
    val_loss, correct = 0, 0
    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)
        fea, output = model(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct.to(torch.float32) / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)
    
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))
        
################################################distillation###################################################
# load teacher model
model_t = Net_T().to(device)
model_t.load_state_dict(torch.load('./model_t.pth')['model'])
criterion = nn.CrossEntropyLoss()
print(model_t)

model_s = Net_S().to(device)
optimizer_s = torch.optim.SGD(model_s.parameters(), lr=0.01, momentum=0.5)
print(model_s)

# define the training function
def train_student(epoch, log_interval=200):
    # Set model_s to training mode
    model_s.train()
    # Set model_t to testing mode
    model_t.eval()
    
    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(train_loader):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)

        # Zero gradient buffers
        optimizer_s.zero_grad() 
        
        # Pass data through the networks
        fea, output = model_s(data)
        fea_t, output_t = model_t(data)

        # Calculate distillation loss
        # Logit distillation
        T = 4
        p_s = F.log_softmax(output/T, dim=1)
        p_t = F.softmax(output_t/T, dim=1)
        loss_kd = F.kl_div(p_s, p_t, size_average=False) * (T**2) / output.shape[0]
        
        # Feature distillation
        #loss_kd = torch.nn.functional.mse_loss(fea, fea_t)
        
        # Similarity preserving distillation
        # to be done
        
        # Calculate total loss
        loss = criterion(output, target) + loss_kd

        # Backpropagate
        loss.backward()
        
        # Update weights
        optimizer_s.step()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

# start distillation
epochs = 5

lossv, accv = [], []
for epoch in range(1, epochs + 1):
    train_student(epoch)
    validate(model_s, lossv, accv)
    
state = {
    'model': model_s.state_dict(),
}
torch.save(state, './model_s.pth')

# plot loss and accuracy
plt.figure(figsize=(5,3))
plt.plot(np.arange(1,epochs+1), lossv)
plt.title('validation loss')
plt.savefig('./loss_student.jpg')
plt.clf()

plt.figure(figsize=(5,3))
plt.plot(np.arange(1,epochs+1), accv)
plt.title('validation accuracy')
plt.savefig('./acc_student.jpg')
plt.clf()