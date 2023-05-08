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


# details of dataset                                    
for (X_train, y_train) in train_loader:
    print('X_train:', X_train.size(), 'type:', X_train.type())
    print('y_train:', y_train.size(), 'type:', y_train.type())
    break
    
pltsize=1.5
plt.figure(figsize=(10*pltsize, pltsize))

# plot some images on MNIST
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.axis('off')
    plt.imshow(X_train[i,:,:,:].numpy().reshape(28,28), cmap="gray_r")
    plt.title('Class: '+str(y_train[i].item()))
plt.savefig('./mnist.jpg')
plt.clf()
    

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
        

model_t = Net_T().to(device)
optimizer = torch.optim.SGD(model_t.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()
print(model_t)

# define the training function
def train(epoch, log_interval=200):
    # Set model_t to training mode
    model_t.train()
    
    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(train_loader):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)

        # Zero gradient buffers
        optimizer.zero_grad() 
        
        # Pass data through the network
        fea, output = model_t(data)

        # Calculate loss
        loss = criterion(output, target)

        # Backpropagate
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
                
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
        
        
# start training
epochs = 5

lossv, accv = [], []
for epoch in range(1, epochs + 1):
    train(epoch)
    validate(model_t, lossv, accv)

state = {
    'model': model_t.state_dict(),
}
torch.save(state, './model_t.pth')
    
    
# plot loss and accuracy
plt.figure(figsize=(5,3))
plt.plot(np.arange(1,epochs+1), lossv)
plt.title('validation loss')
plt.savefig('./loss.jpg')
plt.clf()

plt.figure(figsize=(5,3))
plt.plot(np.arange(1,epochs+1), accv)
plt.title('validation accuracy')
plt.savefig('./acc.jpg')
plt.clf()

# define the visualization function
def vis_mnist(model, correct_tensor, wrong_tensor):
    model.eval()
    correct_count = 0 
    wrong_count = 0 
    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)
        fea, output = model(data)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        #correct += pred.eq(target.data).cpu().sum()
        for i in range(0, data.size(0)):
            if (correct_count < correct_tensor.size(0)) and (pred[i] == target[i]): 
                correct_tensor[correct_count] = data[i]
                correct_count = correct_count + 1
            elif (wrong_count < wrong_tensor.size(0)) and (pred[i] != target[i]): 
                wrong_tensor[wrong_count] = data[i]
                wrong_count = wrong_count + 1
        if (correct_count == correct_tensor.size(0)) and (wrong_count == wrong_tensor.size(0)):
            break

correct_t = torch.zeros(10, 1, 28, 28)
wrong_t = torch.zeros(10, 1, 28, 28)
vis_mnist(model_t, correct_t, wrong_t)

# plot correctly-classified images on MNIST
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.axis('off')
    plt.imshow(correct_t[i,:,:,:].numpy().reshape(28,28), cmap="gray_r")
plt.savefig('./correct.jpg')
plt.clf()

# plot incorrectly-classified images on MNIST
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.axis('off')
    plt.imshow(wrong_t[i,:,:,:].numpy().reshape(28,28), cmap="gray_r")
plt.savefig('./wrong.jpg')
plt.clf()
