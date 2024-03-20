from torchvision.datasets import CIFAR10
import numpy as np 
import torchvision.transforms as transforms
import torch 
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models_cifar import resnet
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
import signal
import os
from data_preprocessing import data_preprocessing




def train(n_epochs, model_name, initial_learning_rate, mixup_condition, schedulLR):

    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Import and preprocess the data:
    trainloader, testloader, mixup = data_preprocessing.data_preprocess()

    train_loss=[]
    test_loss=[]
    best_accuracy = 0
    learning_rates = []


    model = resnet.ResNet34()

    model.to(device)

    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate)

    if schedulLR=='plateau':
        scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    elif schedulLR=='cosine':
        scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, 130, 0.0001)

    def interrupt_handler(sig, frame):
        print('Interrupt detected, Saving model...')
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        })
        print('model saved')
        exit(0)

    signal.signal(signal.SIGINT, interrupt_handler)

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        
        correct = 0
        total = 0
        running_loss = 0.0

        for i, data in tqdm(enumerate(trainloader, 0)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            r = np.random.rand(1)
            if (epoch >= 80) and (mixup_condition == True):
                # generate mixed sample
          
                lam = np.random.beta(1.0, 1.0)
                rand_index = torch.randperm(inputs.size()[0]).cuda()
                target_a = labels
                target_b = labels[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                # compute output
                output = model(inputs)
                loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
            else:
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()

            if epoch<80:
                #accuracy computation
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                accuracy = (100 * correct / total)

            running_loss += loss.item()

            # TRAINING LOSS
            if i % 1563 == 1562:    # print every 6667 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 1563))
                logging.info(f'{model_name}  -  Epoch {epoch + 1}/{n_epochs} - Loss: {running_loss/1563} - Training accuracy: {accuracy}')
                train_loss.append(running_loss/1563)
                running_loss = 0.0



        # VALIDATION STEP
        correct = 0
        total = 0
        # set model in evaluation mode
        model.eval() 
        with torch.no_grad():  # torch.no_grad for TESTING
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                accuracy = (100 * correct / total)

        # SCHEDULER STEP
        scheduler.step(running_loss)

        # LOGFILE UPDATE 
        print('Accuracy of the network on the 10000 test images: %d %%' % (accuracy))
        logging.info(f'{model_name}  -  Epoch {epoch + 1}/{n_epochs} - Validation loss =  {running_loss/313} - Validation accuracy: {accuracy}')

        # ACCURACY COMPUTING
        test_loss.append(running_loss/313)
        running_loss = 0.0
        if best_accuracy < accuracy and epoch>5:
            best_accuracy=accuracy

        # SET MODEL ON TRAIN MODE
        model.train()
        
    PATH = './result/chkp/resnet_teacher.pth'
    torch.save(model.state_dict(), PATH)
    print(train_loss,test_loss)
    return train_loss, test_loss, model

        

    print('Finished Training')


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2



def save_checkpoint(state, filename="checkpoint_teaccher.pth.tar"):
    torch.save(state,filename)






