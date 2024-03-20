from torchvision.datasets import CIFAR10
import numpy as np 
import torchvision.transforms as transforms
import torch 
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import v2
import os


def data_preprocess():
    ## Normalization adapted for CIFAR10
    normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    # Transforms is a list of transformations applied on the 'raw' dataset before the data is fed to the network. 
    # Here, Data augmentation (RandomCrop and Horizontal Flip) are applied to each batch, differently at each epoch, on the training set data only
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_scratch,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize_scratch,
    ])

    ### The data from CIFAR10 will be downloaded in the following folder
    rootdir = './data/cifar10'

    c10train = CIFAR10(rootdir,train=True,download=True,transform=transform_train)
    c10test = CIFAR10(rootdir,train=False,download=True,transform=transform_test)



    trainloader = DataLoader(c10train,batch_size=32,shuffle=True)

    #mixup
  
    mixup=v2.MixUp(num_classes=10)

    testloader = DataLoader(c10test,batch_size=32) 

    ## number of target samples for the final dataset
    num_train_examples = len(c10train)
    num_samples_subset = 40000


    seed  = 2147483647

    ## Generate a list of shuffled indices ; with the fixed seed, the permutation will always be the same, for reproducibility
    indices = list(range(num_train_examples))
    np.random.RandomState(seed=seed).shuffle(indices)## modifies the list in place

    ## We define the Subset using the generated indices 

    # Finally we can define anoter dataloader for the training data
    trainloader = DataLoader(c10train,batch_size=32,shuffle=True)

    return trainloader, testloader, mixup