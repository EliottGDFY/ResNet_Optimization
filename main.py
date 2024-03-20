
import logging
import signal
import os
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from train import train
from train_mixup import train_mixup
from train_distillation import train_distillation

def __main__():


    train_type='normal'
    initial_lr=0.1
    epochs=150
    mixup=False
    scheduler='plateau'

    logging.basicConfig(filename='logfile.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if train_type=='normal':
        
        train_loss, test_loss, model = train(epochs, 'ResNet34', initial_lr, mixup, scheduler)
    elif train_type=='mixup':
        train_loss, test_loss, model = train_mixup(epochs, 'ResNet34', initial_lr, mixup, schedulLR)
    elif train_type=='teached':
        teacher = resnet.ResNet34()
        teacher.load_state_dict(torch.load('result/chkp/resnet_teacher'))

        temperature=30
        distillation_weight=0.1

        accuracies = train_distillation(teacher, scheduler, epochs, initial_lr, distillation_weight, temperature)


    plt.plot(range(1, epochs + 1), train_loss, label='Training Loss')
    plt.plot(range(1, epochs + 1), test_loss, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Over Epochs')
    plt.legend()
    plt.savefig('./result/chkp/plot/resnet_mixup_ROP_loss_plot.png')  # Save the plot to a file
    plt.show()


__main__()