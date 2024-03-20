from data_preprocessing import data_preprocessing
from models_cifar import resnet
import loss_distillation
from tqdm import tqdm
import os
import signal
import time
import torch

def train_distillation(teacher, schedulLR, epochs, initial_learning_rate, distillation_weight, temperature):

    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    accuracies=[]
    best_accuracy = 0
    best_epoch = 0

    trainloader, testloader, mixup = data_preprocessing.data_preprocess()

    student = resnet.Resnet18_fact()

    student.to(device)
    teacher.device()

    student.train()


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate)
    loss_calculator = loss_distillation.LossCalulcator(temperature, distillation_weight).cuda()

    if schedulLR=='plateau':
        scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    elif schedulLR=='cosine':
        scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, 130, 0.0001)


    if teacher is not None:
        teacher.eval()
    
    signal.signal(signal.SIGINT, interrupt_handler)

    for epoch in range(1, epochs+1):
        # train one epoch
        student.train()

        for i, data in tqdm(enumerate(dataloader, 1)):

            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = student(inputs)

            teacher_outputs = None
            if teacher is not None and distillation_weight > 0.0:
                with torch.no_grad():
                    teacher_outputs = teacher(inputs)

            loss = loss_calculator(outputs          = outputs,
                                    labels           = labels,
                                    teacher_outputs  = teacher_outputs)
            loss.backward()
            optimizer.step()

        # validate the network

        accuracy = measure_accuracy(student, testloader, device)
        accuracies.append(accuracy)
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch

        # learning rate schenduling
        scheduler.step()

        # print log

        print("%s: Epoch [%3d/%3d], Iteration [%5d/%5d], Loss [%s]"%(time.ctime(),
                                                                         epoch,
                                                                         epochs,
                                                                         i,
                                                                         len(dataloader),
                                                                         loss_calculator.get_log()))

        
    PATH = './result/chkp/resnet_student.pth'
    torch.save(model.state_dict(), PATH)
    print("Finished Training, Best Accuracy: %f (at %d epochs)"%(best_accuracy, best_epoch))
    return accuracies






def measure_accuracy(model, dataloader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().cpu().item()

    print("Accuracy of the network on the 10000 test images: %f %%"%(100 * correct / total))

    return correct / total



def interrupt_handler(sig, frame):
    print('Interrupt detected, Saving model...')
    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    })
    print('model saved')
    exit(0)

def save_checkpoint(state, filename="checkpoint_student.pth.tar"):
    torch.save(state,filename)
    