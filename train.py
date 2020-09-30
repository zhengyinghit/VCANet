# -*- coding: UTF-8 -*-
from __future__ import print_function, division

import copy
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torchvision.models.resnet import resnet50 as resnet_vc
from vcanet import resnet50, resnet101, resnet152, resnext101_32x8d
import util

gpu_id = '0'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'sketch_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=20,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
dset_classes = image_datasets['train'].classes


# generate the attention map of visual chirality
def return_vcattention(feature_conv, weights, prediction):
    bs, nc, h, w = feature_conv.shape

    vc_att = [torch.mm(weights[prediction[i]].unsqueeze(0), feature_conv[i].reshape((nc, h * w)))
              for i in range(bs)]

    # normalize to [0, 1]
    vc_att = torch.stack([(vc_att[i] - torch.min(vc_att[i])) / torch.max(vc_att[i]) for i in range(bs)])

    # reshape the attention map to h x w
    vc_att = vc_att.reshape(bs, 1, h, w)

    return vc_att


def train_model(model, model_vc, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # hook the feature extractor
    features_blobs = []

    def hook_feature(module, input, output):
        # features_blobs.append(output.data.cpu().numpy())
        features_blobs.append(output)

    # get the output feature map of layer4
    model_vc._modules.get('layer4').register_forward_hook(hook_feature)

    # get weights of the softmax layer
    params = list(model_vc.parameters())  # transform the parameters to list
    weights = params[-2]

    best_model = model
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase.
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            metric_logger = util.MetricLogger(delimiter="  ")
            metric_logger.add_meter('lr', util.SmoothedValue(window_size=1, fmt='{value}'))
            metric_logger.add_meter('img/s', util.SmoothedValue(window_size=10, fmt='{value}'))

            header = 'Epoch: [{}]'.format(epoch)

            # Iterate over data
            print_freq = 5
            for inputs, labels in metric_logger.log_every(dataloaders[phase], print_freq, header):
                start_time = time.time()
                inputs = inputs.cuda()

                labels = labels.cuda()

                with torch.no_grad():
                    outputs_vc = model_vc(inputs)
                    _, prediction_vc = torch.max(outputs_vc, 1)

                    # generate the attention map of visual chirality for each prediction
                    vc_att = return_vcattention(features_blobs[0], weights, prediction_vc)

                    # clear the extracted feature map
                    features_blobs = []

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs, vc_att)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Print the logger info
                batch_size = inputs.shape[0]
                metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
                metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # Deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)

        print()

    # save the best model to local file
    save_path = 'vcanet_resnet50.pth'
    torch.save(best_model.state_dict(), save_path)

    time_elapsed = time.time() - since
    mes1 = 'Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60)
    print(mes1)
    mes2 = 'Best val Acc: {:4f}'.format(best_acc)
    print(mes2)


def main():
    # load the model of visual chirality
    model_path = 'resnet50_vc.pth'

    model_vc = resnet_vc(pretrained=False)
    num_ftrs = model_vc.fc.in_features

    model_vc.fc = nn.Linear(num_ftrs, 2)

    model_vc.load_state_dict(torch.load(model_path))

    model_vc = model_vc.cuda()

    model_vc.eval()

    # the model of sketch classification
    model_ft = resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features

    model_ft.fc = nn.Linear(num_ftrs, dset_classes.__len__())

    model_ft = model_ft.cuda()

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized.
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs.
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # Training
    train_model(model_ft, model_vc, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)


if __name__ == "__main__":

    main()
