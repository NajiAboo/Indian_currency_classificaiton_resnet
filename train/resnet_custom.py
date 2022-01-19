import torchvision

from torch import device, optim
import torch.nn as nn
import torchvision
import preprocessing as pp
import train
import util
from torch.optim import lr_scheduler


model_conv = torchvision.models.resnet50(pretrained=False, num_classes=7)


model_conv = model_conv.to(pp.device)

loss_fn = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.0001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv,gamma=0.1,step_size=4)

model_ft = train.train(model_conv,optimizer_conv, loss_fn, exp_lr_scheduler,pp.device, pp.image_dataloaders, dataset_size=pp.dataset_size,epochs=200 )





# import torch
# model_dict = torch.load('model/final.model')
# model_conv.load_state_dict(model_dict)


# #torch.save(model.state_dict(),"model/final.model")
# util.visualize_model(model_conv,pp.image_dataloaders,pp.device, pp.class_names,6)
