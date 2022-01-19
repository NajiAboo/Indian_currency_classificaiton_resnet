from torch import device, optim
import torch.nn as nn
import torchvision
import preprocessing as pp
import train
import util
from torch.optim import lr_scheduler



model_conv = torchvision.models.resnet18(pretrained=True)

    
for param in model_conv.parameters():
        param.requires_grad = False
        
for param in model_conv.fc.parameters():
        param.requires_grad = True
    
num_ftrs = model_conv.fc.in_features

# define the network head and attach it to the model
headModel = nn.Sequential(
	nn.Linear(num_ftrs, 512),
	nn.ReLU(),
	nn.Dropout(0.25),
	nn.Linear(512, 256),
	nn.ReLU(),
	nn.Dropout(0.5),
 	nn.Linear(256, 128),
  	nn.ReLU(),
   	nn.Dropout(0.5),
   	nn.Linear(128, 64),
    nn.ReLU(),
	nn.Linear(64, 7)
)

model_conv.fc = headModel


model_conv = model_conv.to(pp.device)

loss_fn = nn.CrossEntropyLoss()
optimizer_conv = optim.Adam(model_conv.fc.parameters(), lr=0.001)


model_ft = train.train(model_conv,optimizer_conv, loss_fn, pp.device, pp.image_dataloaders, dataset_size=pp.dataset_size,epochs=1000 )





# import torch
# model_dict = torch.load('model/final.model')
# model_conv.load_state_dict(model_dict)


# #torch.save(model.state_dict(),"model/final.model")
# util.visualize_model(model_conv,pp.image_dataloaders,pp.device, pp.class_names,6)
