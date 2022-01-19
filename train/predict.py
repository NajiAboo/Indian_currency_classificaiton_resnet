import torch
import torchvision
import util
import preprocessing as pp
import torch.nn as nn

model_dict = torch.load('model/final.pt', map_location=pp.device)

model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
        param.requires_grad = False
        
for param in model_conv.fc.parameters():
        param.requires_grad = True

num_ftrs = model_conv.fc.in_features

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

model_conv.to(pp.device)
        
model_conv.load_state_dict(model_dict,strict=False)
#util.visualize_model(model_conv,pp.image_dataloaders,pp.device, pp.class_names,6)


# #torch.save(model.state_dict(),"model/final.model")
print(pp.class_names)
util.visualize_model(model_conv,pp.image_dataloaders,pp.device, pp.class_names,6)