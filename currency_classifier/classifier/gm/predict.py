import torch
import torchvision
from classifier.gm import  preprocessing as pp
import torch.nn as nn
from PIL import Image
# ['10', '100', '20', '200', '2000', '50', '500']
def predict(input_img):
    model_dict = torch.load('classifier/gm/final.pt', map_location=pp.device)
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
    print("here -- line 38")
    img = Image.open(input_img)
    #print(pp.transforms)
    img_t = pp.transformers['val'](img)
    batch_t = torch.unsqueeze(img_t, 0)

    #img = torch.from_numpy(img).type(torch.FloatTensor) 
    print("here -- line 40")
    print(pp.device)
    batch_t = batch_t.to(pp.device)
    
    model_conv.eval()
    outputs = model_conv(batch_t)
    _, preds = torch.max(outputs, 1)
    # print(pp.class_names)
    # print(pp.class_names[preds[0]])
    # print("reuslt")
    # print(preds)
    return pp.class_names[preds[0]]
 