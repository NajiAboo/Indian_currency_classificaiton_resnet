import torch
import copy
import time

def train(model,optimizer: torch.optim,loss_fn, device,dataloader,dataset_size,epochs=2):
    
    since = time.time()
    best_wt = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
   
    
    for epoch in range(epochs):
        
        print(f"epoch {epoch}/{epochs-1}")
        print('-'*10)
        
        for phase in ['train','val']:
            
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0.0
            
            for inputs, target in dataloader[phase]:
                inputs = inputs.to(device)
                target = target.to(device)
                
                optimizer.zero_grad()
                
                
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    _,preds = torch.max(outputs, 1)
                    loss = loss_fn(outputs, target)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == target.data)
                #print(f"running loss : {running_loss}")
            
            if phase == 'train':
                pass
                #scheduler.step()
            
            epoch_loss = running_loss / dataset_size[phase]
            #running_corrects = running_corrects * 1.0
            epoch_accuracy = running_corrects.item() /(dataset_size[phase] * 1.0)
            
            
            
            print(f"{phase}, Loss:{epoch_loss:.4f}, acc: {epoch_accuracy:.4f}")
            
            if phase =='val' and epoch_accuracy > best_acc:
                best_acc = epoch_accuracy
                best_wt = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(),"model/final.pt")
        print()
    
    time_elapsed = time.time() - since
    
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    model.load_state_dict(best_wt)
    torch.save(model.state_dict(),"model/final.pt")
    
    return model
    
    