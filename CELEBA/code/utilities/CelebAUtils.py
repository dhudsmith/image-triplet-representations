import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
import utilities.CelebAMetrics as metrics
import utilities.CelebAModel as cm


#@title Training and testing functions
def train(model, criteria, device, loader, optimizer):
  model.train()

  mean_batch_losses = []
  for batch_idx, batch_dict in enumerate(loader):
    A, B, C, target = [batch_dict[key].to(device) for key in ["A", "B", "C", "target"]]
    optimizer.zero_grad()
    output = model(A,B,C)
    loss = criteria(output.float(), target.float())
    loss.backward()
    optimizer.step()
    mean_batch_losses.append(loss.item())
        
  return np.mean(mean_batch_losses)

            
def test(model, criteria, device, loader):
    model.eval()

    mean_batch_losses = []
    outputs = []
    targets = []

    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(loader):
          A, B, C, target = [batch_dict[key].to(device) for key in ["A", "B", "C", "target"]]
          output = model(A, B, C)
          loss = criteria(output.float(), target.float()) 

          # store results
          mean_batch_losses.append(loss.item())
          outputs.append(output)
          targets.append(target)

    outputs = torch.cat(outputs)
    targets = torch.cat(targets)
    
    return np.mean(mean_batch_losses), outputs, targets

def make_model(device, num_hidden,img_channels_shape, n_mid, n_res, kernel_size, pretrained=False, encoder_path = None):
    model = cm.TripletNet(n_hid=num_hidden,img_channels_shape=img_channels_shape,n_mid=n_mid,n_res=n_res,kernel_size=kernel_size).to(device)
    if pretrained:
        encoder_state_dict = torch.load(encoder_path)
        model.encoder.load_state_dict(encoder_state_dict)
    return model

def run_model(model,device, train_loader, test_loader, epochs, lr, gamma):
  optimizer = optim.Adadelta(model.parameters(), lr=lr)
  scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
  criteria = nn.BCELoss()

  best_test = 10**10
  best_train = 10**10
  best_auroc_filtered = 0.5
  best_auroc_all = 0.5
  best_accuracy_all = 0.5
  best_accuracy_filtered = 0.5
  best_epoch = 0
  for epoch in range(1, epochs + 1):
    train_loss = train(model, criteria, device, train_loader, optimizer)
    test_loss, outputs, targets = test(model, criteria, device, test_loader)
    scheduler.step()

    auroc_filtered = metrics.getAUROC_filtered(outputs, targets)
    auroc_all = metrics.getAUROC_all(outputs, targets)
    accuracy_filtered = metrics.getAccuracy_filtered(outputs, targets)
    accuracy_all = metrics.getAccuracy_all(outputs,targets)
    
    print("Train Loss: %0.3f. Test Loss: %0.3f. AUROC_Filtered: %0.3f. AUROC_All: %0.3f. Accuracy Filtered: %0.3f. Accuracy All: %0.3f. Epoch: %i" % (train_loss, test_loss, auroc_filtered, auroc_all, accuracy_filtered, accuracy_all, epoch))
    if test_loss<best_test:
      best_test = test_loss
      best_train = train_loss.item()
      best_auroc_filtered = auroc_filtered
      best_auroc_all = auroc_all
      best_accuracy_filtered = accuracy_filtered
      best_accuracy_all = accuracy_all
      best_epoch = epoch
    if epoch > best_epoch + 5:
      break

  #return best_test, best_auroc, best_epoch
  return best_test, best_epoch, best_accuracy_all, best_accuracy_filtered, best_auroc_filtered, best_auroc_all

