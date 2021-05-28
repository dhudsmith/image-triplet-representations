import numpy as np
from sklearn.metrics import roc_auc_score

def getAUROC_filtered(outputs, targets):
  ix_keep = targets!=0.5
  filteredOutputs = outputs[ix_keep]
  filteredTargets = targets[ix_keep]
  return roc_auc_score(filteredTargets.cpu().numpy(), filteredOutputs.cpu().numpy())

def getAUROC_all(outputs, targets):
    predictions = outputs.cpu().numpy()
    targets = targets.cpu().numpy()
    binned_predictions = []
    binned_targets = []
    
    bins = [0, 1/2, 1]
    bin_indices = np.digitize(predictions, bins)
    for index in bin_indices:
        if index == 1:
            binned_predictions.append(0)
        elif index == 2:
            binned_predictions.append(1)
            
    bin_indices = np.digitize(targets, bins)
    for target in targets:
        if target == 0:
            binned_targets.append(0)
        elif target == 0.5:
            binned_targets.append(1)
        elif target == 1:
            binned_targets.append(1)
  
    return roc_auc_score(binned_targets, binned_predictions)

def getAccuracy_filtered(outputs, targets):
    ix_keep = targets!=0.5
    filteredOutputs = outputs[ix_keep]
    filteredTargets = targets[ix_keep]
    filteredOutputs = filteredOutputs.cpu().numpy()
    filteredTargets = filteredTargets.cpu().numpy()
        
    bins = [0, 1/2, 1]
    bin_indices = np.digitize(filteredOutputs, bins)
    binned_outputs = []
    for index in bin_indices:
        if index == 1:
            binned_outputs.append(0)
        elif index == 2:
            binned_outputs.append(1)

    correct = 0;
    results = np.equal(binned_outputs, filteredTargets)
    for result in results:
        if result:
            correct += 1
  
    return (correct / filteredTargets.size)

def getAccuracy_all(predictions, targets):
  predictions = predictions.cpu().numpy()
  targets = targets.cpu().numpy()
  new_predictions = []
  correct = 0

  bins = [0, 1/3, 2/3, 1]
  bin_indices = np.digitize(predictions, bins)
  for index in bin_indices:
    if index == 1:
      new_predictions.append(0)
    elif index == 2:
      new_predictions.append(0.5)
    elif index == 3:
      new_predictions.append(1)
  
  results = np.equal(new_predictions,targets)
  for result in results:
    if result:
      correct += 1
  
  return (correct / predictions.size)