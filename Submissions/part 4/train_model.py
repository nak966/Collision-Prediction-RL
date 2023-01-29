from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def train_model(no_epochs):

    batch_size = 5  
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()

    #added by me
    loss_function=torch.nn.BCELoss()
    optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
    # --------

    losses = []
    min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    losses.append(min_loss)
    

    for epoch_i in range(no_epochs):
        model.train()
        for idx, sample in enumerate(data_loaders.train_loader): # sample['input'] and sample['label']

              optimizer.zero_grad()

              in_batch = batchdict['input']
              pred = torch.flatten(model(in_batch))

              label_batch = batchdict['label']
 
              loss = loss_function(pred, label_batch)  #need to use Tensor of 'Labels' here

              losses.append(loss)
              loss.backward()

              # Update the parameters
              optimizer.step()

      torch.save(model.state_dict(), 'saved/saved_model.pkl',_use_new_zipfile_serialization=False)

if __name__ == '__main__':
    no_epochs =
    train_model(no_epochs)
