import torch
import torch.nn as nn

class Action_Conditioned_FF(nn.Module):
    def __init__(self):
        super().__init__()
# STUDENTS: __init__() must initiatize nn.Module and define your network's
# custom architecture
        input_dim = 6
        output_dim = 1 
        # Linear function
        self.layer1 = nn.Linear(input_dim, 40) 
        # Linear function 
        self.layer2 = nn.Linear(40,40)  
        
        self.layer3 = nn.Linear(40, output_dim)    


    def forward(self, input):
# STUDENTS: forward() must complete a single forward pass through your network
# and return the output which should be a tensor

        out = self.layer1(input)
        out = torch.relu(out)
        out = self.layer2(out)
        output = torch.sigmoid(self.layer3(out))

        return output


    def evaluate(self, model, test_loader, loss_function):
# STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
# mind that we do not need to keep track of any gradients while evaluating the
# model. loss_function will be a PyTorch loss function which takes as argument the model's
# output and the desired output.
        for indx,batchdict in enumerate(test_loader):
          
              in_batch = batchdict['input']
                 #need to get Tensor from 'Inputs' --> model(inputs)
              pred = torch.flatten(model(in_batch))
   
              label_batch = batchdict['label']
             
              loss = loss_function(pred, label_batch)  #need to use Tensor of 'Labels' here

              loss_val = loss.item()

        return loss_val 

def main():
    model = Action_Conditioned_FF()

if __name__ == '__main__':
    main()