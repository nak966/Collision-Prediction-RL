import torch
import torch.nn as nn

class Action_Conditioned_FF(nn.Module):
    def __init__(self):
        super().__init__()
# STUDENTS: __init__() must initiatize nn.Module and define your network's
# custom architecture
        input_dim = 6
        output_dim = 1 #i think? 
       # super(Action_Conditioned_FF, self).__init__()
        # Linear function
        self.layer1 = nn.Linear(input_dim, 20)  #using 20 hidden dim here, maybe use 100??

        # Linear function (readout)
        self.layer2 = nn.Linear(20, output_dim)  
     #   pass

    def forward(self, input):
# STUDENTS: forward() must complete a single forward pass through your network
# and return the output which should be a tensor
        y_predicted=self.layer1(input)
        output=torch.sigmoid(self.layer2(y_predicted))
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

              # print(loss.item())
              # print(type(loss))
              # print(type(loss.item()))

              loss_val = loss.item()

        return loss_val #

def main():
    model = Action_Conditioned_FF()

if __name__ == '__main__':
    main()
