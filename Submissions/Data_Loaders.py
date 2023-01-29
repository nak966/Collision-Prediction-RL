import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.model_selection import train_test_split #added.. for now


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
       #*********UNCOMMENT self.data = np.genfromtxt('saved/training_data.csv', delimiter=',')
      #OLDNK self.data = np.genfromtxt('saved/testsheet.csv', delimiter=',')
      #2nd: self.data = np.genfromtxt('saved/2500samps.csv', delimiter=',')
       self.data = np.genfromtxt('saved/TODO REDUCE SAMPLE training_data.csv', delimiter=',')
       
# STUDENTS: it may be helpful for the final part to balance the distribution of your collected data
       self.data = np.float32(self.data) #TEST

        # normalize data and save scaler for inference
       self.scaler = MinMaxScaler()
       self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms
       #pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference
       torch.save(self.scaler, 'saved/scaler.pkl',_use_new_zipfile_serialization=False)

    def __len__(self):
# STUDENTS: __len__() returns the length of the dataset
        return(len(self.data))
        #pass

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
# STUDENTS: for this example, __getitem__() must return a dict with entries {'input': x, 'label': y}
# x and y should both be of type float32. There are many other ways to do this, but to work with autograding
# please do not deviate from these specifications.

        x = self.normalized_data[idx][0:6] #.NORMALIZED OR DATA?? 
        y = self.normalized_data[idx][-1]
        returndict = {'input':x,'label':y}   
        return(returndict)


class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
# STUDENTS: randomly split dataset into two data.DataLoaders, self.train_loader and self.test_loader
# make sure your split can handle an arbitrary number of samples in the dataset as this may vary

        outputs = self.nav_dataset.data[0:len(self.nav_dataset),-1] #should be Last column of Dataset
 
    #*****FINAL ISSUE******: is splitting Train & Test to Keep 0's & 1's 'balanced' or something.. 
          #if using 'RandomSplit' instead (which'll not require the extra import pkg)..
          # then you need to maks sure original Dataset is Balanced first & then the Split will be Balaned too
        trainers,testers = train_test_split(self.nav_dataset,stratify=outputs) #This works BUT now idk how to get Stratify to WOrk!
#also need to split 80/20 probably?                          #need to make sure Stratify is Working && that it takes in correct Mix Array
#        print(trainers)

        self.train_loader = data.DataLoader(trainers, batch_size=batch_size, shuffle=True)
            #THIS works, meaning that the DataLoader has to WrapAround the DATASET class.. somehow!
        self.test_loader = data.DataLoader(testers, batch_size=batch_size, shuffle=True)

def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    # STUDENTS : note this is how the dataloaders will be iterated over, and cannot be deviated from
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']

if __name__ == '__main__':
    main()
