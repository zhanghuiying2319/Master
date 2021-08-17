import os,sys,math,numpy as np, matplotlib.pyplot as plt

import torch
import torch.utils.data

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader

import scipy.ndimage
import cv2

import random

seedv=9

random.seed(seedv)
np.random.seed(seedv)
torch.manual_seed(seedv)

def loadindexlist(outpath,numcv):

  indexlist=[ [] for i in range(numcv) ]

  for cv in range(numcv):  
    fname= os.path.join(outpath,'split_cv'+str(cv)+'.txt')  
    with open(fname,'r') as f:
      line = f.readline().rstrip().split()
    for k in line:
      indexlist[cv].append(int(k))
      
  return indexlist


def getrandomsampler_innersplit5to2simple(outpath,numcv, outercvind, trainvalortest):

  indexlist = loadindexlist(outpath,numcv)
  
  
  trainvalcvinds= [ cvi for cvi in range(numcv) if cvi !=outercvind]
  if trainvalortest == 'train':
    
    indices=[]
    for cvinds in trainvalcvinds[:numcv-3]:
      indices.extend( indexlist[cvinds] )
    
  elif trainvalortest == 'val':
  
    indices=[]
    for cvinds in trainvalcvinds[numcv-3:]:
      indices.extend( indexlist[cvinds] ) 
       
  elif trainvalortest == 'test':
    indices = indexlist[outercvind]
    
  else:
    print('unknown trainvalortest', trainvalortest)
    exit()
    
  sampler = torch.utils.data.SubsetRandomSampler(indices)
  return sampler

def getrandomsampler_innercv(outpath,numcv, outercvind, innercvind, trainvalortest):

  indexlist = loadindexlist(outpath,numcv)
  
  
  trainvalcvinds= [ cvi for cvi in range(numcv) if cvi !=outercvind]
  if trainvalortest == 'train':
    
    indices=[]
    for cvi in range(numcv-1):
      if cvi !=innercvind:
        indices.extend(  indexlist[ trainvalcvinds[cvi] ] )
    
  elif trainvalortest == 'val':
  
    indices = indexlist[trainvalcvinds[innercvind]]

      
  elif trainvalortest == 'test':
    indices = indexlist[outercvind]
    
  else:
    print('unknown trainvalortest', trainvalortest)
    exit()
  
  #https://www.youtube.com/watch?v=CFpGXUxXk2g  
  
  sampler = torch.utils.data.SubsetRandomSampler(indices)
  return sampler
  
#for better perf  
#https://github.com/hcarlens/pytorch-tabular/blob/master/fast_tensor_data_loader.py

#train: random flip h/v , random rotation
# also grayscale mod?

#eval: return 2 flips times n rotations for avg classif
# eval simple: return img


def multrot(imgtensor,numrotations, effsize, transforms = None):


  meanval=0.0 #0.0 #0.111 for fmean models

  stackedimg = meanval*torch.ones((2*numrotations,3, effsize, effsize) )# 2*numrot for h-flips

  counter=-1
  for deg in np.linspace(-180,+180,numrotations, endpoint=False):
    counter+=1
    
    img = imgtensor.clone() 
    rotimg = torch.from_numpy(scipy.ndimage.rotate(img, deg, reshape=False))  #reshape=True))

    d1=rotimg.shape[0]
    d2=rotimg.shape[1]
    
    t = meanval*torch.ones((1, effsize, effsize) )
    offset1 = (effsize- d1)//2
    offset2 = (effsize- d2)//2
    t[0,offset1:offset1+d1,offset2:offset2+d2] = rotimg
    img = torch.cat ( (  t,t,t  ),dim=0 )

    if transforms:
      img = transforms(img)
  
    #2 mirrors here
    stackedimg[counter]= img
    counter+=1
    stackedimg[counter]=torch.flip(img, dims=[2])

  return stackedimg

class icedataset_testrot_uresize(torch.utils.data.Dataset):
  def __init__(self, thenumpyfile, transforms ):
  
    self.transforms=transforms
    
    self.uniformresize = 128 # a bit above 80% quant
    # alternative: embed into 224x224 with padding, and corner / random position crop if test image is too large over 224
  
    numcl =9
    with open(thenumpyfile,'rb') as f:
      a = np.load(f,allow_pickle=True)
      b = np.load(f,allow_pickle=True)
      c = np.load(f,allow_pickle=True)
  
    #self.rawimgslist=[]
    #for l in range(a.shape[0]):
    #  self.rawimgslist.append( torch.tensor(a[l]) )
    
    self.processedimgslist=[]
    for l in range(a.shape[0]):
      
      d1=a[l].shape[0]
      d2=a[l].shape[1]
      
      if d1>d2:
        dsize=(self.uniformresize,  int( round( self.uniformresize*d2/d1  ) )  )
      else:
        dsize= (  int( round( self.uniformresize*d1/d2  ) ) , self.uniformresize  )
      resizedimg = cv2.resize( a[l], dsize= dsize )
      self.processedimgslist.append( torch.tensor(resizedimg)  )
    
    
    
    labels = np.zeros((c.shape[0],numcl), dtype = np.int32)
    for l in range(c.shape[0]):
      labels[l,:]=c[l]
      
    self.labels=torch.IntTensor(labels)
    
  def __getitem__(self,idx):

    '''
    numrotations=8
    #pad #
    effsize = self.uniformresize #int(self.uniformresize*1.42)
    
    imgtensor = self.processedimgslist[idx]
    stackedimg = multrot(imgtensor,numrotations, effsize, transforms = self.transforms)
    '''
    t = torch.zeros((1, self.uniformresize, self.uniformresize) )

    d1=self.processedimgslist[idx].shape[0]
    d2=self.processedimgslist[idx].shape[1]
    
    if d1 > d2:
      offset = (self.uniformresize- d2)//2
      t[0,:,offset:offset+self.processedimgslist[idx].shape[1]] = self.processedimgslist[idx].clone()  
    else:  
      offset = (self.uniformresize- d1)//2
      t[0,offset:offset+self.processedimgslist[idx].shape[0],:] = self.processedimgslist[idx].clone()  
    img = torch.cat ( (  t,t,t  ),dim=0 )
    
    if self.transforms:
      img = self.transforms(img)
      
    sample = [ img , self.labels[idx,:]  ,idx]
    return sample
    
  def __len__(self):
    return self.labels.shape[0]
  


def evaluate_augavg(model, dataloader, criterion, device, numcl):

    model.eval()

    lf = torch.nn.CrossEntropyLoss()
    
    classcounts = torch.zeros(numcl)
    confusion_matrix = torch.zeros(numcl, numcl)
    
    with torch.no_grad():
      losses = []
      globalacc2 = 0
      nsamples = 0
      for batch_idx, data in enumerate(dataloader):
      
      
        if (batch_idx%100==0) and (batch_idx>=100):
          print('at val batchindex: ',batch_idx)
    
        inputs = data[0].to(device)        


        labels = data[1]
        slab = torch.nonzero(labels,as_tuple=True)[1].to(device)   


        avgcpuout = None
        avgloss = None
        for viewind in range(1): # loops over all rotations
        
          # inputs.shape = (batchsize, differentrotationsofthesameimage,3colorchannels,h,w)
          
          outputs = model(inputs[:,:,:,:])
          loss =  lf(outputs, slab)
          cpuout= outputs.to('cpu')
          avgcpuout = cpuout
          avgloss = loss
          '''
          if avgcpuout is None:
            avgcpuout = cpuout / inputs.shape[1]
            avgloss = loss / inputs.shape[1]
          else:
            avgcpuout += cpuout / inputs.shape[1]
            avgloss += loss / inputs.shape[1]
          '''          
        losses.append(avgloss.item())
        _, preds = torch.max(avgcpuout, 1)

        for si in range(labels.shape[0]):
          inds = torch.nonzero(labels[si,:],as_tuple=True)
          confusion_matrix[inds[0],preds[si].long()]+=1
        classcounts+=torch.sum(labels,dim=0)

        for si in range(labels.shape[0]):
          lbinds = torch.nonzero(labels[si,:],as_tuple=True)
          globalacc2+= torch.sum( preds[si]==lbinds[0] )
        nsamples += labels.shape[0]
        
      globalacc = globalacc2/ float(nsamples)  
          
      '''
      globalacc=0
      for c in range(numcl):
        globalacc+=    confusion_matrix[c,c]
      globalacc/=torch.sum(classcounts)    
      '''
      cwacc = confusion_matrix.diag() / classcounts
    
    return globalacc, cwacc, confusion_matrix, classcounts, np.mean(losses)




def runstuff():

  '''
  if len(sys.argv)!=2:
    print('len(sys.argv)!=2', len(sys.argv))
    exit()
  '''
  config = dict()
  
  
  # kind of a dataset property
  config['numcl']=9
  config['numcv']=10
  config['splitpath']='icesplits_v4_10052021'
  
  config['use_gpu'] = True
  config['batchsize_val'] = 64

  

  
  #data augmentations
  data_transforms = {
      'test': transforms.Compose([

          #transforms.Resize(128),
          #transforms.CenterCrop(128),
          transforms.Normalize([0.111, 0.111, 0.111], [0.14565, 0.14565, 0.14565])
      ]),
  }

  if True == config['use_gpu']:
    device= torch.device('cuda:0')
  else:
    device= torch.device('cpu')
              
  dataset_test = icedataset_testrot_uresize('./test_withBoundaries_new_Julie.npy', transforms =  data_transforms['test'])
  #savept= './scores_v4_10052021'  
  #savept= './scores_v4_10052021_rebal0half_densenet121_AdamW_10-4_equalproba'  
  savept= './icesplits_v4_10052021_balanced_aspectratios_densenet121_SGD'
  for cvind in range(config['numcv']):
  
    sampler_test = getrandomsampler_innersplit5to2simple(outpath = config['splitpath'] ,numcv = config['numcv'], outercvind = cvind, trainvalortest ='test')
    dataloader_test = torch.utils.data.DataLoader(dataset = dataset_test, batch_size= config['batchsize_val'], shuffle=False, sampler=sampler_test, batch_sampler=None, num_workers=0, collate_fn=None)


    model = models.densenet121(pretrained=False)
    #DenseNet
    num_ft = model.classifier.in_features
    model.classifier = nn.Linear(num_ft, config['numcl'])
    #model.classifier = nn.Linear(num_ft+1, config['numcl'])        
     #ResNet
     #num_ft = model.fc.in_features
     #model.fc = nn.Linear(num_ft, config['numcl'])
    

    savedweights = torch.load( os.path.join(savept,'bestweights_outercv{:d}.pt'.format(cvind)) )
    model.load_state_dict(savedweights)
    
    model = model.to(device)

    globalacc, cwacc, confusion_matrix, classcounts, testlossavg = evaluate_augavg(model, dataloader_test, criterion = None, device = device, numcl  = config['numcl'] )
    
    print('test eval class-wise acc', torch.mean(cwacc.cpu()).item() )
    print('test eval global acc', globalacc.cpu().item() )
    # save results
    np.save(os.path.join(savept,'noavg_cwacc_outercv{:d}.npy'.format(cvind)), cwacc.cpu().numpy() )
    np.save(os.path.join(savept,'noavg_globalacc_outercv{:d}.npy'.format(cvind)), globalacc.cpu().numpy() )
    np.save(os.path.join(savept,'noavg_confusion_matrix_outercv{:d}.npy'.format(cvind)), confusion_matrix.cpu().numpy() )

if __name__=='__main__':
  runstuff()  
  
  
  
