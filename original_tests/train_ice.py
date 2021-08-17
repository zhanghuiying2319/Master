import os,sys,math,numpy as np, matplotlib.pyplot as plt

import torch
import torch.utils.data

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader

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

class icedataset_train_uresize(torch.utils.data.Dataset):
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

    #pad
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
    
    '''
    clslabel  = torch.nonzero(self.labels[idx,:], as_tuple=True)[0]
    if clslabel.shape[0]>1:
      print('err')
      exit()
    sample = [ img , clslabel[0]  ,idx]
    '''
    
    sample = [ img , self.labels[idx,:]  ,idx]
    return sample
    
  def __len__(self):
    return self.labels.shape[0]
  

def train_epoch(model,  trainloader,  criterion, device, optimizer ):

    model.train()
 
    lf = torch.nn.CrossEntropyLoss()
 
    losses = []
    for batch_idx, data in enumerate(trainloader):

        inputs=data[0].to(device)
        labels=data[1].to(device)
        
        optimizer.zero_grad()

        outputs = model(inputs)
        
        if batch_idx ==0:
          print(outputs.shape,labels.shape)
        
        slab = torch.nonzero(labels,as_tuple=True)[1]
        loss =  lf(outputs, slab)
        #loss = torch.mean( torch.nn.functional.log_softmax(outputs, dim=1)*labels)
        #loss = criterion(output, labels)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        
        if (batch_idx%100==0) and (batch_idx>=100):
          print('at batchindex: ',batch_idx,' mean of losses: ',np.mean(losses))

    return np.mean(losses)


def evaluate2(model, dataloader, criterion, device, numcl):

    model.eval()

    lf = torch.nn.CrossEntropyLoss()
    
    classcounts = torch.zeros(numcl)
    confusion_matrix = torch.zeros(numcl, numcl)
    
    with torch.no_grad():
      losses = []
      for batch_idx, data in enumerate(dataloader):
      
      
        if (batch_idx%100==0) and (batch_idx>=100):
          print('at val batchindex: ',batch_idx)
    
        inputs = data[0].to(device)        
        outputs = model(inputs)

        labels = data[1]

        #loss = criterion(outputs, labels.to(device) )
        #loss = torch.mean( torch.nn.functional.log_softmax(outputs, dim=1)*labels.to(device))
        
        slab = torch.nonzero(labels,as_tuple=True)[1].to(device)   
        loss =  lf(outputs, slab)
        
        losses.append(loss.item())

        cpuout= outputs.to('cpu')
        _, preds = torch.max(cpuout, 1)

        for si in range(labels.shape[0]):
          inds = torch.nonzero(labels[si,:],as_tuple=True)
          confusion_matrix[inds[0],preds[si].long()]+=1
        classcounts+=torch.sum(labels,dim=0)
          

      globalacc=0
      for c in range(numcl):
        globalacc+=    confusion_matrix[c,c]
      globalacc/=torch.sum(classcounts)    
      
      cwacc = confusion_matrix.diag() / classcounts
    
    return globalacc, cwacc, confusion_matrix, classcounts, np.mean(losses)


def traineval2_nocv_notest(dataloader_train, dataloader_val ,  model ,  criterion, optimizer, scheduler, num_epochs, device, numcl):

  best_measure = 0
  best_epoch =-1

  trainlosses=[]

  testperfs1=[]
  testperfs2=[]
  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    model.train(True)
    avgloss=train_epoch(model,  dataloader_train,  criterion,  device , optimizer )
    trainlosses.append(avgloss)
    
    if scheduler is not None:
      scheduler.step()

    model.train(False)
    globalacc, cwacc, confusion_matrix, classcounts , avgvalloss = evaluate2(model, dataloader_val, criterion, device, numcl)

    print(avgloss,avgvalloss)

    testperfs1.append(globalacc)
    testperfs2.append(cwacc)
    
    print('at epoch: ', epoch,' classwise perfmeasure ', torch.mean(cwacc).item())
    print('at epoch: ', epoch,' acc ', globalacc.item())
    
    avgperfmeasure = globalacc #torch.mean(cwacc)
    print('at epoch: ', epoch,' avgperfmeasure ', avgperfmeasure.item())

    if avgperfmeasure > best_measure: #higher is better or lower is better?
      bestweights= model.state_dict()
      best_measure = avgperfmeasure
      best_epoch = epoch
      
      bestcwacc = cwacc
      bestacc = globalacc

      '''
      savept= './scores'
      if not os.path.isdir(savept):
        os.makedirs(savept)
      np.save(os.path.join(savept,'predsv1.npy'), cwacc.cpu().numpy() )
      '''
      
    print('current best', best_measure.item(), ' at epoch ', best_epoch)   
       
  return best_epoch,bestcwacc, bestacc , bestweights


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
  config['splitpath']='icesplits_v2_09052021'
  
  config['use_gpu'] = True
#change lr here!!!!!!!!!!!!!!!!!
  config['lr']=0.0001
  config['batchsize_train'] = 32
  config['batchsize_val'] = 64
  
  config['maxnumepochs'] = 35

  config['scheduler_stepsize']=15
  config['scheduler_factor']=0.3


  

  
  #data augmentations
  data_transforms = {
      'train': transforms.Compose([
          #transforms.Resize(128),
          #transforms.RandomCrop(128),
          transforms.RandomHorizontalFlip(),
          transforms.RandomVerticalFlip(),
          transforms.RandomRotation(degrees=180 ),#, interpolation= "bilinear"), 
          transforms.Normalize([0.111, 0.111, 0.111], [0.14565, 0.14565, 0.14565])
      ]),
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
              

 
  dataset_trainval = icedataset_train_uresize('./test_withBoundaries_new_Julie.npy', transforms = data_transforms['train']  )
  dataset_test = icedataset_train_uresize('./test_withBoundaries_new_Julie.npy', transforms =  data_transforms['test'])
  

  #cvind = int(sys.argv[1])
  
  for cvind in range(config['numcv']):
  
    sampler_train = getrandomsampler_innersplit5to2simple(outpath = config['splitpath'] ,numcv = config['numcv'], outercvind = cvind, trainvalortest ='train')
    dataloader_train = torch.utils.data.DataLoader(dataset = dataset_trainval, batch_size= config['batchsize_train'], shuffle=False, sampler=sampler_train, batch_sampler=None, num_workers=0, collate_fn=None)
     
    sampler_val = getrandomsampler_innersplit5to2simple(outpath = config['splitpath'] ,numcv = config['numcv'], outercvind = cvind, trainvalortest ='val')
    dataloader_val = torch.utils.data.DataLoader(dataset = dataset_trainval, batch_size= config['batchsize_val'], shuffle=False, sampler=sampler_val, batch_sampler=None, num_workers=0, collate_fn=None)


    sampler_test = getrandomsampler_innersplit5to2simple(outpath = config['splitpath'] ,numcv = config['numcv'], outercvind = cvind, trainvalortest ='test')
    dataloader_test = torch.utils.data.DataLoader(dataset = dataset_test, batch_size= config['batchsize_val'], shuffle=False, sampler=sampler_test, batch_sampler=None, num_workers=0, collate_fn=None)

    #change different pre-tranined model here!!!!!!!!!!!!!!!
    model = models.densenet121(pretrained=True)
    #resnet
   # num_ft = model.fc.in_features
   # model.fc = nn.Linear(num_ft, config['numcl'])  
    #densenet
    num_ft = model.classifier.in_features
    model.classifier = nn.Linear(num_ft, config['numcl'])
    model = model.to(device)
    #change SGD here! !!!!!!!!!!!!!!!!!!!!!
    #Adam
    #optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    #AdamW
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    #SGD
    #optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    #freezing some layers
    #trainable_params=[]
    #for nam,lay in model.named_parameters():
    #    print(nam)
    #    if 'classifier' in nam:
    #        trainable_params.append(lay)
    #    elif 'denseblock3' in nam:
    #        trainable_params.append(lay)
    #    elif 'denseblock3' in nam:
    #        trainable_params.append(lay)
    #    elif 'features.norm5' in nam:
    #        trainable_params.append(lay)
    #optimizer = optim.SGD(trainable_params, lr=config['lr'], momentum=0.9)
    #optimizer = optim.Adam(trainable_params, lr=config['lr'])
    #optimizer = optim.AdamW(trainable_params, lr=config['lr'])
    somelr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler_stepsize'], gamma= config['scheduler_factor'])
    
    best_epoch,bestcwacc, bestacc , bestweights = traineval2_nocv_notest(dataloader_train, dataloader_val ,  model ,  criterion = None, optimizer = optimizer , scheduler = somelr_scheduler, num_epochs = config['maxnumepochs'] , device = device , numcl = config['numcl'] )
    
    #test evaluation
    model.load_state_dict(bestweights)
    globalacc, cwacc, confusion_matrix, classcounts, testlossavg = evaluate2(model, dataloader_test, criterion = None, device = device, numcl  = config['numcl'] )
    
    print('test eval class-wise acc', torch.mean(cwacc.cpu()).item() )
    print('test eval global acc', globalacc.cpu().item() )
    # save results
    savept= './scores_v2_10052021_densenet121_AdamW_lr10-5_lay34clfn5'
    if not os.path.isdir(savept):
      os.makedirs(savept)
      
    np.save(os.path.join(savept,'cwacc_outercv{:d}.npy'.format(cvind)), cwacc.cpu().numpy() )
    np.save(os.path.join(savept,'globalacc_outercv{:d}.npy'.format(cvind)), globalacc.cpu().numpy() )
    np.save(os.path.join(savept,'confusion_matrix_outercv{:d}.npy'.format(cvind)), confusion_matrix.cpu().numpy() )
    torch.save(bestweights, os.path.join(savept,'bestweights_outercv{:d}.pt'.format(cvind)) )
    
if __name__=='__main__':
  runstuff()  
  
  
  
