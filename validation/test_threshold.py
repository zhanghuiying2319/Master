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
  
def getrandomsampler_innersplit5to2simple_test(outpath,numcv, outercvind, trainvalortest):

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
  return sampler, indices
  
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
  

def evaluate_augavg_testonly(model, dataloader, criterion, device, numcl):

    model.eval()

    lf = torch.nn.CrossEntropyLoss()
    
    classcounts = torch.zeros(numcl)
    #confusion_matrix = torch.zeros(numcl, numcl)
    #failidx= []
    

    with torch.no_grad():
      losses = []
      globalacc2 = 0
      nsamples = 0
      
      label_all=None
      allpredictions = None
      for batch_idx, data in enumerate(dataloader):
      

        #if (batch_idx%100==0) and (batch_idx>=100):
          #print('at val batchindex: ',batch_idx)
    
        inputs = data[0].to(device)        

        #print('at val batchindex: ',batch_idx, inputs.shape)

        labels = data[1]
        #print('labels',labels)
        idx = data[2]

        avgcpuout = None
        avgloss = None
        for viewind in range(1): # loops over all rotations
        
          # inputs.shape = (batchsize, differentrotationsofthesameimage,3colorchannels,h,w)
          
          outputs = model(inputs[:,:,:,:])

          cpuout= outputs.to('cpu') #outputs #outputs.to('cpu')
          
          '''
          if avgcpuout is None:
            avgcpuout = cpuout / inputs.shape[1]
          else:
            avgcpuout += cpuout / inputs.shape[1]
          '''
          # if no averaging over rotations, then  
          avgcpuout = cpuout #instead
          
        if allpredictions is None:
          allpredictions = avgcpuout
        else:
          allpredictions = torch.cat( (allpredictions, avgcpuout),dim=0 )
        #print('allpredictions : ', allpredictions.shape,  avgcpuout.shape)
        if label_all is None:
          label_all = labels
        else:
          label_all = torch.cat( (label_all, labels),dim=0 )
        #print('allpredictions : ', allpredictions.shape,  avgcpuout.shape)
    return  allpredictions, label_all




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
          transforms.Normalize([0.30067, 0.30067, 0.30067], [0.378928, 0.378928, 0.378928])
      ]),
  }

  if True == config['use_gpu']:
    device= torch.device('cuda:0')
  else:
    device= torch.device('cpu')
              
  dataset_test = icedataset_testrot_uresize('./test_withBoundaries_new_Julie.npy', transforms =  data_transforms['test']) #test_withBoundaries_new_Julie.npy', transforms =  data_transforms['test'])
  #savept= './scores_v4_10052021'  
  savept= './scores_v4_10052021_rebal0half_densenet121_AdamW_10-4_equalproba'   
  
  losses = []
  globalacc2 = 0
  nsamples = 0
  modelaveraged_scores = None
  for cvind in range(config['numcv']):
  
    #sampler_test = getrandomsampler_innersplit5to2simple(outpath = config['splitpath'] ,numcv = config['numcv'], outercvind = cvind, trainvalortest ='test')
    #dataloader_test = torch.utils.data.DataLoader(dataset = dataset_test, batch_size= config['batchsize_val'], shuffle=False, sampler=sampler_test, batch_sampler=None, num_workers=0, collate_fn=None)
    dataloader_test = torch.utils.data.DataLoader(dataset = dataset_test, batch_size= config['batchsize_val'], shuffle=False, sampler= None, batch_sampler=None, num_workers=0, collate_fn=None)

    #model = models.resnet18(pretrained=False)
    #num_ft = model.fc.in_features
    #model.fc = nn.Linear(num_ft, config['numcl'])
    model = models.densenet121(pretrained=False)
    num_ft = model.classifier.in_features
    model.classifier = nn.Linear(num_ft, config['numcl'])


    savedweights = torch.load( os.path.join(savept,'bestweights_outercv{:d}.pt'.format(cvind)) )
    model.load_state_dict(savedweights)
    
    model = model.to(device)

    tmpscores,labels = evaluate_augavg_testonly(model, dataloader_test, criterion = None, device = device, numcl  = config['numcl'] )
    #print('tmpscores.shape', tmpscores.shape)
    #print('labels.shape', labels.shape)
    # save results

    if   modelaveraged_scores is None:
      modelaveraged_scores = tmpscores / float(config['numcv'])
    else:
      modelaveraged_scores += tmpscores / float(config['numcv'])
  modelaveraged_probability = torch.nn.functional.softmax(modelaveraged_scores,dim=1)
  print('modelaveraged_probability:',modelaveraged_probability)  
  #model average prediction eg:0,1,2....
  _, preds = torch.max(modelaveraged_scores, 1)
  print('preds:',preds)  
  #model average prediction max probability  eg: 0.99
  preds_proba ,_ = torch.max(modelaveraged_probability, 1)
  print('preds_proba:',preds_proba,preds_proba.shape)
  

    #np.save(os.path.join(savept,'noavg_preds_outercv{:d}.npy'.format(cvind)), np.asarray(preds) )
  
  print('modelaveraged_scores.shape', modelaveraged_scores.shape)
  print('label:',labels.shape[0])
  classcounts = torch.zeros(config['numcl']) 
  classcounts_test = torch.zeros(config['numcl'])
  confusion_matrix = torch.zeros(config['numcl'], config['numcl'])
  #confusion_matrix_test = torch.zeros(config['numcl'], config['numcl'])
  failidx= []
  proba_threshold=[]
  globalacc_all = None
  for i in np.arange(0.35,1,0.01):
    proba_threshold=torch.where(preds_proba.double() >i,preds_proba.double(),float(0))
    #print('proba_threshold:',proba_threshold,proba_threshold.shape)
    index=torch.nonzero(proba_threshold,as_tuple=False)
    print('index:', index.shape)
    np.save(os.path.join(savept,'original_test_noavg_index_outercv{:e}.npy'.format(i)), list(index.shape) )
    confusion_matrix_test = torch.zeros(config['numcl'], config['numcl'])
    for j in index:
      inds_test = torch.nonzero(labels[j,:],as_tuple=True)
      #print('inds_test:',  inds_test)
      confusion_matrix_test[inds_test[1],preds[j].long()]+=1
    print('confusion_matrix_test:', confusion_matrix_test.shape, confusion_matrix_test)
    np.save(os.path.join(savept,'original_test_noavg_confusion_matrix_outercv{:e}.npy'.format(i)), confusion_matrix_test.cpu().numpy() )
    globalacc2_test = 0
    for j in index:
        lbinds_test = torch.nonzero(labels[j,:],as_tuple=True)
        #print('lbinds_test:',lbinds_test)
        globalacc2_test+= torch.sum( preds[j]==lbinds_test[1] )
    classcounts+=torch.sum(index,dim=0)
    #print('classcounts:', classcounts)
    print('globalacc2_test:',globalacc2_test)
    #nsamples_test += index.shape[0]       
    globalacc_test = globalacc2_test/ float(index.shape[0]) 
    print('globalacc_test:',globalacc_test,globalacc_test.shape)

    #if globalacc_all is None:
    #    globalacc_all=globalacc_test
    #else:
    #    globalacc_all = torch.cat( (globalacc_all,globalacc_test),dim=0 )
    #print('globalacc_all:',globalacc_all.shape, globalacc_all)
    cwacc_test = confusion_matrix.diag() / classcounts_test
    print('cwacc_test:',cwacc_test)
    #np.save(os.path.join(savept,'test_noavg_cwacc_outercv{:e}.npy'.format(i)), cwacc_test.cpu().numpy() )
    np.save(os.path.join(savept,'original_test_noavg_globalacc_outercv{:e}.npy'.format(i)), globalacc_test.cpu().numpy() )


  for si in range(labels.shape[0]):
    #print(labels[si,:])
    inds = torch.nonzero(labels[si,:],as_tuple=True)
    #print('inds',inds[0],preds[si].long())
    confusion_matrix[inds[0],preds[si].long()]+=1
          
    #if inds[0] !=preds[si].long():
    #    failidx.append(idx[si])
            
  classcounts+=torch.sum(labels,dim=0)

  for si in range(labels.shape[0]):
    lbinds = torch.nonzero(labels[si,:],as_tuple=True)
    globalacc2+= torch.sum( preds[si]==lbinds[0] )
  nsamples += labels.shape[0]
        
  globalacc = globalacc2/ float(nsamples)  
          

  cwacc = confusion_matrix.diag() / classcounts
  print('test eval class-wise acc', torch.mean(cwacc.cpu()).item() )
  print('test eval global acc', globalacc.cpu().item() )
  np.save(os.path.join(savept,'test_modelavg_preds.npy'), modelaveraged_scores.to('cpu').numpy() )


if __name__=='__main__':
  runstuff()  

  

