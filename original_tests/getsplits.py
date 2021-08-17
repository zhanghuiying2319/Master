import os,sys,math,numpy as np, matplotlib.pyplot as plt





def getsplits_2cls_plusminus(labels ,numcv):
    
    import numpy as np


    indexlist=[ [] for i in range(numcv) ]
    #4 sets

    if np.sum(labels==0)>0:
      print('??? np.sum(labels==0)>0')
      exit()

    for lbtype in [-1,1]:

      indices=[i for i in range(len(labels)) if (labels[i]==lbtype)  ]
      if( len(indices) >0):
        np.random.shuffle(indices)
        
        num=int(math.floor(len(indices)/float(numcv)))       
        if num==0:
          print('ERR: number of data in slice is 0, exiting!',len(indices),numcv,[lbtype,censtype])
          exit()    

        for cv in range(numcv):
          for k in range(cv*num,(cv+1)*num):
            indexlist[cv].append(indices[k])

        for k in range(numcv*num,len(indices)):
          rnd=np.random.randint(0,numcv)
          indexlist[rnd].append(indices[k])

   
    return indexlist



def getsplits_realvalued(obslb,numcv):

  bsizemult=2
  inds=np.argsort(obslb)

  indspercv=[ [] for _ in range(numcv) ]

  for i in inds:
    print(obslb[i])

  numbuckets=int(math.floor(len(obslb)/(bsizemult*numcv)))
  rem=  len(obslb)-numbuckets*(bsizemult*numcv)

  print(numbuckets,rem)

  for bind in range(numbuckets):

    bstart=bind*(bsizemult*numcv)
    bend=(bind+1)*(bsizemult*numcv)

    if bind==numbuckets-1 :
      bend+=int(rem)

    np.random.shuffle(inds[bstart:bend])
    #print(bstart,bend)

    for cv in range(numcv):
      indspercv[cv].extend( inds[bstart+cv*bsizemult:bstart+(cv+1)*bsizemult]  )

    if bind==numbuckets-1 :
      cv=np.random.choice(np.asarray([i for i in range(numcv)],dtype=np.int32), 1)
      cv=cv[0]
      #print(type(cv),type(bend),bstart+numcv*bsizemult,bend)
      indspercv[cv].extend(inds[bstart+numcv*bsizemult:bend] )
    
  return indspercv


def getsplits_ncls_categoricalapprox(labels ,numcl,numcv):
  #labels.shape = (numdata,numcl)
  if labels.shape[1]!=numcl:
    print('wrong number of classes')
    exit()
  
  
  counts = np.zeros(numcl)
  for cl in range(numcl):
    counts[cl]=np.sum(labels[:,cl]>0)
  print(counts)  
  
  if np.sum(counts)<labels.shape[0]:
    print('np.sum(counts)<labels.shape[0]', np.sum(counts),labels.shape[0])
    exit()
  
  
  
  if np.sum(counts)>labels.shape[0]:
    #iterative resolution of multiple labels
    #partial overlap
    print('partial overlap')
    pseudolabels = labels
    for si in range (labels.shape[0]):
      if np.sum( labels[si,:]>0) >1:
        #have overlap, suppress for splitting inversely proportionally
        valinds = [ cl for cl in range(numcl) if labels[si,cl]>0  ]
        # inversely proportional, preserve smallest class
        p = 1.0/counts[valinds]   #np.array([ 1.0/counts[cl] for cl in valinds  ])
        p = p / np.sum(p)
        
        chosencl= np.random.choice(valinds, size=1, replace=False, p=p)
        for cl in range(numcl):
          if cl !=chosencl:
            pseudolabels[si, cl] = 0
        if np.sum( pseudolabels[si,:]>0) !=1:
          print( ' np.sum( pseudolabels[si,:]>0) !=1:',  pseudolabels[si,:] )
          exit()
  else:
    print('no overlap')
    pseudolabels = labels
  
  posinds=[ [] for cl in range(numcl) ]
  for cl in range(numcl):
    posinds[cl]=[i for i in range(labels.shape[0]) if (pseudolabels[i,cl]>0)  ]
    print(len(posinds[cl]))

  #exit()  
    
  indexlist=[ [] for i in range(numcv) ]


  for cl in range(numcl):

    indices=posinds[cl]

    np.random.shuffle(indices)
    
    num=int(math.floor(len(indices)/float(numcv)))       
    if num==0:
      print('ERR: number of data in slice is 0, exiting!',len(indices),numcv,[lbtype,censtype])
      exit()    

    for cv in range(numcv):
      for k in range(cv*num,(cv+1)*num):
        indexlist[cv].append(indices[k])

    for k in range(numcv*num,len(indices)):
      rnd=np.random.randint(0,numcv)
      indexlist[rnd].append(indices[k])

  #... 
  globalp = counts/np.sum(counts)
  print('global', globalp )
  for cv in range(numcv):
    splitcounts = np.zeros(numcl)
    for cl in range(numcl):
      splitcounts[cl] = np.sum( labels[indexlist[cv],cl]>0 )
    print('cv',cv,'local:',  splitcounts / np.sum( splitcounts )  )
    #print('cv',cv,'diff:', globalp- splitcounts / np.sum( splitcounts ) )
    
  return indexlist

def split():
  np.random.seed(seed=3)
  
  outpath = './icesplits_v2_09052021'
  
  numcl = 9
  numcv = 10
  
  with open('test_withBoundaries_new_Julie.npy','rb') as f:
    a = np.load(f,allow_pickle=True)
    b = np.load(f,allow_pickle=True)
    c = np.load(f,allow_pickle=True)


  labels = np.zeros((c.shape[0],numcl))
  for l in range(c.shape[0]):
    labels[l,:]=c[l]



  indexlist = getsplits_ncls_categoricalapprox(labels ,numcl,numcv)

  if not os.path.isdir(outpath):
    os.makedirs(outpath)
    
  for cv in range(numcv):  
    fname= os.path.join(outpath,'split_cv'+str(cv)+'.txt')  
    with open(fname,'w+') as f:
      
      for i,ind in enumerate(indexlist[cv]):
        if i==0:
          f.write('{:d}'.format(ind))
        else:          
          f.write(' '+'{:d}'.format(ind))
      f.write('\n')
    
if __name__=='__main__':
  split()
  
  

