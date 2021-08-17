import numpy as np, os,sys, matplotlib.pyplot as plt


def tester():

  with open('withBoundaries_unlabeled_Julie_relabeled_huiying0802.npy','rb') as f:
    a = np.load(f,allow_pickle=True)
    b = np.load(f,allow_pickle=True)
    c = np.load(f,allow_pickle=True)

  print(a.shape)
  print(b.shape)
  print(c.shape)

  lb = np.zeros((0,9))
  sizes=np.zeros( (a.shape[0],2))


  mom1 = 0
  mom2 = 0

  mom1a = 0
  mom2a = 0

  mom1asr = 0
  mom2asr = 0

  for i in range(a.shape[0]):
    print('a[i].shape', a[i].shape, np.mean(a[i]))
    print(b[i][0], b[i][1])
    
    sizes[i,0]=b[i][0]
    sizes[i,1]=b[i][1]
    
    print('c[i].shape',c[i].shape, c[i])
    lb=np.concatenate( (lb, np.expand_dims(c[i],0)))
    
    print(type(a[i]),type(b[i]),type(c[i]))
  #print(a)
  #print(b)
  #print(np.linalg.norm(a-b))
    mom1+= np.mean(a[i]) / a.shape[0]
    mom2+= np.mean(a[i] * a[i]) /a.shape[0]

    ar = (sizes[i,0]*sizes[i,1])**0.5
    mom1a+= np.mean( ar) / a.shape[0]
    mom2a+= np.mean(ar * ar) /a.shape[0]

    

    aspectr = min(sizes[i,0], sizes[i,1] ) / float( max(sizes[i,0], sizes[i,1] ) )
    mom1asr+= np.mean( aspectr) / a.shape[0]
    mom2asr+= np.mean(aspectr * aspectr) /a.shape[0]



  print('mean std',mom1,  (mom2-mom1*mom1)**0.5 )

  hsizep = np.percentile(sizes[:,0], q= [ i*10 for i in range(11)  ])
  wsizep = np.percentile(sizes[:,1], q= [ i*10 for i in range(11)  ])


  print(sizes.shape[0]) # 4977
  print(sizes[:,0])

  print(hsizep)
  print(wsizep)



  for c in range(9):
    print('c',c,np.sum(lb[:,c]))


  print('area stats ',mom1a,  (mom2a-mom1a*mom1a)**0.5 )
  print('aspect stats ',mom1asr,  (mom2asr-mom1asr*mom1asr)**0.5 )
  
  
  #oldmean std 0.11091382547154618 0.14565146317287114
  
  #newmean 0.10980557842113262 0.14267757376590667
  #area stats  86.67150856860262 82.00518632883818
  #aspect stats  0.7743248779682481 0.17000014957327692

  
  # 0.09000902092429723 0.13107353611434053
  #area stats  83.70748629459149 69.54964718209402
  #aspect stats  0.7127774041455117 0.19795099063488308
  
  '''
    c 0 9087.0
    c 1 239.0
    c 2 201.0
    c 3 1934.0
    c 4 684.0
    c 5 688.0
    c 6 416.0
    c 7 1548.0
    c 8 1462.0
    
    class counts for the new dataset.
    
    standard dataloader: __getitem__(self,idx)
    get idx-th sample
    
    stochastic dataloader principledway, original probabilities:
    
    __getitem__(self,idx)
    

    # maintain a list: for each class which indices belong to that class
    inds[c] = ...
    
    # compute probability
    for c in range(numcl):
      classp[c] = counts[c] / np.sum(counts)
          
    # draw 
    class chosenclass from according to p[c] 

    from numpy.random import default_rng
    rng = default_rng()

    chosenclass = rng.choice( range(numcl), size=1, replace=False, p=classp)

    # then draw sample index from c0 uniformly
    
    idx = rng.choice( inds[chosenclass], size=1, replace=False, p=None)

    ##################################


    stochastic dataloader principledway, downscaled class 0:
    
    __getitem__(self,idx)
    

    # maintain a list: for each class which indices belong to that class
    inds[c] = ...
    
    # adjust count for class 0
    
     counts[0] /=2.0 
    
    # compute probability
    
    
    for c in range(numcl):
        classp[c] = counts[c] / np.sum(counts)
          
    # draw 
    class chosenclass from according to p[c] 

    from numpy.random import default_rng
    rng = default_rng()

    chosenclass = rng.choice( range(numcl), size=1, replace=False, p=classp)

    # then draw sample index from c0 uniformly
    
    idx = rng.choice( inds[chosenclass], size=1, replace=False, p=None)

  '''
  
def tester2():
  with open('testread.npy','rb') as f:
    d = np.load(f,allow_pickle=True)
  print(type(d), d.shape)
  

if __name__=='__main__':
  tester()
  
  
