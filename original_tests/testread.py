import numpy as np, os,sys, matplotlib.pyplot as plt


def tester():

  with open('test_withBoundaries_new.npy','rb') as f:
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

    #ar = (sizes[i,0]*sizes[i,1])**0.5
    #mom1a+= np.mean( ar) / a.shape[0]
    #mom2a+= np.mean(ar * ar) /a.shape[0]

    

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

  #mean std 0.11091382547154618 0.14565146317287114

  for c in range(9):
    print('c',c,np.sum(lb[:,c]))


  #print('area stats ',mom1a,  (mom2a-mom1a*mom1a)**0.5 )
  print('aspect stats ',mom1asr,  (mom2asr-mom1asr*mom1asr)**0.5 )
  
def tester2():
  with open('testread.npy','rb') as f:
    d = np.load(f,allow_pickle=True)
  print(type(d), d.shape)
  

if __name__=='__main__':
  tester()
  
  
