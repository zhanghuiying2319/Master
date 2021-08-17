

import os,sys,math,numpy as np, matplotlib.pyplot as plt
def runstuff():

  #savept= './scores_v2_23042021' #0.7580982029438019 0.5812124161981046
  #savept= './scores_v2_23042021_sizes' #0.7552803814411163 0.583345946110785
  #savept= './scores_v2_23042021_sizes_posenc' #0.7596977412700654 0.583193539083004
  #savept= './scores_v2_23042021_aspectratios' #0.7534634172916412 0.575133552961051
  #savept= './scores_v2_23042021_aspectratios_posenc10' #0.7560707509517669 0.5779228328727186
  savept= './scores_v2_09052021_aspectratios_densenet121_lr10-4_adamW' 


  numcv =10
  
  cwacc= np.zeros(9)
  globalacc=0
  
  
  confusion = np.zeros((9,9))
  for cvind in range(numcv):
    cwacctmp=np.load(os.path.join(savept,'cwacc_outercv{:d}.npy'.format(cvind)) )
    globalacctmp=np.load(os.path.join(savept,'globalacc_outercv{:d}.npy'.format(cvind)) )
    confusion_matrixtmp=np.load(os.path.join(savept,'confusion_matrix_outercv{:d}.npy'.format(cvind)))
    
    globalacc+=globalacctmp / float(numcv)
    cwacc+=cwacctmp / float(numcv)
    
    conf = np.load(    os.path.join(savept,'confusion_matrix_outercv{:d}.npy'.format(cvind)))
    confusion+=conf

  counts = np.sum(confusion, axis=1)  
  nconfusion = np.array(confusion)
  for r in range(9):
    nconfusion[r,:]= nconfusion[r,:] / np.sum(confusion[r,:] )

  for r in range(9):
    print(r, 1 - nconfusion[r,r] )
  print(counts)

  plt.matshow(nconfusion)
  plt.show()
    
  print(globalacc, np.mean(cwacc) )
  
if __name__=='__main__':
  runstuff() 
