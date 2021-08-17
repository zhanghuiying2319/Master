

import os,sys,math,numpy as np, matplotlib.pyplot as plt
def runstuff():

  #savept= './scores_v2_23042021' #0.7580982029438019 0.5812124161981046
  #savept= './scores_v2_23042021_sizes' #0.7552803814411163 0.583345946110785
  #savept= './scores_v2_23042021_sizes_posenc' #0.7596977412700654 0.583193539083004
  #savept= './scores_v2_23042021_aspectratios' #0.7534634172916412 0.575133552961051
  #savept= './scores_v2_23042021_aspectratios_posenc10' #0.7560707509517669 0.5779228328727186
  #savept= './oldscores/scores_v2_23042021_innercv_sizes' #0.7794053614139558 0.6070150885730982

  #savept= './scores_v4_10052021'
  #savept= './scores_v4_10052021_rebal0half'  
  #savept= './scores_v4_10052021_rebal0half_densenet121_SGD'#0.8922465562820435 0.8617303603225284
  #savept= './scores_v4_10052021_rebal0half_densenet121_AdamW_10-4'#0.9014121651649475 0.8683200238479508

  savept= './scores_v4_10052021_rebal0half_densenet121_SGD_equalproba'

  numcv =10
  numcl = 9
  
  cwacc= np.zeros(numcl)
  globalacc=0
  
  
  confusion = np.zeros((numcl,numcl))
  for cvind in range(numcv):
    cwacctmp=np.load(os.path.join(savept,'rotavg_cwacc_outercv{:d}.npy'.format(cvind)) )
    globalacctmp=np.load(os.path.join(savept,'rotavg_globalacc_outercv{:d}.npy'.format(cvind)) )
    
    confusion_matrixtmp=np.load(os.path.join(savept,'rotavg_confusion_matrix_outercv{:d}.npy'.format(cvind)))
    
    globalacc+=globalacctmp / float(numcv)
    cwacc+=cwacctmp / float(numcv)
    
    conf = np.load(    os.path.join(savept,'rotavg_confusion_matrix_outercv{:d}.npy'.format(cvind)))
    confusion+=conf

  counts = np.sum(confusion, axis=1)  
  nconfusion = np.array(confusion)
  for r in range(numcl):
    nconfusion[r,:]= nconfusion[r,:] / np.sum(confusion[r,:] )

  for r in range(numcl):
    print(r, 1 - nconfusion[r,r] )
  print(counts)

  plt.matshow(nconfusion)
  plt.show()
    
  print(globalacc, np.mean(cwacc) )
  
if __name__=='__main__':
  runstuff() 
