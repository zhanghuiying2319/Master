

import os,sys,math,numpy as np, matplotlib.pyplot as plt
def runstuff():
  #savept= './scores_v2_23042021' #0.7436183691024781 0.5736260370351374
  #savept= './scores_v2_23042021_sizes' #0.7456393063068391 0.5900695040822029
  #savept= './scores_v2_23042021_aspectratios' #0.7385990262031555 0.5795704715885222
  #savept= './scores_v2_23042021_aspectratios_posenc10' #0.7458199739456176 0.5812392295338213
  #savept= './scores_v2_23042021_sizes_posenc' #0.7430180013179779 0.5872658278793097
  #savept= './scores_v2_10052021_resnet18'#0.8316610991954805 0.69428976956341 
  #savept= './scores_v2_10052021'#0.8317853093147278 0.7048914320766926
  #savept= './scores_v2_10052021_resnet152'#0.8280969798564911 0.6937087890174654
  #savept= './scores_v2_10052021_densenet121'#0.8319704532623291 0.7073589269485738
  #savept= './scores_v2_10052021_densenet169'#0.8308633685111999 0.7009367884861099
  #savept= './scores_v2_10052021_densenet169_lr10-4'#0.8231152713298797 0.6661039313508405
  #savept= './scores_v2_10052021_densenet201'#0.829327666759491 0.7049643322825432
  #savept= './scores_v2_10052021_densenet121_lr10-4'#0.8157327711582184 0.658889321403371
  #savept= './scores_v2_10052021_densenet121_AdamW'#0.7851732373237611 0.5805510822683573
  #savept= './scores_v2_10052021_densenet121_AdamW_lr10-4'#0.8328917860984801 0.7013759356406
  #savept= './scores_v2_10052021_densenet121_lay34'#0.824346488714218 0.6768812665508853
  #savept= './scores_v2_10052021_densenet121_lr10-4_Adamw_lay34'#0.8228672623634338 0.6793526419334941
  #savept= './scores_v2_10052021_densenet121_lr10-5_Adamw_lay34'#0.7684365868568419 0.4613950693876379

  #savept= './scores_v2_10052021_densenet121_lr10-5_SGD_lay34'

  savept= './scores_v2_09052021_aspectratios_densenet121_SGD'
  numcv =10
  
  cwacc= np.zeros(9)
  globalacc=0
  for cvind in range(numcv):
    cwacctmp=np.load(os.path.join(savept,'cwacc_outercv{:d}.npy'.format(cvind)) )
    globalacctmp=np.load(os.path.join(savept,'globalacc_outercv{:d}.npy'.format(cvind)) )
    confusion_matrixtmp=np.load(os.path.join(savept,'confusion_matrix_outercv{:d}.npy'.format(cvind)))
    
    globalacc+=globalacctmp / float(numcv)
    cwacc+=cwacctmp / float(numcv)
  print(globalacc, np.mean(cwacc) )
  
if __name__=='__main__':
  runstuff() 
