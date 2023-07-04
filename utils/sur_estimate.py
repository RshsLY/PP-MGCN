import time
import  numpy as np
from sksurv.metrics import concordance_index_censored
def c_index_cal(Y,sur_time,censor):
   # tim=time.time()
    for i in range(len(Y)):
        Y[i]=Y[i].cpu().detach().numpy()
        sur_time[i]=sur_time[i].cpu().detach().numpy()
        censor[i]=censor[i].cpu().detach().numpy().astype(np.bool)
    censor=np.array(censor)
    sur_time=np.array(sur_time)
    Y=np.array(Y)
    ss = 0
    s = 0
    for i in range(len(Y)):
        for j in range(len(Y)):
            if censor[i] == 0 and sur_time[i] < sur_time[j]:
                ss = ss + 1
            if censor[i] == 0 and sur_time[i] < sur_time[j] and Y[i] < Y[j]:
                s = s + 1
    #print(time.time()-tim)
    return s / ss
