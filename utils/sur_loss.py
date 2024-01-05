import torch

def sur_loss(h,sur_time,censor):
    if sur_time==0:
        h = torch.squeeze(h)
        h = torch.split(h, [1, h.shape[0] - 1])
        s1 = -torch.sum(torch.log(1.0 - h[0] + 1e-30))
        if censor == 1:
            s1=s1-s1
            return s1
        else:
            return s1
    else :
        h=torch.squeeze(h)
        h=torch.split(h,[sur_time,1,h.shape[0]-1-sur_time])

        s0=-torch.sum(torch.log(h[0]+1e-30))
        s1=-torch.sum(torch.log(1.0-h[1]+1e-30))
        if censor==1:
            return s0
        else :
            return s0+s1

def diff_CL(h,hc,sur_time,censor):
    if sur_time==0:
        h = torch.squeeze(h)
        h = torch.split(h, [1, h.shape[0] - 1])

        hc = torch.squeeze(hc)
        hc = torch.split(hc, [1, hc.shape[0] - 1])

        h1_diff = torch.clamp(hc[0] - h[0], min=0)
        s1_diff = -torch.sum(torch.log(1.0 - h1_diff + 1e-30))
        if censor == 1:
            return s1_diff-s1_diff
        else:
            return s1_diff
    else :
        h=torch.squeeze(h)
        h=torch.split(h,[sur_time,1,h.shape[0]-1-sur_time])
        hc = torch.squeeze(hc)
        hc = torch.split(hc, [sur_time, 1, hc.shape[0] - 1 - sur_time])

        h0_diff = torch.clamp(h[0] - hc[0], min=0)
        s0_diff = -torch.sum(torch.log(1.0 - h0_diff + 1e-30))
        h1_diff = torch.clamp(hc[1] - h[1], min=0)
        s1_diff = -torch.sum(torch.log(1.0 - h1_diff + 1e-30))
        if censor==1:
            return s0_diff
        else :
            return s0_diff+s1_diff

def sur_loss_cc(h,h1,sur_time,censor):
    return sur_loss(h,sur_time,censor)+diff_CL(h,h1,sur_time,censor)

def sur_loss_cc(h,h1,sur_time,censor,p):
    return p*sur_loss(h,sur_time,censor)+(1.0-p)*diff_CL(h,h1,sur_time,censor)
def sur_loss_ccc(h,h1,h2,sur_time,censor):
    return sur_loss(h,sur_time,censor)+diff_CL(h,h1,sur_time,censor)+diff_CL(h,h2,sur_time,censor)