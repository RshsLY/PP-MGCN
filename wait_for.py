import subprocess

for i in range(8):
    gpu_id=i
    # 使用nvidia-smi命令查询显卡状态，获取PID列表
    pids = subprocess.check_output(['nvidia-smi', '--query-gpu=pid', '--format=csv,noheader,nounits',
                                    '--id={}'.format(gpu_id)]).decode().split('\n')[:-1]
    pids = [int(pid) for pid in pids]
    if(len(pids)==-0):
        subprocess.call(["python", "/data/liuyong/ms_gcn/sur_main.py", "--gpu_index={}".format(gpu_id)])

