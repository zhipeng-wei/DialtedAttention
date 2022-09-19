import os
from multiprocessing import Pool, current_process, Queue
from utils import OPT_PATH
import glob

iou_cmd = 'python gradcam_generate.py --gpu {gpu} --magnitude {mag} --transform_name {name}'

if __name__ == '__main__':
    NUM_GPUS = 2
    PROC_PER_GPU = 2
    queue = Queue()
    
    for gpu_ids in range(NUM_GPUS):
        for _ in range(PROC_PER_GPU):
            queue.put(gpu_ids)

    def baseline_run(cmd, mag, name):
        gpu_id = queue.get()
        try: 
            ident = current_process().ident
            print('{}: starting process on GPU {}'.format(ident, gpu_id))
            print ('the running command is', cmd.format(gpu=gpu_id, mag=mag, name=name))
            os.system(cmd.format(gpu=gpu_id, mag=mag, name=name))
        finally:
            queue.put(gpu_id)
    
    pool = Pool(processes=PROC_PER_GPU * NUM_GPUS)
    names = ['Flip', 'Crop', 'Solarize', 'Sharpness', 'Contrast', 'Color', 'Brightness', 'Rotate', \
        'ShearX', 'ShearY', 'TranslateX', 'TranslateY']
    mags = [2,4,6,8,10]
    for mag in mags:
        for name in names:
            pool.apply_async(baseline_run, (iou_cmd, mag, name, ))

    pool.close()
    pool.join()
