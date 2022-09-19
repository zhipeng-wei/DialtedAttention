import os
from multiprocessing import Pool, current_process, Queue
from utils import OPT_PATH
import glob

# Only run it on densenet121.
di_upper_bounds_cmd = 'python main.py --gpu {gpu} --white_box densenet121 --loss_fn CE --attack DIWithDifferentUpperBound --target --upper_bound {upper_bound} {save} --file_tailor exp_f1_{upper_bound}'


if __name__ == '__main__':
    NUM_GPUS = 2
    PROC_PER_GPU = 2
    queue = Queue()
    
    for gpu_ids in range(NUM_GPUS):
        for _ in range(PROC_PER_GPU):
            queue.put(gpu_ids)

    def attack(attack_cmd, upper_bound):
        gpu_id = queue.get()
        try: 
            ident = current_process().ident
            print('{}: starting process on GPU {}'.format(ident, gpu_id))
            if upper_bound in [0, 330, 450, 570]:
                save = '--saveperts'
            else:
                save = '--no-saveperts'
            print (attack_cmd.format(gpu=gpu_id, upper_bound=upper_bound, save=save))
            os.system(attack_cmd.format(gpu=gpu_id, upper_bound=upper_bound, save=save))
        finally:
            queue.put(gpu_id)

    pool = Pool(processes=PROC_PER_GPU * NUM_GPUS)
    upper_bounds = [0, 330, 390, 450, 510, 570, 630]
    for upper_bound in upper_bounds:
        pool.apply_async(attack, (di_upper_bounds_cmd, upper_bound,))

    pool.close()
    pool.join()

