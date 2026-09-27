"""
Training a linear controller on latent + recurrent state
with CMAES.
"""
import argparse
import sys
import queue  # 新增：用于捕获死锁超时
from os.path import join, exists
from os import mkdir, unlink, listdir, getpid
from time import sleep
from torch.multiprocessing import Process, Queue
import torch
import cma
from models import Controller
from tqdm import tqdm
import numpy as np
from utils.misc import RolloutGenerator, ASIZE, RSIZE, LSIZE
from utils.misc import load_parameters
from utils.misc import flatten_parameters

################################################################################
#                           Thread routines                                    #
################################################################################
def slave_routine(p_queue, r_queue, e_queue, p_index, logdir, tmp_dir, time_limit):
    gpu = p_index % torch.cuda.device_count()
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')

    sys.stdout = open(join(tmp_dir, str(getpid()) + '.out'), 'a')
    sys.stderr = open(join(tmp_dir, str(getpid()) + '.err'), 'a')

    with torch.no_grad():
        r_gen = RolloutGenerator(logdir, device, time_limit)

        while e_queue.empty():
            if p_queue.empty():
                sleep(.1)
            else:
                s_id, params = p_queue.get()
                r_queue.put((s_id, r_gen.rollout(params)))

################################################################################
#                           Evaluation                                         #
################################################################################
def evaluate(solutions, results, p_queue, r_queue, rollouts=100):
    index_min = np.argmin(results)
    best_guess = solutions[index_min]
    restimates = []

    for s_id in range(rollouts):
        p_queue.put((s_id, best_guess))

    print("Evaluating...")
    for _ in tqdm(range(rollouts)):
        try:
            # 加入超时保护，防止评估阶段死锁
            result = r_queue.get(timeout=60)
            restimates.append(result[1])
        except queue.Empty:
            print("\n[FATAL] Workers died silently during evaluation. Check exp_dir/tmp/ logs.")
            sys.exit(1)

    return best_guess, np.mean(restimates), np.std(restimates)


################################################################################
#                           Main Execution                                     #
################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, help='Where everything is stored.')
    parser.add_argument('--n-samples', type=int, help='Number of samples used to obtain return estimate.')
    parser.add_argument('--pop-size', type=int, help='Population size.')
    parser.add_argument('--target-return', type=float, help='Stops once the return gets above target_return')
    parser.add_argument('--display', action='store_true', help="Use progress bars if specified.")
    # 【修复】将默认工作进程数从 32 降为 8，防止 CPU/GPU 被过载堵死
    parser.add_argument('--max-workers', type=int, help='Maximum number of workers.', default=8)
    args = parser.parse_args()

    n_samples = args.n_samples
    pop_size = args.pop_size
    num_workers = min(args.max_workers, n_samples * pop_size)
    time_limit = 1000

    tmp_dir = join(args.logdir, 'tmp')
    if not exists(tmp_dir):
        mkdir(tmp_dir)
    else:
        for fname in listdir(tmp_dir):
            unlink(join(tmp_dir, fname))

    ctrl_dir = join(args.logdir, 'ctrl')
    if not exists(ctrl_dir):
        mkdir(ctrl_dir)

    p_queue = Queue()
    r_queue = Queue()
    e_queue = Queue()

    print(f"Starting {num_workers} worker processes...")
    for p_index in range(num_workers):
        Process(target=slave_routine, 
                args=(p_queue, r_queue, e_queue, p_index, args.logdir, tmp_dir, time_limit)).start()

    controller = Controller(LSIZE, RSIZE, ASIZE)

    cur_best = None
    ctrl_file = join(ctrl_dir, 'best.tar')
    print("Attempting to load previous best...")
    if exists(ctrl_file):
        state = torch.load(ctrl_file, map_location={'cuda:0': 'cpu'})
        cur_best = - state['reward']
        controller.load_state_dict(state['state_dict'])
        print("Previous best was {}...".format(-cur_best))

    parameters = controller.parameters()
    es = cma.CMAEvolutionStrategy(flatten_parameters(parameters), 0.1, {'popsize': pop_size})

    epoch = 0
    log_step = 3
    
    while not es.stop():
        if cur_best is not None and - cur_best > args.target_return:
            print("Already better than target, breaking...")
            break

        r_list = [0] * pop_size  
        solutions = es.ask()

        for s_id, s in enumerate(solutions):
            for _ in range(n_samples):
                p_queue.put((s_id, s))

        if args.display:
            pbar = tqdm(total=pop_size * n_samples)
            
        for _ in range(pop_size * n_samples):
            try:
                # 【修复】加入 60 秒硬超时机制。如果子进程挂了，主进程会立刻抛出异常，而不是永远卡死
                r_s_id, r = r_queue.get(timeout=60)
                r_list[r_s_id] += r / n_samples
                if args.display:
                    pbar.update(1)
            except queue.Empty:
                print("\n[FATAL] Timeout waiting for workers. They likely crashed. Check exp_dir/tmp/ logs.")
                sys.exit(1)
                
        if args.display:
            pbar.close()

        es.tell(solutions, r_list)
        es.disp()

        if epoch % log_step == log_step - 1:
            best_params, best, std_best = evaluate(solutions, r_list, p_queue, r_queue)
            print("Current evaluation: {}".format(best))
            if not cur_best or cur_best > best:
                cur_best = best
                print("Saving new best with value {}+-{}...".format(-cur_best, std_best))
                load_parameters(best_params, controller)
                torch.save(
                    {'epoch': epoch,
                     'reward': - cur_best,
                     'state_dict': controller.state_dict()},
                    join(ctrl_dir, 'best.tar'))
            if - best > args.target_return:
                print("Terminating controller training with value {}...".format(best))
                break

        epoch += 1

    es.result_pretty()
    e_queue.put('EOP')
