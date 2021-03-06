# coding: utf-8
# This script enumerate models that have minimum evaluation losses
# for each model directory


import os
import json


def get_best_model_info(trainer_state_path, extra_keys, sort_metric='eval_loss'):
    """Read rainer_state to return which model is the besi in the run
    We use eval loss as model score.
    """
    
    sig = 1
    if sort_metric.endswith('!'):
        sig = -1
        sort_metric = sort_metric[:-1]

    state = {}
    with open(trainer_state_path, 'r') as f:
        state = json.load(f)
    
    log_history = state.get('log_history', None)
    if log_history:
        items_sorted = sorted([_ for _ in log_history if sort_metric in _], key=lambda x: sig*x[sort_metric])
        keys = ['epoch', 'step', 'eval_loss'] + extra_keys
        best_results = {kv[0]:kv[1] for kv in items_sorted[0].items() if kv[0] in keys}
        return best_results
    return None


def print_best_models(root_path, show_only_exist, extra_keys, sort_metric):
   
    target_name = 'trainer_state.json'
    header = ['#model', 'trial', 'task', 'epoch', 'step', 'eval_loss', 'checkpoint'] + extra_keys
    for i in range(len(header)):
        if header[i] == sort_metric.rstrip('!'):
            header[i] = '*'+header[i]

    print(*header, sep='\t')   
    
    for cur_dir, dir_names, file_names in os.walk(root_path):
        
        if 'checkpoint' not in cur_dir and target_name in file_names:
            
            # determine the best model
            trainer_state_path = os.path.join(cur_dir, target_name)
            best_valid_info = get_best_model_info(trainer_state_path, extra_keys, sort_metric)
            
            # print out the infomation about the best model
            if best_valid_info:
                task_name = os.path.basename(cur_dir)
                model_name = os.path.basename(os.path.dirname(cur_dir))
                trial_name = os.path.basename(os.path.dirname(os.path.dirname(cur_dir)))
                checkpoint_path = os.path.join(cur_dir, 'checkpoint-%d'%best_valid_info['step'])
                
                if show_only_exist and not os.path.exists(checkpoint_path):
                    continue
                print(model_name, trial_name, task_name, 
                   best_valid_info['epoch'], best_valid_info['step'], best_valid_info['eval_loss'],
                   checkpoint_path, *[best_valid_info[_] for _ in extra_keys],
                   sep='\t')


if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Enumerate best models')
    parser.add_argument('--root', '-r', type=str, required=True,
        help='path to a root directory to walk recursively')
    parser.add_argument('--exist', '-e', type=int, default=1,
        help='If set 1, list up information only about models that really exist.')
    parser.add_argument('--keys', '-k', type=str, default=None,
        help='Additional keys to display; comma splitted and do not use spaces.')
    parser.add_argument('--metric', '-m', type=str, default='eval_loss',
        help='Metric for sorting. If you want the descending order, add "!" at the end')
    args = parser.parse_args()
    
    extra_keys = [] if args.keys is None else args.keys.split(',')
    print_best_models(args.root, args.exist, extra_keys, args.metric)
