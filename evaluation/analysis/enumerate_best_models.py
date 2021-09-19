# coding: utf-8
# This script enumerate models that have minimum evaluation losses
# for each model directory


import os
import json


def get_best_model_info(trainer_state_path):
    """Read rainer_state to return which model is the besi in the run
    We use eval loss as model score.
    """
    state = {}
    with open(trainer_state_path, 'r') as f:
        state = json.load(f)
    
    log_history = state.get('log_history', None)
    if log_history:
        items_sorted = sorted([_ for _ in log_history if 'eval_loss' in _], key=lambda x: x['eval_loss'])
        return {
            'epoch': items_sorted[0]['epoch'],
            'step': items_sorted[0]['step'],
            'eval_loss': items_sorted[0]['eval_loss'],
        }
    
    return None


def print_best_models(root_path, show_only_exist):
   
    target_name = 'trainer_state.json'
    header = ['#model', 'trial', 'task', 'epoch', 'step', 'eval_loss', 'checkpoint']
    print(*header, sep='\t')   
    
    for cur_dir, dir_names, file_names in os.walk(root_path):
        
        if 'checkpoint' not in cur_dir and target_name in file_names:
            
            # determine the best model
            trainer_state_path = os.path.join(cur_dir, target_name)
            best_valid_info = get_best_model_info(trainer_state_path)
            
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
                   checkpoint_path, sep='\t')


if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Enumerate best models')
    parser.add_argument('--root', '-r', type=str, required=True,
        help='path to a root directory to walk recursively')
    parser.add_argument('--exist', '-e', type=int, default=1,
        help='If set 1, list up information only about models that really exist.')
    args = parser.parse_args()
    
    print_best_models(args.root, args.exist)
