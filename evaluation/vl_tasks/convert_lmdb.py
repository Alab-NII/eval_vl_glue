# conding: utf-8
#
# Convert lmdb to pickle files.
# This script converts the preprocesed lmdb by e-bug to a pickle files.
# Sub-directories that correspond to splits are created in the destionation directory
# and pickle files will be placed in their split.
# We decided to use pickle files because loading lmdb was slow in our environment.


import lmdb
import pickle
import os
import tqdm
import numpy as np

from eval_vl_glue import _decode_tsv_item

default_fields = [
    "img_id", "img_h", "img_w", "objects_id", "objects_conf",
    "attrs_id", "attrs_conf", "num_boxes", "boxes", "features",
]

# Get a split and file name from a key
# Given a key of lmdb
# Expected to return a split name and a file name
# Current version is for nlvr2
def nlvr2_get_split_and_file_name(key):
    
    return key.split('-')[0], key


# Feature compiler
# Given an item object of lmdb and fields to be kept
# Expected to return an object, which will be pickled to save
# Current version is for nlvr2
def nlvr2_feature_compiler(item, fields):
    
    item = _decode_tsv_item(item)
    return {k:v for k, v in item.items() if k in fields}


def convert(
        src_dir_path, dest_dir_path, 
        get_split_and_file_name, feature_compiler, fields, max_num=None
    ):
    """
    Make the pickle files of image features from the lmdb file in src_dir_path.
    Pickled files are placed in sub-directries that made according to the keys of the lmdb.
    split and file_name are calculated from key by get_split_and_file_name.
    Features to be saved is decided by feature_compiler.
    """
    
    os.makedirs(dest_dir_path, exist_ok=False)
    
    env = lmdb.open(
        src_dir_path,
        max_readers=1,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    
    with env.begin(write=False) as txn:
        keys = pickle.loads(txn.get("keys".encode()))
        if max_num:
            keys = keys[:max_num]
        for key in tqdm.tqdm(keys):
            # decide where to save
            split, file_name = get_split_and_file_name(key.decode())
            dirpath = os.path.join(dest_dir_path, split)
            file_path = os.path.join(dirpath, file_name)
            
            # read main content
            item = pickle.loads(txn.get(key))
            item = feature_compiler(item, fields)
            
            # save
            if not os.path.exists(dirpath):
                os.makedirs(dirpath, exist_ok=True)
            with open(file_path, 'wb') as f:
                pickle.Pickler(f).dump(item)


if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(description='convert lmdb to pickle files')
    parser.add_argument('--src', '-s', required=True, type=str,
        help='path to the src lmdb directory')
    parser.add_argument('--dest', '-d', required=True, type=str,
        help='path to the dest directory')
    parser.add_argument('--fields', '-f', default=','.join(default_fields), type=str,
        help='field to be kept; comma splitted.')
    args = parser.parse_args()
    
    get_split_and_file_name = nlvr2_get_split_and_file_name
    feature_compiler = nlvr2_feature_compiler

    print('start')
    convert(args.src, args.dest, get_split_and_file_name, feature_compiler, args.fields)
    print('end')
