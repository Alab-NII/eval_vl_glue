# coding: utf-8
# Preparing conceptual captions to calculate vocabulary with the GLUE Tasks.
# Note that this script handles just text.


import csv
import zlib
import numpy as np


class ConceptualCaptionsTextDataset(object):
    
    default_data_paths = {
        'train': 'conceptual_captions/Train_GCC-training.tsv',
        'valid': 'conceptual_captions/Validation_GCC-1.1.0-Validation.tsv',
    }
    
    default_id_list_paths = {
        'train': 'conceptual_captions/train_ids.txt',
        'valid': 'conceptual_captions/valid_ids.txt',
    }
    
    def __init__(self, data_paths=None, id_list_paths=None):
        self.data_paths = data_paths or self.default_data_paths
        self.id_list_paths = id_list_paths or self.default_id_list_paths
        self.splits = {}
        for key in self.data_paths:
            path = self.data_paths[key]
            id_list_path = self.id_list_paths.get(key, None)
            self.splits[key] = ConceptualCaptionsTextDatasetSplit(path, id_list_path)
    
    def __getitem__(self, key):
        return self.splits[key]


class ConceptualCaptionsTextDatasetSplit(object):
    """This class reads text from the tsv files distributed by the authors of the Conceptual Captions corpus.
    It also filters the items wiith the image id lists for volta pretraining to keep just items used in the pretraining.
    """
    
    @staticmethod
    def get_volta_image_id(url):
        """Map an image url to an id used in the volta pretraining"""
        return str(zlib.crc32(url.encode('utf-8')) & 0xffffffff)
    
    def __init__(self, data_path, id_list_path=None):
        """Arguments:
            data_path: path to a tsv file of the split of the Conceptual Captions corpus
            id_list_path: path to a txt file tha contains the image id lists for volta pretraining
                if None no filtering will be done.
        """
        self.data_path = data_path
        self.id_list_path = id_list_path
        self.do_filtering = id_list_path is not None
        self.data = []
        self.data_by_volta_image_id = {}
        self._load()
    
    def _load(self):
        
        data = []
        # We will collect all image ids for this split in this list
        # For the sake of speed, we will filter the data together later.
        if self.do_filtering:
            volta_image_id_array = []
        else:
            volta_image_id_array = None
        
        with open(self.data_path, 'r') as f:
            for row in csv.reader(f, delimiter='\t'):
                item = {
                    'sentence': row[0], 
                    'image_url': row[1], 
                    'volta_image_id': self.get_volta_image_id(row[1]),
                }
                data.append(item)
                if self.do_filtering:
                    volta_image_id_array.append(int(item['volta_image_id']))
        
        if self.do_filtering:
            volta_image_id_array = np.asarray(volta_image_id_array)
            
            # List up image ids used in the pretraining
            id_array = []
            with open(self.id_list_path, 'r') as f:
                id_array.extend(int(line.strip()) for line in f.readlines())
            id_array = np.asarray(id_array)
            
            # Make a list of flags that show whether an id used in the pretraining or not.
            isin_array = np.isin(volta_image_id_array, id_array, assume_unique=True)
            # Filter data according to the flag list
            data = [item for flag, item in zip(isin_array, data) if flag == True]
        
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
