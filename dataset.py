import webdataset as wds
import wids

import torch
from torch.utils.data import Dataset
import numpy as np
from glob import glob
import os
import datetime


class SDOMLlite(Dataset):
    def __init__(self, data_dir, channels=['hmi_m', 'aia_0094', 'aia_0131', 'aia_0171', 'aia_0193', 'aia_0211', 'aia_1600', 'aia_1700']):
        self.data_dir = data_dir
        self.channels = channels
        index_file = glob(os.path.join(data_dir, '*.json'))
        if len(index_file) == 0:
            raise RuntimeError('No index file (.json) found')
        index_file = index_file[0]
        print('SDOML-lite')
        print('Directory  : {}'.format(self.data_dir))
        print('Index      : {}'.format(index_file))
        self.webdataset = wids.ShardListDataset(index_file)

        date_start, date_end = self.find_date_range()
        self.date_start = date_start
        self.date_end = date_end
        print('Start date : {}'.format(self.date_start))
        print('End date   : {}'.format(self.date_end))
        print('Channels   : {}'.format(self.channels))
        
        self.channels_webdataset_keys = ['.'+c+'.npy' for c in self.channels]
        
        self.date_to_index = {}
        self.dates = []
        for i in range(len(self.webdataset)):
            cs = self.webdataset[i].keys()
            has_all_channels = True
            for c in self.channels_webdataset_keys:
                if c not in cs:
                    has_all_channels = False
                    break
            if has_all_channels:
                date = self.get_date(i)
                self.date_to_index[date] = i
                self.dates.append(date)
                
        if len(self.dates) == 0:
            raise RuntimeError('No frames found with given list of channels')
                
        print('Total frames  : {:,}'.format(len(self.dates)))
        print('Dropped frames: {:,}'.format(len(self.webdataset) - len(self.dates)))                
           
    def get_date(self, index):
        return datetime.datetime.strptime(self.webdataset[index]['__key__'], '%Y/%m/%d/%H%M')
    
    def find_date_range(self):
        date_start = self.get_date(0)
        date_end = self.get_date(len(self.webdataset)-1) # wids doesn't support -1 indexing
        return date_start, date_end
    
    def __len__(self):
        return len(self.dates)
    
    def __getitem__(self, index):
        if isinstance(index, int):
            date = self.dates[index]
        elif isinstance(index, datetime.datetime):
            date = index
        elif isinstance(index, str):
            date = datetime.datetime.fromisoformat(index)
        else:
            raise ValueError('Expecting index to be int, datetime.datetime, or str (in the format of 2022-11-01T00:01:00)')
        data = self.get_frame(date)    
        return data, date.isoformat()
    
    def get_frame(self, date):
        index = self.date_to_index[date]
        data = self.webdataset[index]
        channels = []
        for c in self.channels_webdataset_keys:
            channels.append(data[c])
        channels = np.stack(channels)
        return channels

# WORK IN PROGRESS