import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
from pytracking.utils.load_text import load_text
import os
from PIL import Image
from pathlib import Path


import os
import json
import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
from pytracking.utils.load_text import load_text


class VesselDataset(BaseDataset):
   
    
    def __init__(self, attribute=None):
        super().__init__()
        self.base_path = self.env_settings.vessel_path
        self.sequence_info_list = self._get_sequence_info_list()

        self.att_dict = None

        if attribute is not None:
            self.sequence_info_list = self._filter_sequence_info_list_by_attribute(attribute, self.sequence_info_list)

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path, 
        sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext) for frame_num in range(start_frame+init_omit, end_frame+1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64, backend='numpy')

        return Sequence(sequence_info['name'], frames, 'vessel', ground_truth_rect[init_omit:,:],
                        object_class=sequence_info['object_class'])

    def get_attribute_names(self, mode='short'):
        if self.att_dict is None:
            self.att_dict = self._load_attributes()

        names = self.att_dict['att_name_short'] if mode == 'short' else self.att_dict['att_name_long']
        return names

    def _load_attributes(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               'dataset_attribute_specs', 'vessel_attributes.json'), 'r') as f:
            att_dict = json.load(f)
        return att_dict

    def _filter_sequence_info_list_by_attribute(self, att, seq_list):
        if self.att_dict is None:
            self.att_dict = self._load_attributes()

        if att not in self.att_dict['att_name_short']:
            if att in self.att_dict['att_name_long']:
                att = self.att_dict['att_name_short'][self.att_dict['att_name_long'].index(att)]
            else:
                raise ValueError('\'{}\' attribute invalid.')

        return [s for s in seq_list if att in self.att_dict[s['name'][4:]]]

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        
        sequence_info_list = [
            {"name": "vessel_0001", "path": "data_seq/vessel_0001", "startFrame": 1, "endFrame": 243, "nz": 6,
             "ext": "png", "anno_path": "anno/vessel_0001.txt", "object_class": "vehicle"},
            {"name": "vessel_0002", "path": "data_seq/vessel_0002", "startFrame": 1, "endFrame": 342, "nz": 6,
             "ext": "png", "anno_path": "anno/vessel_0002.txt", "object_class": "vehicle"},
            {"name": "vessel_0003", "path": "data_seq/vessel_0003", "startFrame": 1, "endFrame": 367, "nz": 6,
             "ext": "png", "anno_path": "anno/vessel_0003.txt", "object_class": "vehicle"}]

        return sequence_info_list

class VesselTrainDataset(VesselDataset):
    def __init__(self):
        super().__init__()

    def _get_sequence_info_list(self):
        
        sequence_info_list = [
            {"name": "vessel_0001", "path": "data_seq/vessel_0001", "startFrame": 1, "endFrame": 243, "nz": 6,
             "ext": "png", "anno_path": "anno/vessel_0001.txt", "object_class": "vehicle"},
            {"name": "vessel_0002", "path": "data_seq/vessel_0002", "startFrame": 1, "endFrame": 342, "nz": 6,
             "ext": "png", "anno_path": "anno/vessel_0002.txt", "object_class": "vehicle"},
            {"name": "vessel_0003", "path": "data_seq/vessel_0003", "startFrame": 1, "endFrame": 367, "nz": 6,
             "ext": "png", "anno_path": "anno/vessel_0003.txt", "object_class": "vehicle"}]

        return sequence_info_list