import os
import json
from collections import defaultdict
from datetime import datetime


class FrameDataLogger:
    def __init__(self):
        self.file_path = None
        self.data_file = None

    def open_data_file(self, data_dir, experiment_name, experimenter_name, subject_ID, subject_group,
                       subject_subgroup, config_file, croping_details, datetime_now=None):
        """Open data file and write header information."""
        vid_information = defaultdict(dict)
        self.data_dir = data_dir
        self.experiment_name = experiment_name
        self.subject_ID = subject_ID
        self.experimenter_name = experimenter_name
        self.subject_group = subject_group
        self.subject_subgroup = subject_subgroup
        self.config_file = config_file
        self.croping_details = croping_details
        if datetime_now is None: datetime_now = datetime.now()
        file_name = os.path.join(
            str(self.subject_ID) + '_video_data_' + str(datetime_now.strftime('-%Y-%m-%d-%H%M%S')) + '.txt')
        self.file_path = os.path.join(self.data_dir, file_name)
        self.data_file = open(self.file_path, 'w', newline='\n')
        vid_information['Experiment'] = self.experiment_name
        vid_information['Experimenter'] = self.experimenter_name
        vid_information['Subject ID'] = self.subject_ID
        vid_information['Group'] = self.subject_group
        vid_information['Sub Group'] = self.subject_subgroup
        vid_information['Start time'] = datetime_now.strftime('%Y/%m/%d %H:%M:%S')
        self.data_file.write('B {}\n\n'.format(json.dumps(vid_information)))
        self.data_file.write('M {}\n\n'.format(json.dumps(self.config_file)))
        self.data_file.write('V {}\n\n'.format(json.dumps(croping_details)))

    def close_files(self):
        if self.data_file:
            self.data_file.close()
            self.data_file = None
            self.file_path = None

    def write_to_file(self, frames, mcu_frames, state_name, timestamps, velocity, location, pose_array):
        vid_information = defaultdict(dict)
        vid_information['frames'] = frames
        vid_information['mcu_frame'] = mcu_frames
        vid_information['state_name'] = state_name
        vid_information['timestamps'] = timestamps
        vid_information['speed'] = velocity
        vid_information['location'] = location
        vid_information['pose_array'] = pose_array
        if vid_information:
            self.data_file.write('{} \n'.format(json.dumps(vid_information)))
            self.data_file.flush()

