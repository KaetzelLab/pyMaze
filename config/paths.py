import os

top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Top level pyControl folder.

config_dir      = os.path.join(top_dir, 'config')
framework_dir   = os.path.join(top_dir, 'pyControl')
devices_dir     = os.path.join(top_dir, 'devices')
tasks_dir       = os.path.join(top_dir, 'tasks')
data_dir        = os.path.join(top_dir, 'data')
models_dir      = os.path.join(top_dir, 'models')
zones_dir       = os.path.join(top_dir, 'zones_config')
video_dir       = os.path.join(top_dir, 'videos')
transfer_dir    = None # Folder to copy data to at end of run experiment.

