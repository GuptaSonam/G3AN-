import ffmpeg
import argparse
import os
import numpy
import pandas
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--src_dir', type=str, default='/media/vplab/Sonam_HDD/sonam-arti/Unconditional_Video_genration/data/weizmann_action/merged', help='')
parser.add_argument('--dst_dir', type=str, default='/media/vplab/Sonam_HDD/sonam-arti/Unconditional_Video_genration/data/weizmann_action/merged_flip', help='')
args = parser.parse_args()


path = args.src_dir
files = os.listdir(path)
n_files= len(files)

for i in range(n_files):
    fname = files[i].split('.')
    video_path_src = os.path.join(path, files[i])
    video_path_dst = os.path.join(args.dst_dir, fname[0] + '_flip.avi')
    stream = ffmpeg.input(video_path_src)
    stream = ffmpeg.hflip(stream)
    stream = ffmpeg.output(stream, video_path_dst)
    ffmpeg.run(stream)
