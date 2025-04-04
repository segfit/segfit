import os
import argparse
import pdb
import subprocess
import shutil
import pickle as pkl
from pathlib import Path


def run_command(command, use_shell=False, env=None):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=use_shell,
        env=env,
        bufsize=1 
    )
    try:
        for line in process.stdout:
            print(line, end="")
        for error in process.stderr:
            print(error, end="")
    except KeyboardInterrupt:
        process.terminate()
        print("Process terminated.")
    return process.wait()


parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, required=True)
parser.add_argument("--human3d_ckpt", type=str, required=True)
parser.add_argument("--is_input_z_up", type=str, required=False, default="True")
parser.add_argument("--smplx_gt_dir", type=str, required=False, default=None)
parser.add_argument('--is_hi4d', action='store_true')
args = parser.parse_args()

input_path, human3d_ckpt, smplx_gt_dir = (
    args.input_path, args.human3d_ckpt, args.smplx_gt_dir)
is_input_z_up = args.is_input_z_up.lower() == "true"

if not os.path.exists("results"):
    os.makedirs("results")

# paths to python binaries
human3d_env_python = os.path.join(Path.home(), "miniconda3/envs/human3d_cuda113/bin/python")
pytorch3d_env_python = os.path.join(Path.home(), "miniconda3/envs/pytorch3d/bin/python")

# get segmentation from hi4d
run_command([
    human3d_env_python,
    'human3d/infer_mhbps.py',
    f'segfit.data_path={args.input_path}',
    f'general.checkpoint={args.human3d_ckpt}',
    f'segfit.is_input_z_up={args.is_input_z_up}'
], env={**os.environ, "PYTHONUNBUFFERED": "1"})
shutil.rmtree("saved")

# prepare for model fitting
if not args.is_hi4d:
    run_command([
        pytorch3d_env_python,
        'src/prepare_smplx_gts.py',
        '--smplx_gt_dir',
        args.smplx_gt_dir,
        '--is_input_z_up',
        args.is_input_z_up
    ], env={**os.environ, "PYTHONUNBUFFERED": "1"})
else:
    run_command([
        pytorch3d_env_python,
        'src/prepare_smplx_gts.py',
        '--smplx_gt_dir',
        args.smplx_gt_dir,
        '--is_input_z_up',
        args.is_input_z_up,
        '--is_hi4d'
    ], env={**os.environ, "PYTHONUNBUFFERED": "1"})

# run model fitting
if not args.is_hi4d:
    run_command([
        pytorch3d_env_python,
        'src/main.py',
    ], env={**os.environ, "PYTHONUNBUFFERED": "1"})
else:
    run_command([
        pytorch3d_env_python,
        'src/main.py',
        '--is_hi4d'
    ], env={**os.environ, "PYTHONUNBUFFERED": "1"})
