#!/bin/bash
#SBATCH --partition=gpu_min11gb
#SBATCH --qos=gpu_min11gb
#SBATCH --job-name=run_metrics_coco
#SBATCH --output=/nas-ctm01/homes/bfcoelho/metrics_coco_retina.out
#SBATCH --error=/nas-ctm01/homes/bfcoelho/metrics_coco_retina.err

echo "Running metrics"

python DIBA_runcoco_metrics.py -m param.beta=1000,10000,100000 db.dataset='Coco'

echo "Finished"