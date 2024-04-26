#!/bin/bash
#SBATCH --time 2:00:00
#SBATCH --mem=4G
#SBATCH --account rrg-pbellec

source env/dev/bin/activate

python src/data/assemble_hcp.py