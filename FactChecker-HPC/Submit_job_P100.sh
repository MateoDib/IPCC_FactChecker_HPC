#!/bin/bash
#SBATCH -p publicgpu          # Partition publique avec des GPU
#SBATCH -N 1                  # 1 nœud
#SBATCH --exclusive           # Nœud entièrement dédié à ce job
#SBATCH --gres=gpu:4          # 4 GPU par nœud
#SBATCH --constraint="gpuv100|gpurtx6000" # Utiliser les GPU V100 ou RTX6000
#SBATCH --cpus-per-task=24    # Nombre de cœurs par tâche (ajustez selon votre besoin)
#SBATCH -t 12:00:00           # Temps limite de 12 heures
#SBATCH --mail-type=BEGIN     # Notification par e-mail au début du job
#SBATCH --mail-type=END       # Notification par e-mail à la fin du job
#SBATCH --mail-user=mateo.dib@etu.unistra.fr
#SBATCH -o /home2020/home/beta/aebeling/Mateo/Environmental_News_Checker-1/FactChecker-HPC/output_V100_RTX6000.log    # Fichier de sortie pour les logs

# Charger les modules nécessaires
module load python/python-3.11.4

# Activer l'environnement virtuel Python
source /home2020/home/beta/aebeling/python/bin/activate

nvidia-smi
python /home2020/home/beta/aebeling/Mateo/Environmental_News_Checker-1/FactChecker-HPC/Environmental_News_Checker-main/main.py

# Désactiver l'environnement virtuel
deactivate
