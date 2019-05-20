def create_sh(RANDOM_SEED_NUMBER, PSC_WEIGHT_NUMBER):
    sh_text = """#!/bin/bash

#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time 12:00:00
#SBATCH --account def-bengioy
#SBATCH --output="sidewalk_seedRANDOM_SEED_NUMBER_PSC_WEIGHT_NUMBER"
#chmod +x main_recurrent.py

module load python
module load cuda/10.0.130

#source $HOME/.bashrc

#source activate observe
#pip3 install -e .
#export BABYAI_DONE_ACTIONS=1
source $HOME/ENV_observe/bin/activate
cd ..
/project/def-bengioy/ai/bin/xvfb-run -d -n 4000 -s "-screen 0 1024x768x24 -ac -noreset" python -u  main_full.py --algo ppo --seed RANDOM_SEED_NUMBER --num-processes 16 --num-steps 150 --lr 0.00005 --env-name_dem MiniWorld-SidewalkAddict-v0 --env-name_agent MiniWorld-Sidewalk-v0 --pscWeight PSC_WEIGHT_NUMBER --nameDemonstrator test2 --num-frames 1500000 --useNeural 1 --loadNeural test

echo 'DONE'"""

    sh_text_updated = sh_text.replace('RANDOM_SEED_NUMBER', str(RANDOM_SEED_NUMBER)).replace('PSC_WEIGHT_NUMBER',
                                                                                             str(PSC_WEIGHT_NUMBER))

    return sh_text_updated


def create_sh_Maze(RANDOM_SEED_NUMBER, PSC_WEIGHT_NUMBER):
    sh_text = """#!/bin/bash

#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time 12:00:00
#SBATCH --account def-bengioy
#SBATCH --output="sidewalk_seedRANDOM_SEED_NUMBER_PSC_WEIGHT_NUMBER"
#chmod +x main_recurrent.py

module load python
module load cuda/10.0.130

#source $HOME/.bashrc

#source activate observe
#pip3 install -e .
#export BABYAI_DONE_ACTIONS=1
source $HOME/ENV_observe/bin/activate
cd ..
/project/def-bengioy/ai/bin/xvfb-run -d -n 4000 -s "-screen 0 1024x768x24 -ac -noreset" python -u  main_full.py --algo ppo --seed RANDOM_SEED_NUMBER --num-processes 16 --num-steps 150 --lr 0.00005 --env-name_dem MiniWorld-TMazeAddict-v0 --env-name_agent MiniWorld-TMaze-v0 --pscWeight PSC_WEIGHT_NUMBER --nameDemonstrator testMaze --num-frames 1500000 --useNeural 1 --loadNeural test

echo 'DONE'"""

    sh_text_updated = sh_text.replace('RANDOM_SEED_NUMBER', str(RANDOM_SEED_NUMBER)).replace('PSC_WEIGHT_NUMBER',
                                                                                             str(PSC_WEIGHT_NUMBER))

    return sh_text_updated

def create_sh_FourRoom(RANDOM_SEED_NUMBER, PSC_WEIGHT_NUMBER):
    sh_text = """#!/bin/bash

#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time 12:00:00
#SBATCH --account def-bengioy
#SBATCH --output="sidewalk_seedRANDOM_SEED_NUMBER_PSC_WEIGHT_NUMBER"
#chmod +x main_recurrent.py

module load python
module load cuda/10.0.130

#source $HOME/.bashrc

#source activate observe
#pip3 install -e .
#export BABYAI_DONE_ACTIONS=1
source $HOME/ENV_observe/bin/activate
cd ..
/project/def-bengioy/ai/bin/xvfb-run -d -n 4000 -s "-screen 0 1024x768x24 -ac -noreset" python -u  main_full.py --algo ppo --seed RANDOM_SEED_NUMBER --num-processes 16 --num-steps 150 --lr 0.00005 --env-name_dem MiniWorld-FourRoomsAddict-v0 --env-name_agent MiniWorld-FourRooms-v0 --pscWeight PSC_WEIGHT_NUMBER --nameDemonstrator testFour --num-frames 1500000 --useNeural 1 --loadNeural test

echo 'DONE'"""

    sh_text_updated = sh_text.replace('RANDOM_SEED_NUMBER', str(RANDOM_SEED_NUMBER)).replace('PSC_WEIGHT_NUMBER',
                                                                                             str(PSC_WEIGHT_NUMBER))

    return sh_text_updated


def create_sh_batch(params):
    info_dict = {}
    for RANDOM_SEED_NUMBER, PSC_WEIGHT_NUMBER in params:
        info_dict['sh_dir/ex_seed{}_Psc{}.sh'.format(RANDOM_SEED_NUMBER, PSC_WEIGHT_NUMBER)] = create_sh(
            RANDOM_SEED_NUMBER, PSC_WEIGHT_NUMBER)

    for filename in info_dict:
        with open(filename, 'w') as f:
            f.write(info_dict[filename])
            
def create_sh_batch_Four(params):
    info_dict = {}
    for RANDOM_SEED_NUMBER, PSC_WEIGHT_NUMBER in params:
        info_dict['sh_dir/ex_seed{}_Psc{}.sh'.format(RANDOM_SEED_NUMBER, PSC_WEIGHT_NUMBER)] = create_sh_FourRoom(
            RANDOM_SEED_NUMBER, PSC_WEIGHT_NUMBER)
 
def create_sh_batch_Maze(params):
    info_dict = {}
    for RANDOM_SEED_NUMBER, PSC_WEIGHT_NUMBER in params:
        info_dict['sh_dir/ex_seed{}_Psc{}.sh'.format(RANDOM_SEED_NUMBER, PSC_WEIGHT_NUMBER)] = create_sh_Maze(
            RANDOM_SEED_NUMBER, PSC_WEIGHT_NUMBER)

    for filename in info_dict:
        with open(filename, 'w') as f:
            f.write(info_dict[filename])

create_sh_batch_Maze([(4851, 1),(2217, 1), (3021, 1), (790, 1), (9381, 1), (5862, 1), (6596, 1), (9789, 1), (4274, 1), (5850, 1), (7108, 0.1),(2217, 0.1), (6114, 0.1), (5664, 0.1), (7002, 0.1), (5850, 0.1), (6298, 0.1), (4513, 0.1), (638, 0.1), (118, 0.1), (2292, 0.1)])
create_sh_batch_Four([(4851, 1),(2217, 1), (3021, 1), (790, 1), (9381, 1), (5862, 1), (6596, 1), (9789, 1), (4274, 1), (5850, 1), (7108, 0.1),(2217, 0.1), (6114, 0.1), (5664, 0.1), (7002, 0.1), (5850, 0.1), (6298, 0.1), (4513, 0.1), (638, 0.1), (118, 0.1), (2292, 0.1)])
create_sh_batch([(4851, 1),(2217, 1), (3021, 1), (790, 1), (9381, 1), (5862, 1), (6596, 1), (9789, 1), (4274, 1), (5850, 1), (7108, 0.1),(2217, 0.1), (6114, 0.1), (5664, 0.1), (7002, 0.1), (5850, 0.1), (6298, 0.1), (4513, 0.1), (638, 0.1), (118, 0.1), (2292, 0.1)])
