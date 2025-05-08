#!/bin/bash
# change to your anaconda path
source /Users/huyu/anaconda3/etc/profile.d/conda.sh
conda activate continuity_env
# change to your work path
cd /Users/huyu/Desktop/continuity

python3 -u 1.model_tuning.py -s Settings/Tuning-ROS-XG/tuning_fullset_full.json
python3 2.model_parameter_checking.py -s Settings/Tuning-ROS-XG/tuning_fullset_full.json
python3 3.model_bootstrapping.py -s Settings/Tuning-ROS-XG/tuning_fullset_full.json -g Settings/GroupFairnessParams/Blank.json -m Settings/MLParams
python3 4.model_assessment.py -s Settings/Tuning-ROS-XG/tuning_fullset_full.json -g Settings/GroupFairnessParams/Old_Young.json -m Settings/MLParams
python3 4.model_assessment.py -s Settings/Tuning-ROS-XG/tuning_fullset_full.json -g Settings/GroupFairnessParams/Black_White.json -m Settings/MLParams
python3 4.model_assessment.py -s Settings/Tuning-ROS-XG/tuning_fullset_full.json -g Settings/GroupFairnessParams//Hispanic_White.json -m Settings/MLParams
python3 4.model_assessment.py -s Settings/Tuning-ROS-XG/tuning_fullset_full.json -g Settings/GroupFairnessParams/Gender.json -m Settings/MLParams