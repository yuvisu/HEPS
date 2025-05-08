# Project Title

HEPS project --> Link to the paper "The impact of electronic health records (EHR) data-continuity on prediction model fairness and racial-ethnic disparities"

## Description

Codebase for the HEPS project.

## Getting Started

### Dependencies

environment.yml

### Installing

Environment: install anaconda and run the command to install the dependencies

```
conda env create -f environment.yml
```

### Executing program

Before executing the program, you need to change the "root path" and "conda environment path" in the setting file "Settings/Tuning-ROS-XG/tuning_fullset_full.json" and "main.sh" file

Then you may run the following items sequentially to process the data.
```
./1. model_tuning.py
./2. model_parameter_checking.py
./3. model_bootstrapping.py
./4. model_assessment.py

```

Or you can just run the main.sh file, it will go through from Step 1 to 4.

# 1.model_tuning.py

Purpose: Automatically processes each outcome across all years and fits the corresponding model as specified in the settings (e.g., "Settings/Tuning-ROS-XG").

Output: Tuned models saved in the Models directory, Cross-validation results stored in Output/CV, SHAP interpretation results saved to Output/SHAP

# 2.model_parameter_checking.py

Purpose: Loads the models created in Step 1 and identifies the optimal parameters for each model.

Outputs: Best model parameters saved as JSON files under Settings/MLParams

# 3.model_bootstrapping.py

Purpose: Performs bootstrapping using the optimal parameters identified in Step 2.

Outputs: Performance metrics stored in Output/SCORE, Statistical summaries saved in the statistic directory

# 4.model_assessment.py

Purpose: Evaluates model fairness across different groups.

Outputs: Fairness scores saved in the Output/SCORE_fairness directory

# 5. Statistic
Use the notebook Statistic/format_results.ipynb to compile and view all summary results. This file is to use check the results without implementing training/testing splicing.

## Help

```
Modify the json files in the Settings to build your experiments.

The dataset provided is mock data (100 samples), located in the preprocessed folder. This synthetic data mimics the distribution of the original dataset, which is derived from real-world sources. We don't provide the original data as it must be requested via proper channels.

HEPS calculation function is in util/measure.get_mpec

```

## Authors

Contributors names and contact info
[@YuHu](https://huyu.vercel.app)
[@YuHuang](https://yuvisu.github.io/)

## Version History

* 0.1
    * Initial Release
    * This is for review purposes
    
## License

MIT License (https://opensource.org/licenses/MIT)








