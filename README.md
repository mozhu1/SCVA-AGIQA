# Aggregation of Semantic Consistency and Visual Attention for AI-Generated Image Quality Assessment


## DATASETS

### 1. Download datasets

| Dataset     |                             Link                             |
| ----------- | :----------------------------------------------------------: |
| AGIQA-1K    | [download](https://github.com/lcysyzxdxc/AGIQA-1k-Database)  |
| AGIQA-3K    | [download](https://github.com/lcysyzxdxc/AGIQA-3k-Database.) |
| AIGCIQA2023 | [download](https://github.com/wangjiarui153/AIGCIQA2023)   |
| AIGIQA-20K | [download](https://www.modelscope.cn/datasets/lcysyzxdxc/AIGCQA-30K-Image)   |

### 2. Descriptive prompts

We have provided documents containing descriptive prompts in the folders of AGIQA-1K, AGIQA-3K, and AIGCIQA2023. When using them, you just need to set up the paths according to the figure above and change the dataset paths in the configuration files.

```shell 
AGIQ1K/
├── AIGC-1K_answer.csv
└── file/
    ├── image1.png
    ├── image2.png
    └──...
AGIQA3K/
├── data.csv
├── image1.jpg
├── image2.png
└──...
AIGCIQA2023K/
├── merged_output_aigciqa2023.csv
├── DATA/
│   └── MOS/
│       └── mosz1.mat
└── Image/
    ├── subfolder1/
    │   ├── image1.jpg
    │   └──...
    └──...
AIGIQA-20K/
├── all/
    ├──all_answer_prompt.csv
    ├──DALLE2_0000.png
    ├──DALLE2_0001.png
    ├──...
 


```

### 3. Generate descriptive prompts(optional)

If you want to generate descriptive prompts on your own, we provide a code example named `chat_with_doubao.py` based on the AGIQA-1K dataset. The same principle applies to other datasets. The API can be obtained from this link https://www.volcengine.com/experience/ark


## Usage

```shell
Environment: Python 3.10.15 cuda11.8
```


###  1. Dependency Installation 

```shell
conda create -n scva_agiqa python=3.10.13 -y
conda activate scva_agiqa
pip install -r requirements.txt
```

###  2. Configuration Setup 

Before running the experiment, modify the `DATA_PATH` in your configuration file (e.g., `/configs/agiqa1k.yaml`) to point to your actual dataset location, then ensure `main.py` is configured to load this specific YAML file for parameter settings.

###  3. Hugging Face Authentication  

```shell
huggingface-cli login --token <your_token>
```

###  4. Train and test

```shell
python main.py
```

## Project Structure Guide

For contributors looking to modify or extend the codebase, here are the key directories and files you should be aware of:

- **Configuration Files**: Located in `SCVA-AGIQA-main/config`  
  This directory contains all configuration settings, parameters, and environment variables used throughout the project.

- **Dataset Loading**: Found in `SCVA-AGIQA-main/datasets2`  
  All scripts related to data loading, preprocessing, and dataset management reside here. Modify these files when working with different data sources or formats.

- **Model Implementation**: Primary model code is in `SCVA-AGIQA-main/models_blip/blip.py`  
  This file contains the core implementation of the BLIP model architecture. Make changes here when adjusting model structure, adding new layers, or modifying forward/backward passes.





