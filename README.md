# SCVA-AGIQA
# Text-Visual Semantic Constrained AI-Generated Image Quality Assessment


## DATASETS

### 1. Download datasets

| Dataset     |                             Link                             |
| ----------- | :----------------------------------------------------------: |
| AGIQA-1K    | [download](https://github.com/lcysyzxdxc/AGIQA-1k-Database)  |
| AGIQA-3K    | [download](https://github.com/lcysyzxdxc/AGIQA-3k-Database.) |
| AIGCIQA2023 |   [download](https://github.com/wangjiarui153/AIGCIQA2023)   |
| AIGIQA-20K |   [download](https://github.com/wangjiarui153/AIGCIQA2023)   |

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
```

### 3. Generate descriptive prompts(optional)

If you want to generate descriptive prompts on your own, we provide a code example named `chat_with_doubao.py` based on the AGIQA-1K dataset. The same principle applies to other datasets. The API can be obtained from this link https://www.volcengine.com/experience/ark


## Usage

```shell
Environment: Python 3.10.15 cuda11.8
```

### 1. Code Acquisition

```shell
git clone https://github.com/mozhu1/SC-AGIQA.git
cd ./SC-AGIQA-main
```

### 2. Dependency Installation 

```shell
conda create -n sc_agiqa python=3.10.13 -y
conda activate sc_agiqa
pip install -r requirements.txt
```

### 3. Configuration Setup 

Before running the experiment, modify the `DATA_PATH` in your configuration file (e.g., `/configs/agiqa1k.yaml`) to point to your actual dataset location, then ensure `main.py` is configured to load this specific YAML file for parameter settings.

### 4. Hugging Face Authentication  

```shell
huggingface-cli login --token <your_token>
```

### 5. Train and test

```shell
python main.py
```






