# NeoDataset Augmentation - Implementation Status

## ✅ Completed Components

### 1. Core Pipeline Modules
All research-backed weak supervision modules have been implemented:

- **neodataset_loader.py**: Downloads NeoDataset from HuggingFace, preprocesses, and extracts features
- **labeling_functions.py**: 18 research-backed labeling functions based on ISO 29148, story point analysis, and software engineering patterns
- **weak_supervision_pipeline.py**: Snorkel-based weak supervision to aggregate noisy labels
- **cleanlab_pipeline.py**: Confident learning to detect and filter label noise

### 2. Orchestration Scripts
- **explore_neodataset.py**: EDA and preprocessing validation
- **augment_neodataset.py**: Full end-to-end augmentation pipeline

## 🎯 Next Steps

### Step 1: Install ML Dependencies
Collecting snorkel==0.9.9
  Downloading snorkel-0.9.9-py3-none-any.whl.metadata (9.7 kB)
Collecting cleanlab==2.6.0
  Downloading cleanlab-2.6.0-py3-none-any.whl.metadata (23 kB)
Collecting datasets==2.14.0
  Downloading datasets-2.14.0-py3-none-any.whl.metadata (19 kB)
Collecting munkres>=1.0.6 (from snorkel==0.9.9)
  Downloading munkres-1.1.4-py2.py3-none-any.whl.metadata (980 bytes)
Collecting numpy>=1.16.5 (from snorkel==0.9.9)
  Downloading numpy-2.3.5-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (62 kB)
Collecting scipy>=1.2.0 (from snorkel==0.9.9)
  Downloading scipy-1.16.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (62 kB)
Collecting pandas>=1.0.0 (from snorkel==0.9.9)
  Downloading pandas-2.3.3-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl.metadata (91 kB)
Requirement already satisfied: tqdm>=4.33.0 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from snorkel==0.9.9) (4.67.1)
Collecting scikit-learn>=0.20.2 (from snorkel==0.9.9)
  Downloading scikit_learn-1.7.2-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (11 kB)
Collecting torch>=1.2.0 (from snorkel==0.9.9)
  Downloading torch-2.9.1-cp312-cp312-manylinux_2_28_x86_64.whl.metadata (30 kB)
Collecting tensorboard>=2.9.1 (from snorkel==0.9.9)
  Downloading tensorboard-2.20.0-py3-none-any.whl.metadata (1.8 kB)
Collecting networkx>=2.2 (from snorkel==0.9.9)
  Downloading networkx-3.5-py3-none-any.whl.metadata (6.3 kB)
Requirement already satisfied: termcolor>=2.4.0 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from cleanlab==2.6.0) (3.2.0)
Collecting pyarrow>=8.0.0 (from datasets==2.14.0)
  Downloading pyarrow-22.0.0-cp312-cp312-manylinux_2_28_x86_64.whl.metadata (3.2 kB)
Collecting dill<0.3.8,>=0.3.0 (from datasets==2.14.0)
  Downloading dill-0.3.7-py3-none-any.whl.metadata (9.9 kB)
Requirement already satisfied: requests>=2.19.0 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from datasets==2.14.0) (2.32.4)
Collecting xxhash (from datasets==2.14.0)
  Downloading xxhash-3.6.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (13 kB)
Collecting multiprocess (from datasets==2.14.0)
  Downloading multiprocess-0.70.18-py312-none-any.whl.metadata (7.5 kB)
Collecting fsspec>=2021.11.1 (from fsspec[http]>=2021.11.1->datasets==2.14.0)
  Downloading fsspec-2025.10.0-py3-none-any.whl.metadata (10 kB)
Collecting aiohttp (from datasets==2.14.0)
  Using cached aiohttp-3.13.2-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (8.1 kB)
Collecting huggingface-hub<1.0.0,>=0.14.0 (from datasets==2.14.0)
  Downloading huggingface_hub-0.36.0-py3-none-any.whl.metadata (14 kB)
Requirement already satisfied: packaging in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from datasets==2.14.0) (25.0)
Requirement already satisfied: pyyaml>=5.1 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from datasets==2.14.0) (6.0.3)
Collecting filelock (from huggingface-hub<1.0.0,>=0.14.0->datasets==2.14.0)
  Downloading filelock-3.20.0-py3-none-any.whl.metadata (2.1 kB)
Requirement already satisfied: typing-extensions>=3.7.4.3 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from huggingface-hub<1.0.0,>=0.14.0->datasets==2.14.0) (4.15.0)
Collecting hf-xet<2.0.0,>=1.1.3 (from huggingface-hub<1.0.0,>=0.14.0->datasets==2.14.0)
  Downloading hf_xet-1.2.0-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.9 kB)
Collecting aiohappyeyeballs>=2.5.0 (from aiohttp->datasets==2.14.0)
  Using cached aiohappyeyeballs-2.6.1-py3-none-any.whl.metadata (5.9 kB)
Collecting aiosignal>=1.4.0 (from aiohttp->datasets==2.14.0)
  Using cached aiosignal-1.4.0-py3-none-any.whl.metadata (3.7 kB)
Collecting attrs>=17.3.0 (from aiohttp->datasets==2.14.0)
  Using cached attrs-25.4.0-py3-none-any.whl.metadata (10 kB)
Collecting frozenlist>=1.1.1 (from aiohttp->datasets==2.14.0)
  Using cached frozenlist-1.8.0-cp312-cp312-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl.metadata (20 kB)
Collecting multidict<7.0,>=4.5 (from aiohttp->datasets==2.14.0)
  Using cached multidict-6.7.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (5.3 kB)
Collecting propcache>=0.2.0 (from aiohttp->datasets==2.14.0)
  Using cached propcache-0.4.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (13 kB)
Collecting yarl<2.0,>=1.17.0 (from aiohttp->datasets==2.14.0)
  Using cached yarl-1.22.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (75 kB)
Requirement already satisfied: idna>=2.0 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from yarl<2.0,>=1.17.0->aiohttp->datasets==2.14.0) (3.10)
Requirement already satisfied: python-dateutil>=2.8.2 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from pandas>=1.0.0->snorkel==0.9.9) (2.9.0.post0)
Collecting pytz>=2020.1 (from pandas>=1.0.0->snorkel==0.9.9)
  Downloading pytz-2025.2-py2.py3-none-any.whl.metadata (22 kB)
Collecting tzdata>=2022.7 (from pandas>=1.0.0->snorkel==0.9.9)
  Downloading tzdata-2025.2-py2.py3-none-any.whl.metadata (1.4 kB)
Requirement already satisfied: six>=1.5 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from python-dateutil>=2.8.2->pandas>=1.0.0->snorkel==0.9.9) (1.17.0)
Requirement already satisfied: charset_normalizer<4,>=2 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from requests>=2.19.0->datasets==2.14.0) (3.4.2)
Requirement already satisfied: urllib3<3,>=1.21.1 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from requests>=2.19.0->datasets==2.14.0) (2.5.0)
Requirement already satisfied: certifi>=2017.4.17 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from requests>=2.19.0->datasets==2.14.0) (2025.8.3)
Collecting joblib>=1.2.0 (from scikit-learn>=0.20.2->snorkel==0.9.9)
  Downloading joblib-1.5.2-py3-none-any.whl.metadata (5.6 kB)
Collecting threadpoolctl>=3.1.0 (from scikit-learn>=0.20.2->snorkel==0.9.9)
  Downloading threadpoolctl-3.6.0-py3-none-any.whl.metadata (13 kB)
Collecting absl-py>=0.4 (from tensorboard>=2.9.1->snorkel==0.9.9)
  Downloading absl_py-2.3.1-py3-none-any.whl.metadata (3.3 kB)
Collecting grpcio>=1.48.2 (from tensorboard>=2.9.1->snorkel==0.9.9)
  Downloading grpcio-1.76.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (3.7 kB)
Requirement already satisfied: markdown>=2.6.8 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from tensorboard>=2.9.1->snorkel==0.9.9) (3.10)
Collecting pillow (from tensorboard>=2.9.1->snorkel==0.9.9)
  Downloading pillow-12.0.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.8 kB)
Collecting protobuf!=4.24.0,>=3.19.6 (from tensorboard>=2.9.1->snorkel==0.9.9)
  Downloading protobuf-6.33.1-cp39-abi3-manylinux2014_x86_64.whl.metadata (593 bytes)
Requirement already satisfied: setuptools>=41.0.0 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from tensorboard>=2.9.1->snorkel==0.9.9) (80.9.0)
Collecting tensorboard-data-server<0.8.0,>=0.7.0 (from tensorboard>=2.9.1->snorkel==0.9.9)
  Downloading tensorboard_data_server-0.7.2-py3-none-manylinux_2_31_x86_64.whl.metadata (1.1 kB)
Collecting werkzeug>=1.0.1 (from tensorboard>=2.9.1->snorkel==0.9.9)
  Using cached werkzeug-3.1.3-py3-none-any.whl.metadata (3.7 kB)
Collecting sympy>=1.13.3 (from torch>=1.2.0->snorkel==0.9.9)
  Downloading sympy-1.14.0-py3-none-any.whl.metadata (12 kB)
Requirement already satisfied: jinja2 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from torch>=1.2.0->snorkel==0.9.9) (3.1.6)
Collecting nvidia-cuda-nvrtc-cu12==12.8.93 (from torch>=1.2.0->snorkel==0.9.9)
  Downloading nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cuda-runtime-cu12==12.8.90 (from torch>=1.2.0->snorkel==0.9.9)
  Downloading nvidia_cuda_runtime_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cuda-cupti-cu12==12.8.90 (from torch>=1.2.0->snorkel==0.9.9)
  Downloading nvidia_cuda_cupti_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cudnn-cu12==9.10.2.21 (from torch>=1.2.0->snorkel==0.9.9)
  Downloading nvidia_cudnn_cu12-9.10.2.21-py3-none-manylinux_2_27_x86_64.whl.metadata (1.8 kB)
Collecting nvidia-cublas-cu12==12.8.4.1 (from torch>=1.2.0->snorkel==0.9.9)
  Downloading nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cufft-cu12==11.3.3.83 (from torch>=1.2.0->snorkel==0.9.9)
  Downloading nvidia_cufft_cu12-11.3.3.83-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-curand-cu12==10.3.9.90 (from torch>=1.2.0->snorkel==0.9.9)
  Downloading nvidia_curand_cu12-10.3.9.90-py3-none-manylinux_2_27_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cusolver-cu12==11.7.3.90 (from torch>=1.2.0->snorkel==0.9.9)
  Downloading nvidia_cusolver_cu12-11.7.3.90-py3-none-manylinux_2_27_x86_64.whl.metadata (1.8 kB)
Collecting nvidia-cusparse-cu12==12.5.8.93 (from torch>=1.2.0->snorkel==0.9.9)
  Downloading nvidia_cusparse_cu12-12.5.8.93-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.8 kB)
Collecting nvidia-cusparselt-cu12==0.7.1 (from torch>=1.2.0->snorkel==0.9.9)
  Downloading nvidia_cusparselt_cu12-0.7.1-py3-none-manylinux2014_x86_64.whl.metadata (7.0 kB)
Collecting nvidia-nccl-cu12==2.27.5 (from torch>=1.2.0->snorkel==0.9.9)
  Downloading nvidia_nccl_cu12-2.27.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (2.0 kB)
Collecting nvidia-nvshmem-cu12==3.3.20 (from torch>=1.2.0->snorkel==0.9.9)
  Downloading nvidia_nvshmem_cu12-3.3.20-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (2.1 kB)
Collecting nvidia-nvtx-cu12==12.8.90 (from torch>=1.2.0->snorkel==0.9.9)
  Downloading nvidia_nvtx_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.8 kB)
Collecting nvidia-nvjitlink-cu12==12.8.93 (from torch>=1.2.0->snorkel==0.9.9)
  Downloading nvidia_nvjitlink_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cufile-cu12==1.13.1.3 (from torch>=1.2.0->snorkel==0.9.9)
  Downloading nvidia_cufile_cu12-1.13.1.3-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.7 kB)
Collecting triton==3.5.1 (from torch>=1.2.0->snorkel==0.9.9)
  Downloading triton-3.5.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (1.7 kB)
Collecting mpmath<1.4,>=1.1.0 (from sympy>=1.13.3->torch>=1.2.0->snorkel==0.9.9)
  Downloading mpmath-1.3.0-py3-none-any.whl.metadata (8.6 kB)
Requirement already satisfied: MarkupSafe>=2.1.1 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from werkzeug>=1.0.1->tensorboard>=2.9.1->snorkel==0.9.9) (3.0.3)
INFO: pip is looking at multiple versions of multiprocess to determine which version is compatible with other requirements. This could take a while.
Collecting multiprocess (from datasets==2.14.0)
  Downloading multiprocess-0.70.17-py312-none-any.whl.metadata (7.2 kB)
  Downloading multiprocess-0.70.16-py312-none-any.whl.metadata (7.2 kB)
  Downloading multiprocess-0.70.15-py311-none-any.whl.metadata (7.2 kB)
Downloading snorkel-0.9.9-py3-none-any.whl (103 kB)
Downloading cleanlab-2.6.0-py3-none-any.whl (325 kB)
Downloading datasets-2.14.0-py3-none-any.whl (492 kB)
Downloading dill-0.3.7-py3-none-any.whl (115 kB)
Downloading huggingface_hub-0.36.0-py3-none-any.whl (566 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 566.1/566.1 kB 28.5 MB/s  0:00:00

Downloading hf_xet-1.2.0-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.3/3.3 MB 143.2 MB/s  0:00:00

Downloading fsspec-2025.10.0-py3-none-any.whl (200 kB)
Using cached aiohttp-3.13.2-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (1.8 MB)
Using cached multidict-6.7.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (256 kB)
Using cached yarl-1.22.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (377 kB)
Using cached aiohappyeyeballs-2.6.1-py3-none-any.whl (15 kB)
Using cached aiosignal-1.4.0-py3-none-any.whl (7.5 kB)
Using cached attrs-25.4.0-py3-none-any.whl (67 kB)
Using cached frozenlist-1.8.0-cp312-cp312-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl (242 kB)
Downloading munkres-1.1.4-py2.py3-none-any.whl (7.0 kB)
Downloading networkx-3.5-py3-none-any.whl (2.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.0/2.0 MB 116.9 MB/s  0:00:00

Downloading numpy-2.3.5-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (16.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.6/16.6 MB 225.8 MB/s  0:00:00

Downloading pandas-2.3.3-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl (12.4 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.4/12.4 MB 248.3 MB/s  0:00:00

Using cached propcache-0.4.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (221 kB)
Downloading pyarrow-22.0.0-cp312-cp312-manylinux_2_28_x86_64.whl (47.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 47.7/47.7 MB 214.8 MB/s  0:00:00

Downloading pytz-2025.2-py2.py3-none-any.whl (509 kB)
Downloading scikit_learn-1.7.2-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (9.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 9.5/9.5 MB 216.8 MB/s  0:00:00

Downloading joblib-1.5.2-py3-none-any.whl (308 kB)
Downloading scipy-1.16.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (35.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 35.7/35.7 MB 238.2 MB/s  0:00:00

Downloading tensorboard-2.20.0-py3-none-any.whl (5.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.5/5.5 MB 210.1 MB/s  0:00:00

Downloading tensorboard_data_server-0.7.2-py3-none-manylinux_2_31_x86_64.whl (6.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.6/6.6 MB 209.8 MB/s  0:00:00

Downloading absl_py-2.3.1-py3-none-any.whl (135 kB)
Downloading grpcio-1.76.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (6.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.6/6.6 MB 196.0 MB/s  0:00:00

Downloading protobuf-6.33.1-cp39-abi3-manylinux2014_x86_64.whl (323 kB)
Downloading threadpoolctl-3.6.0-py3-none-any.whl (18 kB)
Downloading torch-2.9.1-cp312-cp312-manylinux_2_28_x86_64.whl (899.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 899.7/899.7 MB 35.3 MB/s  0:00:09

Downloading nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_x86_64.whl (594.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 594.3/594.3 MB 53.9 MB/s  0:00:05

Downloading nvidia_cuda_cupti_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (10.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10.2/10.2 MB 133.2 MB/s  0:00:00

Downloading nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl (88.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 88.0/88.0 MB 114.3 MB/s  0:00:00

Downloading nvidia_cuda_runtime_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (954 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 954.8/954.8 kB 100.0 MB/s  0:00:00

Downloading nvidia_cudnn_cu12-9.10.2.21-py3-none-manylinux_2_27_x86_64.whl (706.8 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 706.8/706.8 MB 50.4 MB/s  0:00:06

Downloading nvidia_cufft_cu12-11.3.3.83-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (193.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 193.1/193.1 MB 131.9 MB/s  0:00:01

Downloading nvidia_cufile_cu12-1.13.1.3-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (1.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 106.4 MB/s  0:00:00

Downloading nvidia_curand_cu12-10.3.9.90-py3-none-manylinux_2_27_x86_64.whl (63.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 63.6/63.6 MB 132.7 MB/s  0:00:00

Downloading nvidia_cusolver_cu12-11.7.3.90-py3-none-manylinux_2_27_x86_64.whl (267.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 267.5/267.5 MB 129.4 MB/s  0:00:02

Downloading nvidia_cusparse_cu12-12.5.8.93-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (288.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 288.2/288.2 MB 122.0 MB/s  0:00:02

Downloading nvidia_cusparselt_cu12-0.7.1-py3-none-manylinux2014_x86_64.whl (287.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 287.2/287.2 MB 122.7 MB/s  0:00:02

Downloading nvidia_nccl_cu12-2.27.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (322.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 322.3/322.3 MB 113.9 MB/s  0:00:02

Downloading nvidia_nvjitlink_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl (39.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 39.3/39.3 MB 133.9 MB/s  0:00:00

Downloading nvidia_nvshmem_cu12-3.3.20-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (124.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 124.7/124.7 MB 131.3 MB/s  0:00:00

Downloading nvidia_nvtx_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (89 kB)
Downloading triton-3.5.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (170.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 170.5/170.5 MB 132.3 MB/s  0:00:01

Downloading sympy-1.14.0-py3-none-any.whl (6.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.3/6.3 MB 145.4 MB/s  0:00:00

Downloading mpmath-1.3.0-py3-none-any.whl (536 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 536.2/536.2 kB 40.1 MB/s  0:00:00

Downloading tzdata-2025.2-py2.py3-none-any.whl (347 kB)
Using cached werkzeug-3.1.3-py3-none-any.whl (224 kB)
Downloading filelock-3.20.0-py3-none-any.whl (16 kB)
Downloading multiprocess-0.70.15-py311-none-any.whl (135 kB)
Downloading pillow-12.0.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (7.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7.0/7.0 MB 245.3 MB/s  0:00:00

Downloading xxhash-3.6.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (193 kB)
Installing collected packages: pytz, nvidia-cusparselt-cu12, munkres, mpmath, xxhash, werkzeug, tzdata, triton, threadpoolctl, tensorboard-data-server, sympy, pyarrow, protobuf, propcache, pillow, nvidia-nvtx-cu12, nvidia-nvshmem-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufile-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, numpy, networkx, multidict, joblib, hf-xet, grpcio, fsspec, frozenlist, filelock, dill, attrs, aiohappyeyeballs, absl-py, yarl, tensorboard, scipy, pandas, nvidia-cusparse-cu12, nvidia-cufft-cu12, nvidia-cudnn-cu12, multiprocess, huggingface-hub, aiosignal, scikit-learn, nvidia-cusolver-cu12, aiohttp, torch, cleanlab, snorkel, datasets


Successfully installed absl-py-2.3.1 aiohappyeyeballs-2.6.1 aiohttp-3.13.2 aiosignal-1.4.0 attrs-25.4.0 cleanlab-2.6.0 datasets-2.14.0 dill-0.3.7 filelock-3.20.0 frozenlist-1.8.0 fsspec-2025.10.0 grpcio-1.76.0 hf-xet-1.2.0 huggingface-hub-0.36.0 joblib-1.5.2 mpmath-1.3.0 multidict-6.7.0 multiprocess-0.70.15 munkres-1.1.4 networkx-3.5 numpy-2.3.5 nvidia-cublas-cu12-12.8.4.1 nvidia-cuda-cupti-cu12-12.8.90 nvidia-cuda-nvrtc-cu12-12.8.93 nvidia-cuda-runtime-cu12-12.8.90 nvidia-cudnn-cu12-9.10.2.21 nvidia-cufft-cu12-11.3.3.83 nvidia-cufile-cu12-1.13.1.3 nvidia-curand-cu12-10.3.9.90 nvidia-cusolver-cu12-11.7.3.90 nvidia-cusparse-cu12-12.5.8.93 nvidia-cusparselt-cu12-0.7.1 nvidia-nccl-cu12-2.27.5 nvidia-nvjitlink-cu12-12.8.93 nvidia-nvshmem-cu12-3.3.20 nvidia-nvtx-cu12-12.8.90 pandas-2.3.3 pillow-12.0.0 propcache-0.4.1 protobuf-6.33.1 pyarrow-22.0.0 pytz-2025.2 scikit-learn-1.7.2 scipy-1.16.3 snorkel-0.9.9 sympy-1.14.0 tensorboard-2.20.0 tensorboard-data-server-0.7.2 threadpoolctl-3.6.0 torch-2.9.1 triton-3.5.1 tzdata-2025.2 werkzeug-3.1.3 xxhash-3.6.0 yarl-1.22.0

### Step 2: Explore Dataset (Optional)


This will:
- Download NeoDataset (~20K user stories)
- Show descriptive statistics
- Save preprocessed version

### Step 3: Run Full Augmentation


This will:
1. Load and preprocess NeoDataset
2. Apply 18 labeling functions
3. Train Snorkel label model
4. Detect label noise with Cleanlab
5. Generate final augmented dataset

**Output Files:**
-  - Full labeled dataset
-  - High-confidence subset (risk_confidence > 0.75)
-  - Intermediate Snorkel output

### Step 4: Review Results
Check the label distribution and health score in the terminal output.

**Expected Metrics:**
- Label Health Score: > 0.60 (good quality)
- High-confidence examples: > 10,000 stories
- Label distribution: Roughly 30-70% split (RISK vs SAFE)

## 📊 Labeling Functions Overview

### Lexical Functions (ISO 29148 Ambiguity Patterns)
- : Subjective quality adjectives
- : Unbounded scope indicators
- : Vague quantifiers
- : Non-specific verbs
- : Unrealistic temporal assumptions

### Metadata Functions (Story Point Analysis)
- : Large story points (≥8)
- : Small story points (≤2)
- : Non-Fibonacci points

### Structural Functions (Syntactic Completeness)
- : No AC or lists
- : Has blocking dependencies
- : < 15 words
- : > 200 words

### Domain-Specific Functions
- : API/third-party work
- : Technical debt
- : Security features
- : Performance optimization
- : Bug fixes (typically safe)
- : Documentation (typically safe)

## 🔬 Scientific Methodology

This implementation follows the research documented in :

1. **Data Programming (Snorkel)**: Combines noisy heuristics via generative modeling
2. **Confident Learning (Cleanlab)**: Detects label-data mismatches
3. **Proxy Validation**: Uses independent metadata for validation
4. **Scientifically Grounded LFs**: Based on peer-reviewed research (ISO 29148, Cone of Uncertainty, etc.)

## 📝 Risk Definition

**Risk**: Likelihood of spillover, scope creep, or significant underestimation

A story is labeled **RISK** if it has characteristics that historically correlate with:
- Work not completed in sprint (spillover)
- Requirements expanding during development (scope creep)
- Actual effort significantly exceeding estimate

## 🚀 After Augmentation

Once you have the augmented dataset, you can:
1. Train an ML classifier (TF-IDF + LogReg, BERT, etc.)
2. Integrate the model into the SprintGuard PSA
3. Replace the current  with the trained model

See the main project README for integration details.

## 📄 Files Created



## 🐛 Troubleshooting

**ImportError: No module named 'snorkel'**
- Install ML dependencies: Requirement already satisfied: snorkel in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (0.9.9)
Requirement already satisfied: cleanlab in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (2.6.0)
Requirement already satisfied: datasets in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (2.14.0)
Requirement already satisfied: munkres>=1.0.6 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from snorkel) (1.1.4)
Requirement already satisfied: numpy>=1.16.5 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from snorkel) (2.3.5)
Requirement already satisfied: scipy>=1.2.0 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from snorkel) (1.16.3)
Requirement already satisfied: pandas>=1.0.0 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from snorkel) (2.3.3)
Requirement already satisfied: tqdm>=4.33.0 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from snorkel) (4.67.1)
Requirement already satisfied: scikit-learn>=0.20.2 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from snorkel) (1.7.2)
Requirement already satisfied: torch>=1.2.0 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from snorkel) (2.9.1)
Requirement already satisfied: tensorboard>=2.9.1 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from snorkel) (2.20.0)
Requirement already satisfied: networkx>=2.2 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from snorkel) (3.5)
Requirement already satisfied: termcolor>=2.4.0 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from cleanlab) (3.2.0)
Requirement already satisfied: pyarrow>=8.0.0 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from datasets) (22.0.0)
Requirement already satisfied: dill<0.3.8,>=0.3.0 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from datasets) (0.3.7)
Requirement already satisfied: requests>=2.19.0 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from datasets) (2.32.4)
Requirement already satisfied: xxhash in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from datasets) (3.6.0)
Requirement already satisfied: multiprocess in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from datasets) (0.70.15)
Requirement already satisfied: fsspec>=2021.11.1 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from fsspec[http]>=2021.11.1->datasets) (2025.10.0)
Requirement already satisfied: aiohttp in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from datasets) (3.13.2)
Requirement already satisfied: huggingface-hub<1.0.0,>=0.14.0 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from datasets) (0.36.0)
Requirement already satisfied: packaging in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from datasets) (25.0)
Requirement already satisfied: pyyaml>=5.1 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from datasets) (6.0.3)
Requirement already satisfied: filelock in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from huggingface-hub<1.0.0,>=0.14.0->datasets) (3.20.0)
Requirement already satisfied: typing-extensions>=3.7.4.3 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from huggingface-hub<1.0.0,>=0.14.0->datasets) (4.15.0)
Requirement already satisfied: hf-xet<2.0.0,>=1.1.3 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from huggingface-hub<1.0.0,>=0.14.0->datasets) (1.2.0)
Requirement already satisfied: aiohappyeyeballs>=2.5.0 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from aiohttp->datasets) (2.6.1)
Requirement already satisfied: aiosignal>=1.4.0 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from aiohttp->datasets) (1.4.0)
Requirement already satisfied: attrs>=17.3.0 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from aiohttp->datasets) (25.4.0)
Requirement already satisfied: frozenlist>=1.1.1 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from aiohttp->datasets) (1.8.0)
Requirement already satisfied: multidict<7.0,>=4.5 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from aiohttp->datasets) (6.7.0)
Requirement already satisfied: propcache>=0.2.0 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from aiohttp->datasets) (0.4.1)
Requirement already satisfied: yarl<2.0,>=1.17.0 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from aiohttp->datasets) (1.22.0)
Requirement already satisfied: idna>=2.0 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from yarl<2.0,>=1.17.0->aiohttp->datasets) (3.10)
Requirement already satisfied: python-dateutil>=2.8.2 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from pandas>=1.0.0->snorkel) (2.9.0.post0)
Requirement already satisfied: pytz>=2020.1 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from pandas>=1.0.0->snorkel) (2025.2)
Requirement already satisfied: tzdata>=2022.7 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from pandas>=1.0.0->snorkel) (2025.2)
Requirement already satisfied: six>=1.5 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from python-dateutil>=2.8.2->pandas>=1.0.0->snorkel) (1.17.0)
Requirement already satisfied: charset_normalizer<4,>=2 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from requests>=2.19.0->datasets) (3.4.2)
Requirement already satisfied: urllib3<3,>=1.21.1 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from requests>=2.19.0->datasets) (2.5.0)
Requirement already satisfied: certifi>=2017.4.17 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from requests>=2.19.0->datasets) (2025.8.3)
Requirement already satisfied: joblib>=1.2.0 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from scikit-learn>=0.20.2->snorkel) (1.5.2)
Requirement already satisfied: threadpoolctl>=3.1.0 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from scikit-learn>=0.20.2->snorkel) (3.6.0)
Requirement already satisfied: absl-py>=0.4 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from tensorboard>=2.9.1->snorkel) (2.3.1)
Requirement already satisfied: grpcio>=1.48.2 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from tensorboard>=2.9.1->snorkel) (1.76.0)
Requirement already satisfied: markdown>=2.6.8 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from tensorboard>=2.9.1->snorkel) (3.10)
Requirement already satisfied: pillow in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from tensorboard>=2.9.1->snorkel) (12.0.0)
Requirement already satisfied: protobuf!=4.24.0,>=3.19.6 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from tensorboard>=2.9.1->snorkel) (6.33.1)
Requirement already satisfied: setuptools>=41.0.0 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from tensorboard>=2.9.1->snorkel) (80.9.0)
Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from tensorboard>=2.9.1->snorkel) (0.7.2)
Requirement already satisfied: werkzeug>=1.0.1 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from tensorboard>=2.9.1->snorkel) (3.1.3)
Requirement already satisfied: sympy>=1.13.3 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from torch>=1.2.0->snorkel) (1.14.0)
Requirement already satisfied: jinja2 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from torch>=1.2.0->snorkel) (3.1.6)
Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.8.93 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from torch>=1.2.0->snorkel) (12.8.93)
Requirement already satisfied: nvidia-cuda-runtime-cu12==12.8.90 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from torch>=1.2.0->snorkel) (12.8.90)
Requirement already satisfied: nvidia-cuda-cupti-cu12==12.8.90 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from torch>=1.2.0->snorkel) (12.8.90)
Requirement already satisfied: nvidia-cudnn-cu12==9.10.2.21 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from torch>=1.2.0->snorkel) (9.10.2.21)
Requirement already satisfied: nvidia-cublas-cu12==12.8.4.1 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from torch>=1.2.0->snorkel) (12.8.4.1)
Requirement already satisfied: nvidia-cufft-cu12==11.3.3.83 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from torch>=1.2.0->snorkel) (11.3.3.83)
Requirement already satisfied: nvidia-curand-cu12==10.3.9.90 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from torch>=1.2.0->snorkel) (10.3.9.90)
Requirement already satisfied: nvidia-cusolver-cu12==11.7.3.90 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from torch>=1.2.0->snorkel) (11.7.3.90)
Requirement already satisfied: nvidia-cusparse-cu12==12.5.8.93 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from torch>=1.2.0->snorkel) (12.5.8.93)
Requirement already satisfied: nvidia-cusparselt-cu12==0.7.1 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from torch>=1.2.0->snorkel) (0.7.1)
Requirement already satisfied: nvidia-nccl-cu12==2.27.5 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from torch>=1.2.0->snorkel) (2.27.5)
Requirement already satisfied: nvidia-nvshmem-cu12==3.3.20 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from torch>=1.2.0->snorkel) (3.3.20)
Requirement already satisfied: nvidia-nvtx-cu12==12.8.90 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from torch>=1.2.0->snorkel) (12.8.90)
Requirement already satisfied: nvidia-nvjitlink-cu12==12.8.93 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from torch>=1.2.0->snorkel) (12.8.93)
Requirement already satisfied: nvidia-cufile-cu12==1.13.1.3 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from torch>=1.2.0->snorkel) (1.13.1.3)
Requirement already satisfied: triton==3.5.1 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from torch>=1.2.0->snorkel) (3.5.1)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from sympy>=1.13.3->torch>=1.2.0->snorkel) (1.3.0)
Requirement already satisfied: MarkupSafe>=2.1.1 in /home/jovyan/.conda/envs/jupyter-base/lib/python3.12/site-packages (from werkzeug>=1.0.1->tensorboard>=2.9.1->snorkel) (3.0.3)

**FileNotFoundError: data/neodataset/**
- The script will create directories automatically

**Low label health score (< 0.5)**
- This may indicate LF conflicts or dataset mismatch
- Review LF statistics in pipeline output
- Consider refining LFs based on domain knowledge

**Memory issues**
- NeoDataset is ~20K stories, may require 4GB+ RAM
- Use a subset for testing: modify loader to return 

## 📚 References

All methods are based on peer-reviewed research cited in .
