import os
from pathlib import Path

project_name = "codeInsight"

list_of_files = [
    f"{project_name}/models/__init__.py",
    f"{project_name}/models/model_loader.py",
    f"{project_name}/models/peft_trainer.py",
    
    f"{project_name}/training/__init__.py",
    f"{project_name}/training/train.py",
    
    f"{project_name}/evaluation/__init__.py",
    f"{project_name}/evaluation/evaluator.py", 
    
    f"{project_name}/inference/__init__.py",
    f"{project_name}/inference/code_assistant.py",
    
    f"{project_name}/data/__init__.py",
    f"{project_name}/data/dataset_builder.py",
    
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/config.py",
    
    f"{project_name}/safety/__init__.py",
    f"{project_name}/safety/safety_checker.py", 
    
    f"{project_name}/exception/__init__.py",
    f"{project_name}/logger/__init__.py",
    
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/training_pipeline.py",
    f"{project_name}/pipeline/prediction_pipeline.py",
    
    "app.py",
    "Demo.py",
    "requirements.txt",
    "Dockerfile",
    "setup.py",
    ".gitignore",
    "README.md",
    "config/model.yaml",
    "config/schema.yaml",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    
    if not filepath.exists() or filepath.stat().st_size == 0:
        filepath.touch()
    else:
        print(f'{filename} is already present in {filedir} and has some content. Skipping creation.')
        


# fine-tuned-llm-code-assistant/
# │── README.md                # Project overview, setup, usage instructions
# │── requirements.txt         # Python dependencies
# │── environment.yml          # (Optional) Conda environment file
# │── LICENSE                  # Open-source license
# │── .gitignore               # Ignore cache, checkpoints, etc.
# │── Dockerfile               # Containerization
# │── docker-compose.yml       # (Optional) For multi-service setups
# │── config/
# │   ├── training_config.yaml # Hyperparams, LoRA/PEFT configs
# │   ├── model_config.yaml    # Model settings, quantization options
# │   └── safety_rules.json    # Rules for unsafe code filtering
# │
# │── data/
# │   ├── raw/                 # Original datasets
# │   ├── processed/           # Preprocessed datasets
# │   └── README.md            # Notes on dataset source/prep
# │
# │── notebooks/
# │   ├── 01_data_preparation.ipynb
# │   ├── 02_fine_tuning.ipynb
# │   ├── 03_evaluation.ipynb
# │   └── 04_demo_experiments.ipynb
# │
# │── src/
# │   ├── __init__.py
# │   ├── data/                # Data loading & preprocessing
# │   │   ├── __init__.py
# │   │   └── dataset_builder.py
# │   ├── models/              # LLM & LoRA wrappers
# │   │   ├── __init__.py
# │   │   ├── model_loader.py
# │   │   └── peft_trainer.py
# │   ├── training/            # Training scripts
# │   │   ├── __init__.py
# │   │   └── train.py
# │   ├── evaluation/          # Evaluation metrics
# │   │   ├── __init__.py
# │   │   └── evaluator.py
# │   ├── inference/           # Inference & reasoning pipeline
# │   │   ├── __init__.py
# │   │   └── code_assistant.py
# │   └── safety/              # Code safety filter
# │       ├── __init__.py
# │       └── safety_checker.py
# │
# │── app/
# │   ├── streamlit_app.py     # Web IDE frontend
# │   ├── utils.py             # UI helpers
# │   └── assets/              # Logos, UI images
# │
# │── tests/
# │   ├── test_dataset.py
# │   ├── test_model.py
# │   ├── test_inference.py
# │   └── test_safety.py
# │
# │── deployment/
# │   ├── hf_space/            # Hugging Face Spaces configs
# │   │   ├── app.py
# │   │   └── requirements.txt
# │   └── ci_cd.yml            # GitHub Actions for auto-build
# │
# └── docs/
#     ├── architecture.md      # Model & system architecture
#     ├── explainability.md    # How reasoning is generated
#     └── screenshots/         # Demo UI screenshots