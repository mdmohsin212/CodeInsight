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