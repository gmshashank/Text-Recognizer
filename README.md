# Text-Recognizer

Training MLP on MNIST data

## Directory Structure
```sh
tree -I "logs|__pycache__"
```
.
├── LICENSE
├── README.md
├── text_recognizer
│   ├── __init__.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── base_data_module.py
│   │   ├── mnist.py
│   │   └── util.py
│   ├── lit_models
│   │   ├── __init__.py
│   │   └── base.py
│   ├── models
│   │   ├── __init__.py
│   │   └── mlp.py
│   └── util.py
└── training
    ├── __init__.py
    └── run_experiment.py


## Training

```sh
python training/run_experiment.py --model_class=MLP --data_class=MNIST --max_epochs=20 --gpus=-1
```