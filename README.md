# Text-Recognizer

Training MLP on MNIST data

## Directory Structure
```sh
tree -I "logs|__pycache__"
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
```


## Using a convolutional network for recognizing MNIST

```sh
python training/run_experiment.py --model_class=CNN --data_class=MNIST --max_epochs=5 --gpus=1
```


## Using a convolutional network for recognizing EMNIST
```sh
python training/run_experiment.py --model_class=CNN --data_class=EMNIST --max_epochs=5 --gpus=1 --num_workers=4
```