This project was created for learn this things:
1) base concept of machine learning(creating, training, ...)
2) try to practice my own knowledge in OOP paradigm
3) work with images
4) work with directories and parsing
5) linter flate8
6) typifier mypy

This neural network decides the task of determining the number in the image size 28x28
dataset from kaggle(with changed direcrories): https://www.kaggle.com/datasets/scolianni/mnistasjpg?select=testSet
to_numpy func from habr: https://habr.com/ru/articles/545850/

project tree file:
```bash
.
├── data
│   ├── testSample
│   │   └── testSample
│   ├── testSet
│   │   └── testSet
│   ├── trainingSample
│   │   └── trainingSample
│   └── trainingSet
│       └── trainingSet
├── main.py
├── readme.md
├── req.txt
└── tools.py
```
and results folder will created by programm

How to use (commands for linux users):
1) create venv and install all dependencies from req.py 
'''
python -m venv venv
source ./venv/bin/activate
pip install -r req.txt
'''
2) open file main.py and choose:
debug - True(smaller dataset for train and test), False(full dataset)
atributes for training(epochs, alpha and creating visualization of weights every 100 epochs)

3) run main.py
'''
python main.py
'''
4) check results folder

now the neural network is not working well, even if we increase the number of epochs, in subsequent comments I will refine the training system and improve accuracy

end :)