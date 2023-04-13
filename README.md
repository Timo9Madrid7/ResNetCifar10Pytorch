# ResNetCifar10Pytorch

1. download the project
```
git clone https://github.com/Timo9Madrid7/ResNetCifar10Pytorch.git ResNetCifar10
```

2. enter into the project folder and create a folder to save the downloaded dataset
```
cd ResNetCifar10
mkdir data logs
```

3. show the helping info
```
python main.py --help
```

4. run the program
```
python main.py --device=gpu --epoch=1 --download=True
```
or if you want to run the program in an implicit way
```
nohup python main.py --device=gpu --epoch=1 --download=True &
```

5. check the results
```
cat logs/ResNetCifar10_cuda.log
```
