# SpikingFinalProject

## Training

Non neuromorphic datasets
```
python train.py --activation ReLU of thReLU --maxpooling 1/0 --softmax 1/0 --dataset CIFAR or MNIST
```

Neuromorphic datasets
```
python trainDVS.py --activation ReLU of thReLU --maxpooling 1/0 --softmax 1/0 --dataset NCIFAR or NMNIST
```


## Test Conversion

Non neuromorphic datasets
```
python testSpiking.py --activation ReLU of thReLU --maxpooling 1/0 --softmax 1/0 --dataset CIFAR or MNIST
```

Neuromorphic datasets
```
python testSpinkingDVS.py --activation ReLU of thReLU --maxpooling 1/0 --softmax 1/0 --dataset CIFAR or MNIST
```


#### REQUIRED ARGUMENTS
```
 - dataset: MNIST or CIFAR 
 - model: SimpleCNN or SimpleBN
```


####Optional ARGUMENTS
```
--dataset str default: CIFAR Dataset use to train or test
--lr  float default: 0.001 Learning rate
--epochs int default: 5 Number of epochs for training
--thresh float default: 2 Activation threshhold for threshold relu
--sim_len int default: 32 Simulation length for spiking neurons
--step int default: 4 Simulation steps for spiking neurons
--bsz int default:64 Batch size
--bn bool default: True Whether to use batch normalization or not
--pool str default:avg Whether to use max-pooling or average pooling 
--softmax bool defualt: False Whether to use spiking softmax in testing
--threlu bool default: False Whehter to use threshold relu in training
--train str default: True Train Vgg16 or test converted Vgg16

```
## References

* Conversion of Maxpool and Softmax: adapted from the tensorflow code in [Rueckauer, B. et al. (2016)](https://github.com/NeuromorphicProcessorProject/snn_toolbox)
* Conversion of ReLU: adapted from the pytorch code in [Deng and Gu (2021)](https://github.com/Jackn0/snn_optimal_conversion_pipeline)
* Torch dataloader to tensorflow dataloader [SdahlSean](https://github.com/SdahlSean/PytorchDataloaderForTensorflow)


