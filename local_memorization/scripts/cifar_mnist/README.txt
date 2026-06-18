The scripts in this folder work analogously to the cifar-10-color scripts. We here only highlight the differences. 

- preprocess.py will download and preprocess both CIFAR-10 dataset + MNIST dataset. It will create two folders, preferentially in the WORK folder, or, if none exists, locally: 
    combined_data_mnist_cifar #holds the raw data 
    combined_splits_mnist_cifar #holds {test,train,val}.pt (data splits) and {test,train,val}_stats.pt (mean and covariance of each split). 

In the evaluation scripts, data originating from CIFAR and from MNIST will have one class, each, no distinction of sublclasses of these classes is made. 
