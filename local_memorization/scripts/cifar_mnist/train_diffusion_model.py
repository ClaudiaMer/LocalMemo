import os
import torch
from local_memorization.optim.pipeline import run_full_pipeline

DATA_PATH = "./combined_splits_mnist_cifar/"
NUM_GENERATED = 100

def load_data(args): 
    # load training data
    if not args.cluster:
        data_tensor = torch.load(DATA_PATH+"train.pt")[0]
    else:
        work_dir = os.environ.get("WORK")
        data_tensor = torch.load(work_dir +"/"
                              +DATA_PATH+"train.pt")[0]
    return data_tensor


def load_test_data(args, return_class=False):
    if not args.cluster:
        test_data_tensor = torch.load(DATA_PATH+"test.pt")[0]
        class_tensor = torch.load(DATA_PATH+"test.pt")[1][0]
    else:
        work_dir = os.environ.get("WORK")
        test_data_tensor = torch.load(work_dir +"/"
                              +DATA_PATH+"test.pt")[0]
        class_tensor = torch.load(work_dir +"/"
                              +DATA_PATH+"test.pt")[1]

    if return_class:
        print("class tensor",class_tensor)
        return test_data_tensor, class_tensor
    else:
        return test_data_tensor

def model_save_folder(args): 

    if args.cluster: 
        work_dir = os.environ.get("WORK")+"/cifar10_color_mnist_trained/"
    else: 
        work_dir ="./trained/"
    folder = work_dir+f"checkpoints_seed{args.seed}_N{args.N}"
    if args.adamW: 
        folder += "_adamW"

    folder += "/"
    os.makedirs(folder, exist_ok=True)

    return folder

def load_class(args):  
    if args.cluster: 
        work_dir = os.environ.get("WORK")
        class_tensor = torch.load(work_dir + "/" + DATA_PATH + "train.pt")[1]
    else:
        class_tensor = torch.load(DATA_PATH + "train.pt")[1]

    dataset_labels = class_tensor[0]

    return dataset_labels

if __name__=="__main__": 
    #print(load_class(None))
    run_full_pipeline(load_data, model_save_folder, load_class=load_class, load_test_data=load_test_data, num_generated=NUM_GENERATED)