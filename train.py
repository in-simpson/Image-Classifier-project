import argparse
import torch
from collections import OrderedDict
from os.path import isdir
import os
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

# Function to parse command-line arguments
def argument_parser():
    """
    Parse command-line arguments and return the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train.py")
    parser.add_argument('--architecture', dest="architecture", action="store", default="vgg16", type=str)
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth", help="Directory to save checkpoints")
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
    parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
    args = parser.parse_args()
    return args

# Function to transform and preprocess training data
def transform_train_data(train_dir):
    """
    Transform and preprocess training data.
    Args:
        train_dir (str): Directory containing training data.
    Returns:
        train_data: Transformed training data.
    """
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    return train_data

# Function to transform and preprocess test data
def transform_test_data(test_dir):
    """
    Transform and preprocess test data.
    Args:
        test_dir (str): Directory containing test data.
    Returns:
        test_data: Transformed test data.
    """
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return test_data

# Function to create a data loader
def create_data_loader(data, train=True):
    """
    Create a data loader for training or testing data.
    Args:
        data: Transformed data.
        train (bool): True for training data loader, False for testing data loader.
    Returns:
        loader: Data loader.
    """
    if train:
        loader = torch.utils.data.DataLoader(data, batch_size=50, shuffle=True)
    else:
        loader = torch.utils.data.DataLoader(data, batch_size=50)
    return loader

# Function to check and set the GPU device
def check_gpu(gpu_arg):
    """
    Check if a GPU is available and return the appropriate device.
    Args:
        gpu_arg (str): "gpu" or "cpu".
    Returns:
        device: GPU device if available, else CPU.
    """
    if not gpu_arg:
        return torch.device("cpu")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if device == "cpu":
        print("CUDA was not found on device, using CPU instead.")
    return device

# Function to load a pre-trained model
def load_pretrained_model(architecture="vgg16"):
    """
    Load a pre-trained model.
    Args:
        architecture (str): Model architecture name.
    Returns:
        model: Pre-trained model.
    """
    model = models.vgg16(pretrained=True)
    model.name = "vgg16"
    
    for param in model.parameters():
        param.requires_grad = False
    return model

# Function to create a custom classifier
def create_classifier(model, hidden_units):
    """
    Create a custom classifier for the model.
    Args:
        model: Pre-trained model.
        hidden_units (int): Number of hidden units in the classifier.
    Returns:
        classifier: Custom classifier.
    """
    classifier = nn.Sequential(OrderedDict([
                ('inputs', nn.Linear(25088, 120)),
                ('relu1', nn.ReLU()),
                ('dropout', nn.Dropout(0.5)),
                ('hidden_layer1', nn.Linear(120, 90)),
                ('relu2', nn.ReLU()),
                ('hidden_layer2', nn.Linear(90, 70)),
                ('relu3', nn.ReLU()),
                ('hidden_layer3', nn.Linear(70, 102)),
                ('output', nn.LogSoftmax(dim=1))]))
    
    model.classifier = classifier
    return classifier

# Function to validate the model
def validate(model, testloader, criterion, device):
    """
    Validate the model on test data and return loss and accuracy.
    Args:
        model: Trained model.
        testloader: Test data loader.
        criterion: Loss criterion.
        device: CPU or GPU device.
    Returns:
        test_loss: Test loss.
        accuracy: Test accuracy.
    """
    test_loss = 0
    accuracy = 0
    
    for ii, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy

# Function to train the model
def train_network(model, trainloader, validloader, device, criterion, optimizer, epochs, print_every, steps):
    """
    Train the model on training data.
    Args:
        model: Model with a custom classifier.
        trainloader: Training data loader.
        validloader: Validation data loader.
        device: CPU or GPU device.
        criterion: Loss criterion.
        optimizer: Model optimizer.
        epochs (int): Number of training epochs.
        print_every (int): Frequency to print training progress.
        steps (int): Number of training steps.
    Returns:
        model: Trained model.
    """
    if type(epochs) == type(None):
        epochs = 10
        print("Number of epochs specified as 10.")
    
    print(f"Training process initializing for {epochs} epochs ...\n")
    
    for e in range(epochs):
        running_loss = 0
        model.train()
        
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss, accuracy = validate(model, validloader, criterion, device)
                
                print("Epoch: {}/{} | ".format(e+1, epochs),
                      "Training Loss: {:.4f} | ".format(running_loss/print_every),
                      "Validation Loss: {:.4f} | ".format(valid_loss/len(validloader)),
                      "Validation Accuracy: {:.4f}".format(accuracy/len(validloader)))
                
                running_loss = 0
                model.train()
    
    return model

# Function to test the trained model
def test_model(model, testloader, device):
    """
    Test the trained model on test data and print the accuracy.
    Args:
        model: Trained model.
        testloader: Test data loader.
        device: CPU or GPU device.
    """
    correct, total = 0, 0
    with torch.no_grad():
        model.eval()
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy on test images is: %d%%' % (100 * correct / total))

# Function to save the trained model checkpoint
def save_checkpoint(model, save_dir, train_data):
    """
    Save a checkpoint of the trained model.
    Args:
        model: Trained model.
        save_dir (str): Directory to save the checkpoint.
        train_data: Training data.
    """
    if type(save_dir) == type(None):
        print("Model checkpoint directory not specified, model will not be saved.")
    else:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        model.class_to_idx = train_data.class_to_idx
        checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
        checkpoint = {
            'architecture': model.name,
            'classifier': model.classifier,
            'class_to_idx': model.class_to_idx,
            'state_dict': model.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Model checkpoint saved in {checkpoint_path}")

# Main function to execute the entire training process
def main():
    args = argument_parser()
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_data = transform_train_data(train_dir)
    valid_data = transform_test_data(valid_dir)
    test_data = transform_test_data(test_dir)
    
    trainloader = create_data_loader(train_data)
    validloader = create_data_loader(valid_data, train=False)
    testloader = create_data_loader(test_data, train=False)
    
    model = load_pretrained_model(architecture=args.architecture)
    model.classifier = create_classifier(model, hidden_units=args.hidden_units)
    
    device = check_gpu(gpu_arg=args.gpu)
    model.to(device)
    
    if type(args.learning_rate) == type(None):
        learning_rate = 0.001
        print("Learning rate specified as 0.001")
    else: 
        learning_rate = args.learning_rate
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    print_every = 30
    steps = 0
    
    trained_model = train_network(model, trainloader, validloader, device, criterion, optimizer, args.epochs, print_every, steps)
    
    print("\nTraining process is completed!!")
    
    test_model(trained_model, testloader, device)
    
    save_checkpoint(trained_model, args.save_dir, train_data)

if __name__ == '__main__':
    main()
