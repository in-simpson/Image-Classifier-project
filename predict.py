import argparse
import json
import PIL.Image
import torch
import numpy as np
from math import ceil
from torchvision import models

def arg_parser():
    """
    Parses command-line arguments for the script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="predict.py")
    parser.add_argument('--image', type=str, help='Path to image file for prediction.', required=True)
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file as str.', required=True)
    parser.add_argument('--top_k', type=int, help='Choose top K matches as int.')
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

    args = parser.parse_args()
    
    return args

def load_checkpoint(checkpoint_path):
    """
    Loads a saved model checkpoint from a file.

    Args:
        checkpoint_path (str): Path to the model checkpoint file.

    Returns:
        torch.nn.Module: Loaded model.
    """
    checkpoint = torch.load(checkpoint_path)
    
    model = models.vgg16(pretrained=True)
    model.name = "vgg16"
    
    for param in model.parameters(): 
        param.requires_grad = False
    
    # Load from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image):
    """
    Processes an image for model input.

    Args:
        image (str): Path to the image file.

    Returns:
        numpy.ndarray: Processed image as a NumPy array.
    """
    img = PIL.Image.open(image)

    original_width, original_height = img.size
    
    if original_width < original_height:
        size = [256, 256**600]
    else: 
        size = [256**600, 256]
        
    img.thumbnail(size)
   
    center = original_width / 4, original_height / 4
    left, top, right, bottom = center[0] - (244 / 2), center[1] - (244 / 2), center[0] + (244 / 2), center[1] + (244 / 2)
    img = img.crop((left, top, right, bottom))

    numpy_img = np.array(img) / 255 

    # Normalize each color channel
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    numpy_img = (numpy_img - mean) / std
        
    # Set the color to the first channel
    numpy_img = numpy_img.transpose(2, 0, 1)
    
    return numpy_img

def predict(image_tensor, model, device, cat_to_name, top_k=None):
    """
    Makes predictions on an image using a trained model.

    Args:
        image_tensor (numpy.ndarray): Processed image as a NumPy array.
        model (torch.nn.Module): Trained model for prediction.
        device (torch.device): Device for model computation (CPU or GPU).
        cat_to_name (dict): Mapping of category labels to flower names.
        top_k (int): Number of top predictions to return.

    Returns:
        list, list, list: Top probabilities, top class labels, top flower names.
    """
    model.eval()
    image_tensor = torch.FloatTensor(image_tensor).unsqueeze(0)
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        model = model.to(device)
        output = model(image_tensor)
        probabilities = torch.exp(output)
        top_probabilities, top_indices = probabilities.topk(top_k)
    
    top_probabilities = top_probabilities.cpu().numpy().tolist()[0]
    top_indices = top_indices.cpu().numpy().tolist()[0]
    
    class_to_idx = model.class_to_idx
    idx_to_class = {val: key for key, val in class_to_idx.items()}
    
    top_classes = [idx_to_class[index] for index in top_indices]
    top_flowers = [cat_to_name[class_name] for class_name in top_classes]
    
    return top_probabilities, top_classes, top_flowers

def print_probability(probs, flowers):
    """
    Prints the top probabilities and corresponding flower names.

    Args:
        probs (list): List of top probabilities.
        flowers (list): List of flower names.
    """
    for i, j in enumerate(zip(flowers, probs):
        print("Rank {}:".format(i+1),
              "Flower: {}, likelihood: {:.2%}".format(j[0], j[1])

def main():
    args = arg_parser()
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    model = load_checkpoint(args.checkpoint)
    
    image_tensor = process_image(args.image)
    
    device = torch.device("cuda" if args.gpu == "gpu" and torch.cuda.is_available() else "cpu")
    
    top_probs, top_labels, top_flowers = predict(image_tensor, model, device, cat_to_name, args.top_k)
    
    print_probability(top_probs, top_flowers)

if __name__ == '__main__':
    main()
