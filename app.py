import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle


# Load pre-trained ResNet-50 model
resnet50 = models.resnet50(weights='ResNet50_Weights.DEFAULT')

# Remove the last layer (the classification layer) from the model
resnet50 = torch.nn.Sequential(*list(resnet50.children())[:-1])

# Set the model to evaluation mode
resnet50.eval()
resnet50.cuda()


# Load an image and preprocess it
# image = Image.open("./images/1530.jpg")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def extract_features(img_path,model):
    image = Image.open(img_path)
    img_arr = np.array(image)
    # print(img_arr.shape)

    if len(img_arr.shape) == 2:
    # Convert grayscale image to colored image
        img_arr = np.stack((img_arr,) * 3, axis=-1)
        # Convert the numpy array back to a PIL image
        image = Image.fromarray(img_arr)
        
    image = transform(image)
    image = torch.unsqueeze(image, 0)
    image = image.cuda()
    # Extract features from the image
    features = resnet50(image).flatten()
    features = features.squeeze().detach().cpu().numpy()
    return features

# img_path = "./images/1530.jpg"

# print(extract_features(img_path, resnet50))

filenames = []

for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))

feature_list = []



for file in tqdm(filenames):
    fts = extract_features(file,resnet50)
    feature_list.append(fts)


pickle.dump(feature_list,open('embeddings.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))