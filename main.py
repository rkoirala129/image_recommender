import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

load_tensors = pickle.load(open('embeddings.pkl','rb'))
# load_array= [i.numpy() for i in load_tensors]
# print(load_array)

feature_list = load_tensors

# feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

resnet50 = models.resnet50(weights='ResNet50_Weights.DEFAULT')

# Remove the last layer (the classification layer) from the model
resnet50 = torch.nn.Sequential(*list(resnet50.children())[:-1])

# Set the model to evaluation mode
resnet50.eval()

st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(img_path,model):
    image = Image.open(img_path)
    image = transform(image)
    image = torch.unsqueeze(image, 0)
    # Extract features from the image
    features = resnet50(image).flatten()
    return features

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# steps
# file upload -> save
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        # feature extract
        features = extract_features(os.path.join("uploads",uploaded_file.name),resnet50)
        #st.text(features)
        # recommendention
        indices = recommend(features.detach().numpy(),feature_list)
        # show
        col1,col2,col3,col4,col5 = st.columns(5)

        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
    else:
        st.header("Some error occured in file upload")
