#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
from PIL import Image

def preprocess_person_data(person_data_path):
    data = []

    for label in ['0', '1']:
        label_path = os.path.join(person_data_path, label)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                if os.path.isfile(img_path) and img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    data.append({
                        'image_path': img_path,
                        'label': label
                    })

    df = pd.DataFrame(data)
    return df

# Example usage
person_data_path = 'C:/Users/Yadav/Desktop/traffic_analysis_project/data/person_detection'
person_data = preprocess_person_data(person_data_path)
print(person_data.head())

