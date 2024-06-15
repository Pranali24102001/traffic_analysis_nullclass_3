#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd

def preprocess_traffic_vehicle_data(traffic_vehicle_data_path):
    def process_data_split(split):
        data = []
        images_path = os.path.join(traffic_vehicle_data_path, 'images', split)
        labels_path = os.path.join(traffic_vehicle_data_path, 'labels', split)
        
        for label_file in os.listdir(labels_path):
            if label_file.endswith('.txt'):
                label_path = os.path.join(labels_path, label_file)
                img_file_name = label_file.replace('.txt', '.jpg')
                img_path = os.path.join(images_path, img_file_name)
                if not os.path.exists(img_path):
                    continue

                with open(label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        class_id = parts[0]
                        bbox = list(map(float, parts[1:]))
                        data.append({
                            'image_path': img_path,
                            'bbox': bbox,
                            'class_id': class_id
                        })
        
        df = pd.DataFrame(data)
        return df
    
    traffic_train_data = process_data_split('train')
    traffic_val_data = process_data_split('val')
    traffic_test_data = None  # Placeholder for test data
    
    return traffic_train_data, traffic_val_data, traffic_test_data

