#!/usr/bin/env python
# coding: utf-8

# In[ ]:




# In[ ]:


import os
import pandas as pd
import xmltodict

def preprocess_car_plate_data(images_path, annotations_path):
    data = []

    for annotation_file in os.listdir(annotations_path):
        if annotation_file.endswith('.xml'):
            annotation_path = os.path.join(annotations_path, annotation_file)
            with open(annotation_path) as f:
                annotation = xmltodict.parse(f.read())
            
            # Debugging: Print the structure of the annotation
            print(f"Processing {annotation_path}")
            print(annotation)

            # Extract relevant information from the XML structure
            img_file_name = annotation['annotation']['filename']
            img_path = os.path.join(images_path, img_file_name)
            if not os.path.exists(img_path):
                continue
            
            objects = annotation['annotation'].get('object', [])
            if not isinstance(objects, list):
                objects = [objects]
            
            for obj in objects:
                bbox = obj['bndbox']
                label = obj['name']
                data.append({
                    'image_path': img_path, 
                    'bbox': [int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])], 
                    'label': label
                })

    df = pd.DataFrame(data)
    return df

# Example usage
car_plate_images_path = 'C:/Users/Yadav/Desktop/traffic_analysis_project/data/car_plate_images'
car_plate_annotations_path = 'C:/Users/Yadav/Desktop/traffic_analysis_project/data/car_plate_annotations'
car_plate_data = preprocess_car_plate_data(car_plate_images_path, car_plate_annotations_path)
print(car_plate_data.head())

