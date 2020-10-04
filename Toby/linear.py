from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import cv2

def extract_vector()

if __name__ == "__main__":
    dataframe = pd.read_csv('./../data/train.csv')
    data

    IDs = dataframe.image_name.to_list()
    labels = dataframe.target.to_list()
    train ,val, y_train,y_val = train_test_split(IDs, labels, test_size = 0.2, random_state = 42, shuffle = True)


    
    model = LogisticRegression(solver='liblinear').fit(train_generator('./../data/train/', train, 64, y_train, len(train)))

    readAndProcess(r'C:\Users\ADMINS\Desktop\yolo.png')
