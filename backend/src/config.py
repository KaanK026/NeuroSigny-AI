#Configuration file for ASL Prediction AI

import os

#AWS configurations are in .env

#Mapping indices with labels
LETTERS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "del", "space", "nothing"
]


mobile_net_path='best_mobilenet_model.pth'
resnet_path='best_resnet_model.pth'
cnn_path='best_customcnn_model.pth'

#OS independent paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

resnet_full_path=os.path.join(BASE_DIR, "best_resnet_model.pth")
mobilenet_full_path=os.path.join(BASE_DIR, "best_mobilenet_model.pth")
cnn_full_path=os.path.join(BASE_DIR, "best_customcnn_model.pth")

TRAIN_FILE = os.getenv("TRAIN_FILE", os.path.join(BASE_DIR, "datas", "asl_alphabet_train", "asl_alphabet_train"))