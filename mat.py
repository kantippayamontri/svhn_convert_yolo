import mat73
from pathlib import Path

def load_mat(file_path):
    data_dict = mat73.loadmat(file=file_path)
    names = data_dict["digitStruct"]["name"]
    bboxes= data_dict["digitStruct"]["bbox"]
    return names, bboxes