from pathlib import Path
import mat73

def load_mat(file_path):
    data_dict = mat73.loadmat(file=file_path)
    return (
        (data_dict["digitStruct"]["name"][i], data_dict["digitStruct"]["bbox"][i])
        for i in range(len(data_dict["digitStruct"]["name"]))
    )