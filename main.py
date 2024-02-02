from pathlib import Path
import glob
from mat import load_mat
import cv2
from utils import DataMat, visualize_img_bb
from utils import (
    delete_folder_mkdir,
    overwrite_label,
    copy_file,
    write_yaml,
    make_data_yaml_dict,
)
import os

svhn_format1_path = [
    # Path("/Volumes/kan_ex/public_dataset/number/standford/format1/train"),
    # Path("/Volumes/kan_ex/public_dataset/number/standford/format1/extra"),
    # Path("/Volumes/kan_ex/public_dataset/number/standford/format1/test"),
    (
        Path("/Volumes/kan_ex/public_dataset/number/standford/format1/extra"),
        Path("./finish_format1/extra"),
    ),
    (
        Path("/Volumes/kan_ex/public_dataset/number/standford/format1/train"),
        Path("./finish_format1/train"),
    ),
    (
        Path("/Volumes/kan_ex/public_dataset/number/standford/format1/test"),
        Path("./finish_format1/test"),
    ),
]

label_names = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "colon",
    "dot",
    "float",
    "minus",
    "slash",
]
nc = len(label_names)

# dataMatList = []
for svhn_path, save_folder in svhn_format1_path:
    dataMatList = []
    mat_file = glob.glob(str(svhn_path) + "/*.mat")[0]
    # print(str(svhn_path))
    # print(mat_file)
    names, bboxes = load_mat(mat_file)
    for index in range(len(names)):
        dataMatList.append(
            DataMat(
                name=names[index],
                bb=bboxes[index],
                folder_path=svhn_path,
            )
        )

    delete_folder_mkdir(folder_path=save_folder, remove=True)
    os.mkdir(save_folder / "train")
    save_image_path = save_folder / "train" / "images"
    save_bb_path = save_folder / "train" / "labels"
    os.mkdir(path=str(save_image_path))
    os.mkdir(path=str(save_bb_path))

    for index in range(len(dataMatList)):
        im = cv2.imread(filename=str(dataMatList[index].file_path))
        bb = list(
            box.cvtRoboflowFormat(image_height=im.shape[0], image_width=im.shape[1])
            for box in dataMatList[index].box
        )
        # copy image
        copy_file(
            source_file_path=dataMatList[index].file_path,
            target_file_path=save_image_path,
        )
        # preprocess labels
        target_bb_path = save_bb_path / (dataMatList[index].name.split(".")[0] + ".txt")
        overwrite_label(txt_file_path=str(target_bb_path), bb=bb)

    # write data.yaml file
    yaml_dict = make_data_yaml_dict(nc=nc, names=label_names)
    write_yaml(data=yaml_dict, filepath=str(save_folder / "data.yaml"))