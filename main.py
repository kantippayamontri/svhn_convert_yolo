import glob
import os
import sys
import time
from pathlib import Path

import cv2
import magic
from icecream import ic
from tqdm import tqdm

from convert import convert_number_to_frame
from mat import load_mat
from utils import (
    DataMat,
    convert_xyxy2yolo,
    copy_file,
    delete_folder_mkdir,
    get_filename_bb_folder,
    get_img_dim,
    load_bb,
    load_img_cv2,
    make_data_yaml_dict,
    overwrite_label,
    visualize_img_bb,
    write_yaml,
)

svhn_format1_path = [
    # (
    #     Path("/Volumes/kan_ex/public_dataset/number/standford/format1/extra"),
    #     Path("./finish_format1/extra"),
    # ),
    # (
    #     Path("/Volumes/kan_ex/public_dataset/number/standford/format1/train"),
    #     Path("./finish_format1/train"),
    # ),
    (
        Path("./datasets/svhn_format1/train"),
        Path("./finish_format1_number/train"),
    ),
    (
        Path("./datasets/svhn_format1/extra"),
        Path("./finish_format1_number/extra"),
    ),
    (
        Path("./datasets/svhn_format1/test"),
        Path("./finish_format1_number/test"),
    ),
]

number_label_names = [
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

digital_label_names = [
    "gauge",
    "display",
    "frame",
]

nc = len(number_label_names)

# for svhn_path, save_folder in svhn_format1_path:
#     print(f"from path: {str(svhn_path)}")
#     print(f"target path: {str(save_folder)}")

#     print(f"[-] Readfile")
#     start_time = time.time()
#     mat_file = glob.glob(str(svhn_path) + "/*.mat")[0]
#     dataMatGen = (
#         DataMat(name=_name, bb=_bboxes, folder_path=svhn_path)
#         for (_name, _bboxes) in load_mat(mat_file)
#     )
#     end_time = time.time()
#     print(f"\t[-] use time {(end_time-start_time)} seconds.")

#     delete_folder_mkdir(folder_path=save_folder, remove=True)
#     os.mkdir(save_folder / "train")
#     save_image_path = save_folder / "train" / "images"
#     save_bb_path = save_folder / "train" / "labels"
#     os.mkdir(path=str(save_image_path))
#     os.mkdir(path=str(save_bb_path))

#     print(f"[-] Preprocess")
#     start_time = time.time()
#     len_files = len(os.listdir(str(svhn_path))) - 1 # minus .mat file

#     for _dataMat in tqdm(dataMatGen, total=len_files):

#         _image_width, _image_height = get_img_dim(filepath=str(_dataMat.file_path))

#         bb = list(
#             box.cvtRoboflowFormat(image_height=_image_height, image_width=_image_width)
#             for box in _dataMat.box
#         )
#         # copy image
#         copy_file(
#             source_file_path=_dataMat.file_path,
#             target_file_path=save_image_path,
#         )
#         # preprocess labels
#         target_bb_path = save_bb_path / (_dataMat.name.split(".")[0] + ".txt")
#         overwrite_label(txt_file_path=str(target_bb_path), bb=bb)
#     end_time = time.time()
#     print(f"\t[-] use time {(end_time-start_time)} seconds.")


#     # write data.yaml file
#     yaml_dict = make_data_yaml_dict(nc=nc, names=number_label_names)
#     write_yaml(data=yaml_dict, filepath=str(save_folder / "data.yaml"))
#     print()
#     print()

# for convert from number label to gauge label

is_convert_2_gauge = True

if not is_convert_2_gauge:
    exit()

for _, folder in svhn_format1_path:
    print(f"convert to gauge format at {str(folder)}")
    start_time = time.time()
    new_folder_name = "gaugeFormat_" + folder.name
    new_folder_parent = folder.with_name(new_folder_name)
    new_folder_image = new_folder_parent / "train" / "images"
    new_folder_bb = new_folder_parent / "train" / "labels"

    delete_folder_mkdir(
        folder_path=new_folder_parent, remove=True
    )  # remove and create folder
    os.mkdir(new_folder_parent / "train")
    os.mkdir(new_folder_image)
    os.mkdir(new_folder_bb)

    len_files = len(os.listdir(str(folder / "train" / "images")))
    # img_label_gen = (
    #     ic(match_files)
    #     for match_files in get_filename_bb_folder(
    #         img_path=folder / "train" / "images",
    #         bb_path=folder / "train" / "labels",
    #         source_folder=folder,
    #     )
    # )
    # len_files = 5

    # for (img_path, bb_path) in tqdm(img_label_gen, total=len_files):
    #     ic(img_path, bb_path)

    dataGen = get_filename_bb_folder(
        img_path=folder / "train" / "images",
        bb_path=folder / "train" / "labels",
        source_folder=folder / "train",
    )

    count = 0

    for img_path, bb_path in tqdm(dataGen, total=len_files):
        # ic((img_path, bb_path))

        # copy image
        copy_file(source_file_path=img_path, target_file_path=new_folder_image)

        # convert label
        img_dim = get_img_dim(filepath=img_path)  # img_w, img_h
        bb = load_bb(filepath=bb_path)
        new_bb_xyxy = convert_number_to_frame(img_dim=img_dim, bb=bb)

        # visualize_img_bb(
        #     img=load_img_cv2(filepath=img_path),
        #     bb=list(
        #         [
        #             {
        #                 "class": _bb[0],
        #                 "bb": _bb[1:],
        #             }
        #             for _bb in new_bb_xyxy
        #         ]
        #     ),
        #     with_class=True,
        #     labels=digital_label_names,
        # )


        new_bb_yolo = convert_xyxy2yolo(bb=new_bb_xyxy)
        overwrite_label(txt_file_path=new_folder_bb / bb_path.name, bb=new_bb_yolo)


        # count += 1
        # if count >= 5:
        #     break

    # write data.yaml file
    yaml_dict = make_data_yaml_dict(nc=len(digital_label_names), names=digital_label_names)
    write_yaml(data=yaml_dict, filepath=str(new_folder_parent / "data.yaml"))

    end_time = time.time()
    print(f"\t[-] use time {(end_time-start_time):.4f} seconds.")
