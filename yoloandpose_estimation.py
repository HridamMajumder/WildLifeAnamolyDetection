# If you get an error from PIL restart environment and rerun this cell to update packages version
import torch
MODEL_SIZE = 'h'
from easy_ViTPose import VitInference
model_path  = "vitpose-h-apt36k.pth"
yolo_path = "yolov8s.pt"
Device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = VitInference(model_path, yolo_path, MODEL_SIZE,yolo_size=320, is_video=False,device=Device)


import cv2
from inference import VideoReader
import time
import tqdm
from easy_ViTPose.vit_utils.visualization import joints_dict
import json
from easy_ViTPose.vit_utils.inference import NumpyEncoder, VideoReader
import numpy as np
from ultralytics import YOLO

def label(img):
     yolo = YOLO("yolov8s.pt")
     results = yolo(img)
     label_names = [yolo.names[int(detection.cls)] for detection in results[0].boxes]
     return label_names

def process_video(input_path , output_path_img,output_path_json):
    reader = VideoReader(input_path)
    cap = cv2.VideoCapture(input_path)  # type: ignore
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    wait = 15
    cap = cv2.VideoCapture(input_path)  # type: ignore
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, frame = cap.read()
    cap.release()
    output_size = frame.shape[:2][::-1]
    out_writer = cv2.VideoWriter(output_path_img,cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),fps, output_size)

    keypoints = []
    fps = []
    Label = []
    tot_time = 0.
    cap = cv2.VideoCapture(input_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        label_names = label(frame)
        Label.append(label_names)


    for (ith, img) in tqdm.tqdm(enumerate(reader), total=total_frames):
        t0 = time.time()

        # Run inference
        frame_keypoints = model.inference(img)
        keypoints.append(frame_keypoints)

        delta = time.time() - t0
        tot_time += delta
        fps.append(delta)
        img = model.draw(show_yolo=True)[..., ::-1]
        out_writer.write(img)

    tot_poses = sum(len(k) for k in keypoints)
    print(f'>>> Mean inference FPS: {1 / np.mean(fps):.2f}')
    print(f'>>> Total poses predicted: {tot_poses} mean per frame: '
              f'{(tot_poses / (ith + 1)):.2f}')
    print(f'>>> Mean FPS per pose: {(tot_poses / tot_time):.2f}')

    with open(output_path_json, 'w') as f:
            out = {'keypoints': keypoints,'Label':Label,
                   'skeleton': joints_dict()[model.dataset]['keypoints']}
            json.dump(out, f, cls=NumpyEncoder)
    out_writer.release()
    cv2.destroyAllWindows()

    return output_path_img , output_path_json





vdo_path = r"C:\Users\hrida\Downloads\3775802-hd_1920_1080_25fps.mp4"

process_video(vdo_path, 'output.mp4', 'output.json')

