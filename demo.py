import cv2
import argparse, os, random
import torch
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd
import numpy as np
from model import model_static
from PIL import Image, ImageDraw, ImageFont
from colour import Color

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, help='input video path. live cam is used when not specified')
parser.add_argument('--face', type=str, help='face detection file path. OpenCV DNN is used when not specified')
parser.add_argument('--model_weight', type=str, help='path to model weights file', default='data/model_weights.pkl')
parser.add_argument('--jitter', type=int, help='jitter bbox n times, and average results', default=0)
parser.add_argument('-save_vis', help='saves output as video', action='store_true')
parser.add_argument('-save_text', help='saves output as text', action='store_true')
parser.add_argument('-display_off', help='do not display frames', action='store_true')
args = parser.parse_args()

PROTOTXT_PATH = 'data/deploy.prototxt'
MODEL_PATH = 'data/res10_300x300_ssd_iter_140000.caffemodel'

def bbox_jitter(bbox_left, bbox_top, bbox_right, bbox_bottom):
    cx = (bbox_right + bbox_left) / 2.0
    cy = (bbox_bottom + bbox_top) / 2.0
    scale = random.uniform(0.8, 1.2)
    bbox_right = (bbox_right - cx) * scale + cx
    bbox_left = (bbox_left - cx) * scale + cx
    bbox_top = (bbox_top - cy) * scale + cy
    bbox_bottom = (bbox_bottom - cy) * scale + cy
    return bbox_left, bbox_top, bbox_right, bbox_bottom

def drawrect(drawcontext, xy, outline=None, width=0):
    (x1, y1), (x2, y2) = xy
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill=outline, width=width)

def detect_faces_dnn(frame, net, conf_threshold=0.5):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (l, t, r, b) = box.astype("int")
            l = max(0, l); t = max(0, t); r = min(w, r); b = min(h, b)
            bboxes.append([l, t, r, b])
    return bboxes

def run(video_path, face_path, model_weight, jitter, vis, display_off, save_text):
    red = Color("red")
    colors = list(red.range_to(Color("green"), 10))
    font = ImageFont.truetype("data/arial.ttf", 40)

   

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if video_path is None:
        cap = cv2.VideoCapture(1)
        video_path = 'live.avi'
    else:
        cap = cv2.VideoCapture(video_path)

    frame_cnt = 0
    eye_contact = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    if save_text:
        outtext_name = os.path.basename(video_path).replace('.avi', '_output.txt')
        f = open(outtext_name, "w")
    if vis:
        outvis_name = os.path.basename(video_path).replace('.avi', '_output.avi')
        imwidth = int(cap.get(3)); imheight = int(cap.get(4))
        outvid = cv2.VideoWriter(outvis_name, cv2.VideoWriter_fourcc('M','J','P','G'), cap.get(5), (imwidth, imheight))

    if face_path is None:
        facemode = 'OPENCV'
        face_net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
    else:
        facemode = 'GIVEN'
        column_names = ['frame', 'left', 'top', 'right', 'bottom']
        df = pd.read_csv(face_path, names=column_names, index_col=0)
        df['left'] -= (df['right'] - df['left']) * 0.2
        df['right'] += (df['right'] - df['left']) * 0.2
        df['top'] -= (df['bottom'] - df['top']) * 0.1
        df['bottom'] += (df['bottom'] - df['top']) * 0.1
        df = df.astype(int)

    if not cap.isOpened():
        print("Error opening video stream or file")
        exit()

    test_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    model = model_static(model_weight).to(device)
    snapshot = torch.load(model_weight, map_location=device)
    model.load_state_dict(snapshot)
    model.eval()


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width, channels = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_cnt += 1
        bbox = []

        if facemode == 'OPENCV':
            bbox = detect_faces_dnn(frame, face_net)
        elif facemode == 'GIVEN' and frame_cnt in df.index:
            bbox.append([df.loc[frame_cnt, 'left'], df.loc[frame_cnt, 'top'],
                         df.loc[frame_cnt, 'right'], df.loc[frame_cnt, 'bottom']])

        frame_pil = Image.fromarray(rgb_frame)
        for b in bbox:
            face = frame_pil.crop((b))
            img = test_transforms(face).unsqueeze(0).to(device)
            if jitter > 0:
                for i in range(jitter):
                    bj_left, bj_top, bj_right, bj_bottom = bbox_jitter(b[0], b[1], b[2], b[3])
                    bj = [bj_left, bj_top, bj_right, bj_bottom]
                    facej = frame_pil.crop((bj))
                    img_jittered = test_transforms(facej).unsqueeze(0).to(device)
                    img = torch.cat([img, img_jittered])

            output = model(img)
            if jitter > 0:
                output = torch.mean(output, 0)
            score = torch.sigmoid(output).item()
            if score >= 0.95:
                eye_contact += 1
            coloridx = min(int(round(score * 10)), 9)
            draw = ImageDraw.Draw(frame_pil)
            drawrect(draw, [(b[0], b[1]), (b[2], b[3])], outline=colors[coloridx].hex, width=5)
            draw.text((b[0], b[3]), str(round(score, 2)), fill=(255, 255, 255, 128), font=font)
            if save_text:
                f.write("%d,%f\n" % (frame_cnt, score))

        frame = np.asarray(frame_pil)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        eye_contact_duration = eye_contact/fps
        text = f"eye contact duration: {eye_contact_duration:.2f} sec"
        cv2.putText(frame, text, (30,50),cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 2)

        if not display_off:
            cv2.imshow('', frame)
        if vis:
            outvid.write(frame)
        if not display_off:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    if vis:
        outvid.release()
    if save_text:
        f.close()
    cap.release()
    eye_contact_duration = eye_contact/fps

    print("done")

if __name__ == "__main__":
    run(args.video, args.face, args.model_weight, args.jitter, args.save_vis, args.display_off, args.save_text)
