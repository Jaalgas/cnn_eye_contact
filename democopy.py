import cv2, os, random, argparse
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from model import model_static
from PIL import Image, ImageDraw, ImageFont


def bbox_jitter(l, t, r, b):
    cx, cy = (r + l) / 2, (b + t) / 2
    scale = random.uniform(0.8, 1.2)
    return [(x - cx) * scale + cx for x in [l, t, r, b]]

def drawrect(draw, xy, color, width=5):
    (x1, y1), (x2, y2) = xy
    draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], fill=color, width=width)

def detect_faces_dnn(frame, net, conf_threshold=0.5):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            l, t, r, b = box.astype(int)
            bboxes.append([max(0, l), max(0, t), min(w, r), min(h, b)])
    return bboxes

def run(video_path, face_path, model_weight, jitter, save_vis, display_off, save_text):
    cap = cv2.VideoCapture(0 if video_path is None else video_path)
    if not cap.isOpened(): return print("Error opening video")

    font = ImageFont.truetype("data/arial.ttf", 40)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_static(model_weight).to(device)
    model.load_state_dict(torch.load(model_weight, map_location=device))
    model.eval()

    test_tf = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if face_path:
        df = pd.read_csv(face_path, names=['frame', 'left', 'top', 'right', 'bottom'], index_col=0)
        df['left'] -= (df['right'] - df['left']) * 0.2
        df['right'] += (df['right'] - df['left']) * 0.2
        df['top'] -= (df['bottom'] - df['top']) * 0.1
        df['bottom'] += (df['bottom'] - df['top']) * 0.1
        df = df.astype(int)
    else:
        face_net = cv2.dnn.readNetFromCaffe('data/deploy.prototxt', 'data/res10_300x300_ssd_iter_140000.caffemodel')

    outvid = None
    if save_vis:
        h, w = int(cap.get(4)), int(cap.get(3))
        outvid = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc(*'MJPG'), cap.get(5), (w, h))
    if save_text:
        fout = open("output.txt", "w")

    eye_contact, start_time, duration = False, None, 0
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1
        t_now = cv2.getTickCount() / cv2.getTickFrequency()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(rgb)

        bboxes = (detect_faces_dnn(frame, face_net) if not face_path 
                  else [df.loc[frame_id][['left', 'top', 'right', 'bottom']].tolist()] if frame_id in df.index else [])

        for box in bboxes:
            img = test_tf(pil_frame.crop(box)).unsqueeze(0).to(device)
            if jitter:
                for _ in range(jitter):
                    box_j = bbox_jitter(*box)
                    img_j = test_tf(pil_frame.crop(box_j)).unsqueeze(0).to(device)
                    img = torch.cat([img, img_j])
            score = torch.sigmoid(model(img).mean(0)).item()

            # Use green for eye contact, red otherwise
            color = "#00FF00" if score >= 0.95 else "#FF0000"

            draw = ImageDraw.Draw(pil_frame)
            drawrect(draw, [(box[0], box[1]), (box[2], box[3])], color)
            draw.text((box[0], box[3]), f"{score:.2f}", fill="white", font=font)
            if save_text: fout.write(f"{frame_id},{score:.3f}\n")

            if score >= 0.95:
                if not eye_contact:
                    eye_contact, start_time = True, t_now
            elif eye_contact:
                duration += t_now - start_time
                eye_contact, start_time = False, None

        if eye_contact:
            duration += (cv2.getTickCount() / cv2.getTickFrequency()) - t_now
            start_time = cv2.getTickCount() / cv2.getTickFrequency()

        out_frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
        cv2.putText(out_frame, f"eye contact duration: {duration:.2f} sec", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if not display_off: cv2.imshow('', out_frame)
        if save_vis: outvid.write(out_frame)
        if not display_off and cv2.waitKey(1) & 0xFF == ord('q'): break

    if eye_contact and start_time:
        duration += (cv2.getTickCount() / cv2.getTickFrequency()) - start_time

    print(f"Eye contact duration: {duration:.2f} sec")
    if save_vis: outvid.release()
    if save_text: fout.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--video', type=str)
    p.add_argument('--face', type=str)
    p.add_argument('--model_weight', type=str, default='data/model_weights.pkl')
    p.add_argument('--jitter', type=int, default=0)
    p.add_argument('-save_vis', action='store_true')
    p.add_argument('-save_text', action='store_true')
    p.add_argument('-display_off', action='store_true')
    args = p.parse_args()
    run(args.video, args.face, args.model_weight, args.jitter, args.save_vis, args.display_off, args.save_text)
