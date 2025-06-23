import cv2, random, torch
import numpy as np
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

class EyeContactDetector:
    def __init__(self, model_weight='data/model_weights.pkl'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_static(model_weight).to(self.device)
        self.model.load_state_dict(torch.load(model_weight, map_location=self.device))
        self.model.eval()
        self.face_net = cv2.dnn.readNetFromCaffe('data/deploy.prototxt', 'data/res10_300x300_ssd_iter_140000.caffemodel')
        self.font = ImageFont.truetype("data/arial.ttf", 40)
        self.tf = transforms.Compose([
            transforms.Resize(224), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.eye_contact = False
        self.start_time = None
        self.duration = 0

    def detect_faces(self, frame, conf_threshold=0.5):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > conf_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                l, t, r, b = box.astype(int)
                bboxes.append([max(0, l), max(0, t), min(w, r), min(h, b)])
        return bboxes

    def detect(self, frame):
        t_now = cv2.getTickCount() / cv2.getTickFrequency()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(rgb)

        bboxes = self.detect_faces(frame)
        for box in bboxes:
            img = self.tf(pil_frame.crop(box)).unsqueeze(0).to(self.device)
            score = torch.sigmoid(self.model(img).mean(0)).item()
            color = "#00FF00" if score >= 0.95 else "#FF0000"

            draw = ImageDraw.Draw(pil_frame)
            drawrect(draw, [(box[0], box[1]), (box[2], box[3])], color)
            draw.text((box[0], box[3]), f"{score:.2f}", fill="white", font=self.font)

            if score >= 0.95:
                if not self.eye_contact:
                    self.eye_contact, self.start_time = True, t_now
            elif self.eye_contact:
                self.duration += t_now - self.start_time
                self.eye_contact, self.start_time = False, None

        if self.eye_contact:
            self.duration += (cv2.getTickCount() / cv2.getTickFrequency()) - t_now
            self.start_time = cv2.getTickCount() / cv2.getTickFrequency()

        final_frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
        cv2.putText(final_frame, f"eye contact duration: {self.duration:.2f} sec", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return final_frame
