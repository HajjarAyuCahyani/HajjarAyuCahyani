import os
import sys
import pathlib
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import torch
import cv2
import numpy as np

# === Path Setup ===
FILE = pathlib.Path(__file__).resolve()
ROOT = FILE.parents[0] / 'yolov5'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# === Windows Path Patch ===
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

# === YOLOv5 Imports ===
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# === Flask App Setup ===
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# === Device and Model Loading ===
device = select_device('')
model = DetectMultiBackend('best.pt', device=device)
stride, names, pt = model.stride, model.names, model.pt
model.warmup(imgsz=(1, 3, 640, 640))

# === Letterbox Resize ===
def letterbox(im, new_shape=640, stride=32, auto=True, scaleFill=False, scaleup=True):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dw /= 2
    dh /= 2
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return im, ratio, (dw, dh)

# === Drawing Bounding Boxes ===
def draw_boxes(image, preds, names):
    for *box, _, cls in preds:
        x1, y1, x2, y2 = map(int, box)
        if names[int(cls)].lower() == 'abnormal':
            color = (0, 0, 255)  # Red
        else:
            color = (0, 255, 0)  # Green
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    return image

# === Routes ===
@app.route('/')
def welcome_page():
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    files = request.files.getlist('files')
    if not files or len(files) == 0:
        return redirect(request.url)

    results = []   # simpan info tiap gambar (buat carousel)

    # variabel global untuk akumulasi semua gambar
    total_normal = 0
    total_abnormal = 0
    total_count = 0

    for file in files:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess
        img0 = cv2.imread(filepath)
        img = letterbox(img0, new_shape=640, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.6)[0]

        if pred is None or len(pred) == 0:
            results.append({
                'filename': filename,
                'normal_count': 0,
                'abnormal_count': 0,
                'total_count': 0,
                'image_url': url_for('static', filename=f'uploads/{filename}')
            })
            continue

        # Rescale boxes
        pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], img0.shape).round()

        # Count per gambar
        abnormal_count = sum(1 for *box, conf, cls in pred if names[int(cls)].lower() == 'abnormal')
        normal_count   = sum(1 for *box, conf, cls in pred if names[int(cls)].lower() != 'abnormal')
        total = abnormal_count + normal_count

        # Akumulasi ke global
        total_normal   += normal_count
        total_abnormal += abnormal_count
        total_count    += total

        # Save result image
        img_with_boxes = draw_boxes(img0.copy(), pred, names)
        result_filename = f"result_{filename}"
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_path, img_with_boxes)

        # Simpan info per gambar (buat carousel)
        results.append({
            'filename': filename,
            'normal_count': normal_count,
            'abnormal_count': abnormal_count,
            'total_count': total,
            'image_url': url_for('static', filename=f'uploads/{result_filename}')
        })

    # === hitung summary global ===
    if total_count > 0:
        abnormal_percentage = round(total_abnormal / total_count * 100, 2)
        normal_percentage   = round(total_normal / total_count * 100, 2)
    else:
        abnormal_percentage = 0
        normal_percentage   = 0

    status = 'Abnormal' if abnormal_percentage > 50 else 'Normal'

    summary = {
        'normal_count': total_normal,
        'abnormal_count': total_abnormal,
        'total_count': total_count,
        'normal_percentage': normal_percentage,
        'abnormal_percentage': abnormal_percentage,
        'status': status
    }

    return render_template('result.html', results=results, summary=summary)


# === Run App ===
if __name__ == '__main__':
    app.run(debug=True)
