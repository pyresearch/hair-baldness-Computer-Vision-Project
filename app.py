from flask import Flask, render_template, request, redirect, url_for
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import os
import pyresearch

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure upload and result directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO("last.pt")  # Ensure 'last.pt' is the correct path to your model
names = model.model.names  # Class names from the model

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html', marketing_info={
        'phone': '+966539723031',
        'channel_membership': 'https://www.youtube.com/channel/UCyB_7yHs7y8u9rONDSXgkCg/join',
        'facebook': 'https://www.facebook.com/Pyresearch',
        'youtube': 'https://www.youtube.com/c/Pyresearch',
        'medium': 'https://medium.com/@Pyresearch',
        'instagram': 'https://www.instagram.com/pyresearch/',
        'linkedin': 'https://www.linkedin.com/company/Pyresearch',
        'twitter': 'https://twitter.com/Noorkhokhar10',
        'discord': 'https://discord.com/invite/BHxGBn98',
        'github': 'https://github.com/Pyresearch',
        'quora': 'https://www.quora.com/profile/Pyresearch',
        'personal_github': 'https://github.com/noorkhokhar99'
    })

@app.route('/upload', methods=['POST'])
def upload():
    if 'images' not in request.files:
        return redirect(request.url)

    files = request.files.getlist('images')
    if not files:
        return redirect(request.url)

    result_paths = []
    detected_classes = set()

    for file in files:
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Process the image
            image = cv2.imread(filepath)
            if image is None:
                continue

            results = model.predict(image)
            annotator = Annotator(image, line_width=2)

            if results and len(results) > 0:
                result = results[0]
                if result.masks is not None and len(result.masks) > 0:
                    clss = result.boxes.cls.cpu().tolist()  # Class IDs
                    masks = result.masks.xy  # Segmentation masks
                    for mask, cls in zip(masks, clss):
                        color = colors(int(cls), True)
                        txt_color = annotator.get_txt_color(color)
                        annotator.seg_bbox(mask=mask, mask_color=color, label=names[int(cls)], txt_color=txt_color)
                        detected_classes.add(names[int(cls)])

            # Save the annotated image
            output_filename = 'result_' + filename
            output_path = os.path.join(RESULT_FOLDER, output_filename)
            cv2.imwrite(output_path, image)
            result_paths.append({'input': filepath, 'output': output_path})

    return render_template('result.html', result_paths=result_paths, detected_classes=list(detected_classes), marketing_info={
        'phone': '+966539723031',
        'channel_membership': 'https://www.youtube.com/channel/UCyB_7yHs7y8u9rONDSXgkCg/join',
        'facebook': 'https://www.facebook.com/Pyresearch',
        'youtube': 'https://www.youtube.com/c/Pyresearch',
        'medium': 'https://medium.com/@Pyresearch',
        'instagram': 'https://www.instagram.com/pyresearch/',
        'linkedin': 'https://www.linkedin.com/company/Pyresearch',
        'twitter': 'https://twitter.com/Noorkhokhar10',
        'discord': 'https://discord.com/invite/BHxGBn98',
        'github': 'https://github.com/Pyresearch',
        'quora': 'https://www.quora.com/profile/Pyresearch',
        'personal_github': 'https://github.com/noorkhokhar99'
    })
if __name__ == '__main__':
    # Running on all interfaces, so other devices on your network can access it
    # Replace 5000 with another port if needed
    app.run(host='0.0.0.0', port=80, debug=True)
