import io

import cv2
from PIL import Image

def bgr2png(bgr):
    # convert bgr to rgb
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # convert rgb to Pillow image
    img = Image.fromarray(rgb, "RGB")
    
    output = io.BytesIO()
    img.save(output, format='PNG')
    output.seek(0)
    img_data = output.read()
    output.close()
    
    return img_data
