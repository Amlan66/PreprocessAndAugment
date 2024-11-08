import cv2
import numpy as np
from PIL import Image
import io
import base64

def augment_image(image_bytes, method='horizontal_flip'):
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if method == 'horizontal_flip':
        augmented = cv2.flip(img, 1)
    elif method == 'rotation':
        # Random rotation between -30 and 30 degrees
        angle = np.random.uniform(-30, 30)
        height, width = img.shape[:2]
        matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
        augmented = cv2.warpAffine(img, matrix, (width, height))
    elif method == 'noise':
        # Add Gaussian noise
        noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
        augmented = cv2.add(img, noise)
    
    # Convert to base64 for display
    _, buffer = cv2.imencode('.jpg', augmented)
    img_str = base64.b64encode(buffer).decode()
    
    return f'data:image/jpeg;base64,{img_str}' 