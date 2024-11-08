import cv2
import numpy as np
import base64

def preprocess_image(image_bytes, method='resize', params=None):
    # Set default params if None
    if params is None:
        params = {}
    
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if method == 'resize':
        # Default size of 224x224 if not specified
        target_size = (224, 224)
        processed = cv2.resize(img, target_size)
    elif method == 'normalize':
        # Convert to float32
        processed = img.astype(np.float32)
        
        # Scale to [-1, 1] range
        # First scale to [0,1] by dividing by 255
        processed = processed / 255.0
        # Then scale to [-1,1]
        processed = (processed * 2) - 1
        
        # Convert back to uint8 for display
        # First scale back to [0,1]
        display_img = (processed + 1) / 2
        # Then scale to [0,255] and convert to uint8
        processed = (display_img * 255).astype(np.uint8)
    else:
        processed = img  # Return original image if method not recognized
    
    # Convert to base64 for display
    _, buffer = cv2.imencode('.jpg', processed)
    img_str = base64.b64encode(buffer).decode()
    
    return f'data:image/jpeg;base64,{img_str}'