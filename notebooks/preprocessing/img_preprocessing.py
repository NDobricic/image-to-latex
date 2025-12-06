import cv2
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt

# RESIZE AND PAD

def resize_and_pad(img, target_width=400, target_height=96, pad_value=255):
    h, w = img.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_height, target_width), pad_value, dtype=np.uint8)
    top = (target_height - new_h) // 2
    left = (target_width - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = resized
    return canvas

# BINARIZATION METHODS

def sauvola_binarize(img, window=35, k=0.20, R=128):
    img_f = img.astype(np.float32)
    mean = cv2.boxFilter(img_f, -1, (window, window))
    sqmean = cv2.boxFilter(img_f * img_f, -1, (window, window))
    var = sqmean - mean**2
    std = np.sqrt(np.maximum(var, 0))
    thresh = mean * (1 + k * ((std / R) - 1))
    return ((img_f > thresh).astype(np.uint8) * 255)

def adaptive_gaussian_binarize(img, blockSize=21, C=8):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, blockSize, C)

# BLACK BORDER REMOVAL

def remove_black_border(img):
    """
    Removes black frame in binary images (0/255)
    """
    assert img.ndim == 2
    inv = 255 - img
    coords = cv2.findNonZero(inv)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y+h, x:x+w]

# PERSPECTIVE CORRECTION

def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def correct_perspective(img):
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray
    c = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*peri, True)
    if len(approx) != 4:
        return gray
    pts = approx.reshape(4,2)
    rect = order_points(pts)
    (tl,tr,br,bl) = rect
    widthA = np.linalg.norm(br-bl)
    widthB = np.linalg.norm(tr-tl)
    heightA = np.linalg.norm(tr-br)
    heightB = np.linalg.norm(tl-bl)
    maxW, maxH = int(max(widthA, widthB)), int(max(heightA, heightB))
    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(gray, M, (maxW, maxH))

# PREPROCESSING VARIANTS (BINARY ONLY)

def preprocess_variant_soft(img, W=400, H=96):
    sm = cv2.GaussianBlur(img, (3,3), 0)
    binary = sauvola_binarize(sm, window=35, k=0.15)
    return resize_and_pad(binary, W, H)

def preprocess_variant_hard(img, W=400, H=96):
    binary = adaptive_gaussian_binarize(img, blockSize=21, C=8)
    if np.mean(binary) < 128:
        binary = 255 - binary
    binary = remove_black_border(binary)
    k = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k)
    return resize_and_pad(cleaned, W, H)

# BEST VARIANT SELECTION

def get_best_soft_hard_image(img, W=400, H=96):
    """
    Selects the best binarized variant between soft and hard.
    Prefers solid text and penalizes tiny noisy specks.
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    soft_img = preprocess_variant_soft(img, W, H)
    hard_img = preprocess_variant_hard(img, W, H)

    def score(im):

        # invert
        text = (255 - im) // 255
        text_u8 = text.astype(np.uint8)
        
        # connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(text_u8, connectivity=8)
        
        # ignore background label 0
        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas) == 0:
            return 0
        
        # filter out tiny components as noise 
        good_components = areas[areas >= 10]
        if len(good_components) == 0:
            return 0
        
        # average size of significant components
        avg_size = np.mean(good_components)
        
        # coverage fraction of significant components
        coverage = np.sum(good_components) / text.size
        
        # Sobel inside significant components
        mask = np.isin(labels, np.where(areas >= 10)[0]+1).astype(np.uint8)
        sobelx = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=3)
        edge_mag = np.sqrt(sobelx**2 + sobely**2)
        edge_score = np.sum(edge_mag * mask)
        
        return edge_score * coverage * (avg_size / 100)

    score_soft = score(soft_img)
    score_hard = score(hard_img)

    if score_hard >= score_soft:
        return hard_img, 'hard'
    else:
        return soft_img, 'soft'

    
# UNIVERSAL PIPELINE

def preprocess_photo_universal(img, W=400, H=96):
    warped = correct_perspective(img)
    soft = preprocess_variant_soft(warped, W, H)
    hard = preprocess_variant_hard(warped, W, H)
    best, best_name = get_best_soft_hard_image(warped, W, H)
    return {'soft': soft, 'hard': hard, 'best': best, 'best_name': best_name}

# DISPLAY UTILITIES

def show_variants(d):
    """
    Displays soft, hard, and best variants side by side
    """
    plt.figure(figsize=(12,4))
    for i, (name,img) in enumerate(d.items()):
        if name == 'best_name':
            continue
        plt.subplot(1,3,i+1)
        plt.imshow(img, cmap='gray')
        plt.title(name if name != 'best' else d['best_name'])
        plt.axis('off')
    plt.show()
