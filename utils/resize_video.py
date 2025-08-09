import cv2

def resize_with_aspect_ratio(image, width=None, height=None, color=(0, 0, 0)):
    h, w = image.shape[:2]
    if width is None and height is None:
        return image

    # Calculate scale to fit inside target box
    if width and height:
        r = min(width / w, height / h)
        new_w = int(w * r)
        new_h = int(h * r)
    elif width:
        r = width / w
        new_w = width
        new_h = int(h * r)
    else:
        r = height / h
        new_w = int(w * r)
        new_h = height

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Padding
    if width and height:
        delta_w = width - new_w
        delta_h = height - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        resized = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return resized