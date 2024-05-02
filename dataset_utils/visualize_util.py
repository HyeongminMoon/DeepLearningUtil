def draw_border(img, pt1, pt2, color, thickness, r, d):
        x1,y1 = pt1
        x2,y2 = pt2
        # Top left
        cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        # Top right
        cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        # Bottom left
        cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
        cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        # Bottom right
        cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
        cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)

        cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
        cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
        cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
        cv2.circle(img, (x2 -r, y2-r), 2, color, 12)

        return img

def UI_box(img, bbox, color, label):
    # Plots one bounding box on image img
    tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        img = draw_border(img, (bbox[0], bbox[1] - t_size[1] -3), (bbox[0] + t_size[0], bbox[1]+3), color, 1, 8, 2)
        cv2.putText(img, label, (bbox[0], bbox[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def resize_with_pad(image, target_size):
    target_width, target_height = target_size
    # Get the dimensions of the original image
    height, width = image.shape[:2]

    # Calculate the ratio for resizing
    ratio = min(target_width / width, target_height / height)

    # Calculate the new dimensions
    new_width = int(width * ratio)
    new_height = int(height * ratio)

    # Resize the image using the calculated dimensions
    resized_image = cv2.resize(image, (new_width, new_height))

    # Create a blank canvas with the target dimensions
    padded_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Calculate the position to place the resized image with padding
    start_x = (target_width - new_width) // 2
    start_y = (target_height - new_height) // 2

    # Place the resized image onto the canvas
    padded_image[start_y:start_y + new_height, start_x:start_x + new_width] = resized_image

    return padded_image

def load_video(file_path, image_size=None, original_fps=30, new_fps=5, start_time=None, end_time=None, gray=False, padding=True, is_float=False):
    """Loads a video file into a TF tensor."""
    cap = cv2.VideoCapture(file_path)
    
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    ret = True

    if image_size is None:
        image_size = (frameWidth, frameHeight)
    
    if start_time and end_time:
        start_frame = original_fps * start_time
        end_frame = original_fps * end_time
    else:
        start_frame = 0
        end_frame = frameCount
        
    fc = 0    
    
    fps_factor = original_fps / new_fps
    frame_loc = 0
    now_frame = 0

    # print(end_frame, start_frame, fps_factor)
    # print(int((end_frame - start_frame) / fps_factor))

    if gray:
        buf = np.zeros((int((end_frame - start_frame) / fps_factor), image_size[1], image_size[0]), np.dtype('uint8'))
    else:
        buf = np.zeros((int((end_frame - start_frame) / fps_factor), image_size[1], image_size[0], 3), np.dtype('uint8'))

    while (fc < end_frame - start_frame  and ret):
        ret, tmp = cap.read()
        now_frame += 1
        frame_loc += 1
        if start_frame > now_frame:
            continue
        if end_frame < now_frame:
            break
        if frame_loc > fps_factor:
            if padding:
                tmp = resize_with_pad(tmp, image_size)
            else:
                tmp = cv2.resize(tmp, dsize=image_size)
            if gray:
                buf[fc] = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
            else:
                buf[fc] = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
            fc += 1
            frame_loc -= fps_factor
    cap.release()
    
    if is_float:
        buf = tf.convert_to_tensor(buf)
        buf = tf.image.resize(buf, image_size)
        buf = tf.cast(buf, tf.float32) / 255.
    
    return buf
