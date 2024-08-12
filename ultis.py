import csv
import cv2
import os
import numpy as np
import copy

def extract_frames(video_path, frame_distance, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video stream or file")
        return []
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extracted_frames = []
    count = 0
    for frame_idx in range(0, frame_count, frame_distance):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR) #resize
        
        frame_path = os.path.join(output_folder, f"frame_{count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        count += 1
        extracted_frames.append(frame_path)

    cap.release()
    return extracted_frames

def read_homographies_from_csv(csv_file):
    H_list = []
    with open(csv_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        current_homography = []
        for row in reader:
            if row[0].startswith('H_'):
                if current_homography:
                    H_list.append(np.array(current_homography, dtype=float))
                    current_homography = []
            else:
                current_homography.append([float(x) for x in row])
        if current_homography:
            H_list.append(np.array(current_homography, dtype=float))
    return H_list

def findH_akaze(img1, img2):
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    rawMatches = bf.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for m, n in rawMatches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    return H

def findH_sift(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    rawMatches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in rawMatches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    return H

def findH_orb(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    rawMatches = bf.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for m, n in rawMatches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    return H

def get_combined_homography(homographies, n):
    if n == 0:
        return np.eye(3)
    else:
        H_prev = get_combined_homography(homographies, n-1)
        return np.dot(H_prev, homographies[n-1])

def apply_homography_to_corners(image, H):
    h, w = image.shape[:2]
    corners = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, H)
    return transformed_corners.reshape(-1, 2)

def calculate_output_size(images, homographies):
    all_corners = []
    for i, img in enumerate(images):
        H = get_combined_homography(homographies, i)
        corners = apply_homography_to_corners(img, H)
        all_corners.append(corners)
    all_corners = np.vstack(all_corners)
    min_x = np.min(all_corners[:, 0])
    min_y = np.min(all_corners[:, 1])
    max_x = np.max(all_corners[:, 0])
    max_y = np.max(all_corners[:, 1])
    output_width = int(np.ceil(max_x - min_x))
    output_height = int(np.ceil(max_y - min_y))
    return output_width, output_height, min_x, min_y

def warp_image_to_zero(image, homographies, n, output_shape, offset_x, offset_y):
    H = get_combined_homography(homographies, n)
    # Tạo ma trận dịch chuyển để dịch các ảnh về tọa độ dương
    offset_matrix = np.array([
        [1, 0, -offset_x],
        [0, 1, -offset_y],
        [0, 0, 1]
    ])
    H = np.dot(offset_matrix, H)
    warped_image = cv2.warpPerspective(image, H, output_shape)
    return warped_image

def draw_outer_bounding_box(image, color=(0, 255, 0), thickness=10):
    image_copy = copy.deepcopy(image)
    height, width = image_copy.shape[:2]
    cv2.rectangle(image_copy, (0, 0), (width-1, height-1), color, thickness)
    
    return image_copy

def calculate_similarity_percentage_orb(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    rawMatches = bf.knnMatch(des1, des2, k=2)
    matches = []

    for m, n in rawMatches:
        if m.distance < n.distance * 0.75:
            matches.append(m)

    total_keypoints = len(kp1) + len(kp2)
    num_matches = len(matches)
    if total_keypoints == 0:
        return 0.0  
    similarity_percentage = (2 * num_matches / total_keypoints) * 100
    return similarity_percentage

def calculate_similarity_percentage_akaze(img1, img2):
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    rawMatches = bf.knnMatch(des1, des2,2)
    matches = []

    for m, n in rawMatches:
        if m.distance < n.distance * 0.75:
            matches.append(m)

    total_keypoints = len(kp1) + len(kp2)
    num_matches = len(matches)
    if total_keypoints == 0:
        return 0.0  
    similarity_percentage = (2 * num_matches / total_keypoints) * 100
    return similarity_percentage

def calculate_similarity_percentage_sift(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    rawMatches = bf.knnMatch(des1, des2,2)
    matches = []

    for m, n in rawMatches:
        if m.distance < n.distance * 0.75:
            matches.append(m)

    total_keypoints = len(kp1) + len(kp2)
    num_matches = len(matches)
    if total_keypoints == 0:
        return 0.0  
    similarity_percentage = (2 * num_matches / total_keypoints) * 100
    return similarity_percentage

def draw_outer_bounding_box(image, color=(0, 255, 0), thickness=10):
    image_copy = copy.deepcopy(image)
    height, width = image_copy.shape[:2]
    cv2.rectangle(image_copy, (0, 0), (width-1, height-1), color, thickness)
    
    return image_copy

def concatenate_and_label_images(image1, image2, label1, label2):
    image1 = cv2.resize(image1, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    image2 = cv2.resize(image2, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]
    new_width = width1 + image2.shape[1]
    new_height = max(height1, height2)
    new_frame = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    new_frame[:, :width1] = image1
    x_offset = width1
    y_offset = int((height1/2-height2)/2)
    new_frame[y_offset:y_offset+image2.shape[0], x_offset:x_offset + image2.shape[1]] = image2

    label_font = cv2.FONT_HERSHEY_SIMPLEX
    label_font_scale = 1
    label_color = (255, 255, 255)
    label_thickness = 3

    label1_position = (10, 30)
    cv2.putText(new_frame, label1, label1_position, label_font, label_font_scale, label_color, label_thickness)

    label2_position = (x_offset + 10, 30)
    cv2.putText(new_frame, label2, label2_position, label_font, label_font_scale, label_color, label_thickness)

    return new_frame

def create_video_from_frames(folder_path, output_path, frame_duration):
    images = [img for img in os.listdir(folder_path) if img.endswith(".png") or img.endswith(".jpg")]
    images.sort()  

    frame = cv2.imread(os.path.join(folder_path, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, 1.0 / frame_duration, (width, height))

    for image in images:
        img_path = os.path.join(folder_path, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()
    cv2.destroyAllWindows()
