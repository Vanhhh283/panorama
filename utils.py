import csv
import cv2
import os
import numpy as np
import copy

def extract_frames(video_path, frame_distance, output_folder, resize):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video stream or file")
        return []
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(frame_count)
    extracted_frames = []
    count = 0
    for frame_idx in range(0, frame_count, frame_distance):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print("break")
            break
        
        frame = cv2.resize(frame, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR) #resize
        
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

def get_combined_homography_1(homographies, n):
    if n == 0:
        return np.eye(3)
    else:
        H_prev = get_combined_homography_1(homographies, n-1)
        return np.dot(H_prev, homographies[n-1])
    
def findH_list(homographies):
    H_list = []
    H_list.append(np.eye(3))
    for n in range(len(homographies)):
        Hn = np.dot(H_list[-1],homographies[n])
        # if (abs(sum(Hn[2,:])-1)>0.12):
        #     Hn[2,:] = [0,0,1]
        H_list.append(Hn)
    return H_list

def get_combined_homography(homographies, n):
    H_list = findH_list(homographies)
    return H_list[n]



# def get_combined_homography_2(homo, n):
#     H = get_combined_homography_1(homo, n)
#     H[2,0] = 0
#     H[2,1] = 0
#     return H

# from scipy.linalg import svd

# def get_combined_homography(H, n):
#     H = get_combined_homography_2(H,n)
#     R = H[:2, :2]

#     U, _, Vt = svd(R)
#     R_rot = np.dot(U, Vt)

#     H_new = H.copy()  # Sao chép H để không thay đổi ma trận gốc
#     H_new[:2, :2] = R_rot

#     return H_new

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
    # A = H[:2, :3]
    # A[0,0] = 1
    # A[1,1] = 1
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
    image1 = cv2.resize(image1, (image1.shape[1] // 2, image1.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
    image2 = cv2.resize(image2, (image2.shape[1] // 2, image2.shape[0] // 2), interpolation=cv2.INTER_LINEAR)

    # Giữ nguyên kích thước image1, chỉ thay đổi kích thước image2 theo kích thước của image1
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    # Thay đổi kích thước của image2 sao cho chiều cao của nó bằng chiều cao của image1
    new_width2 = int(width2 * (height1 / height2))
    image2 = cv2.resize(image2, (new_width2, height1), interpolation=cv2.INTER_LINEAR)

    # Tạo khung cho ảnh mới với chiều rộng bằng tổng chiều rộng của cả hai ảnh và chiều cao bằng image1
    new_width = width1 + new_width2
    new_frame = np.zeros((height1, new_width, 3), dtype=np.uint8)

    # Ghép image1 vào phần đầu của khung mới
    new_frame[:, :width1] = image1

    # Ghép image2 vào sau image1
    x_offset = width1
    new_frame[:, x_offset:x_offset + new_width2] = image2

    # Thêm nhãn vào từng ảnh
    label_font = cv2.FONT_HERSHEY_SIMPLEX
    label_font_scale = 1
    label_color = (255, 255, 255)
    label_thickness = 3

    # Vị trí của nhãn 1 (ở ảnh đầu tiên)
    label1_position = (10, 30)
    cv2.putText(new_frame, label1, label1_position, label_font, label_font_scale, label_color, label_thickness)

    # Vị trí của nhãn 2 (ở ảnh thứ hai)
    label2_position = (x_offset + 10, 30)
    cv2.putText(new_frame, label2, label2_position, label_font, label_font_scale, label_color, label_thickness)

    return new_frame

def create_video_from_frames(folder_path, output_path, fps):
    images = [img for img in os.listdir(folder_path) if img.endswith(".png") or img.endswith(".jpg")]
    images.sort()  

    frame = cv2.imread(os.path.join(folder_path, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for image in images:
        img_path = os.path.join(folder_path, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()
    cv2.destroyAllWindows()

def compare(A, B, compare_method="akaze", output_csv="tracking_output.csv"):
    """
    A: danh sách các khung (array) của ảnh pano gốc
    B: danh sách các khung (array) của video cần tracking
    compare_method: thuật toán sử dụng để so sánh (akaze, sift, orb)
    output_csv: đường dẫn xuất ra file CSV chứa thông tin so sánh về các khung
    """
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame_B", "Match_1", "Match_2", "Match_3"])

        for i in range(len(B)):
            scores = []
            img1 = cv2.cvtColor(B[i], cv2.COLOR_BGR2GRAY)

            for j in range(len(A)):
                img2 = cv2.cvtColor(A[j], cv2.COLOR_BGR2GRAY)
                
                if compare_method == "akaze":
                    similarity_percentage = calculate_similarity_percentage_akaze(img1, img2)
                elif compare_method == "sift":
                    similarity_percentage = calculate_similarity_percentage_sift(img1, img2)
                elif compare_method == "orb":
                    similarity_percentage = calculate_similarity_percentage_orb(img1, img2)
                else:
                    print("Không tìm được phương thức tính toán tương ứng")
                    return
                
                scores.append(similarity_percentage)

            sorted_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            top_3_indices = [index for index, _ in sorted_indices[:3]]
            writer.writerow([i] + top_3_indices)
        print("Compare list saved to " + output_csv)


def transform_array(arr):
    # Tạo một mảng mới với cùng shape và dtype như mảng ban đầu
    transformed_arr = np.zeros_like(arr, dtype=np.uint8)
    
    # Thay thế các giá trị 0 thành 255 và các giá trị khác 0 thành 0
    transformed_arr[arr == 0] = 255
    
    return transformed_arr

def add_black_border(image, border_thickness):
    # Biến viền ngoài về màu đen
    copy_image = copy.deepcopy(image)
    copy_image[:border_thickness, :, :] = 0  # Viền trên
    copy_image[-border_thickness:, :, :] = 0  # Viền dưới
    copy_image[:, :border_thickness, :] = 0  # Viền trái
    copy_image[:, -border_thickness:, :] = 0  # Viền phải

    return copy_image