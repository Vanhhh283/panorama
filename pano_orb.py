import cv2
import numpy as np
import os
import csv 

from tqdm import tqdm
from utils import *

videos = [1, 2, 3, 4, 5, 6, 7, 8, 9] 
distances = [120]
for i, video in tqdm(enumerate(videos)):
    for j, frame_distance in tqdm(enumerate(distances)):

        """Lấy mẫu"""
        video_str = str(video)
        input_path = "video" + video_str +".mp4"
        dis = str(frame_distance)
        output_folder = "video" + video_str + "_" + dis
        extracted_frames = extract_frames(input_path, frame_distance, output_folder, resize = 0.5)

        """Tính H"""
        folder_path = output_folder
        image_files = os.listdir(folder_path)
        image_files.sort()
        images = [cv2.imread(os.path.join(folder_path, img_file)) for img_file in image_files]
        H = []
        for i, image in (enumerate(images[1:], 1)):
            homo = findH_orb(images[i-1], image)
            H.append(homo)

        csv_file = "result/video" + video_str + "_" + dis + ".csv"
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            for i, homo in enumerate(H):
                writer.writerow([f"H_{i}"])
                for row in homo:
                    writer.writerow(row)
            print(f"Homographies have been saved to {csv_file}")

        """Ghép ảnh"""
        result_img = "result/video" + video_str + "_" + dis + "_result.jpg"

        output_width, output_height, min_x, min_y = calculate_output_size(images, H)
        output_shape = (output_width, output_height)
        print(output_shape,min_x,min_y)

        result = np.zeros((output_shape[1], output_shape[0], 3), dtype=np.uint8)

        for i, image in enumerate(images):
            warped_image = warp_image_to_zero(image, H, i, output_shape, min_x, min_y)
            result = np.maximum(result, warped_image)
            cv2.imwrite(result_img, result)

        print(f"Result save as {result_img}")   

        """Cải thiện chất lượng"""
        index_replace = np.arange(0, len(images)-1, 10)

        for i in index_replace:
            copy_image = add_black_border(images[i],border_thickness=2)
            warped_image = warp_image_to_zero(images[i], H, i, output_shape, min_x, min_y)
            z_image = warp_image_to_zero(copy_image, H, i, output_shape, min_x, min_y)
            z_image = transform_array(z_image)
            result = np.minimum(result, z_image)
            result = np.maximum(result, warped_image)
            cv2.imwrite(result_img, result)
