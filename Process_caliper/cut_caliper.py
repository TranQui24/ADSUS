import cv2
import numpy as np

input_image_file = "origin_picture.jpg"
output_raw_crop = "debug_1_anh_cat_tho.png"
output_mask_processing = "debug_2_mat_na_tach_nhieu.png"
output_final_cleaned = "caliper+_final_chuan_sach.png"

img = cv2.imread(input_image_file)

if img is not None:
    roi = cv2.selectROI("Chon vung chua dau + va nhan SPACE hoac ENTER", img, False, False)
    cv2.destroyAllWindows()
    
    x, y, w, h = roi
    
    if w > 0 and h > 0:
        marker_square = img[y:y+h, x:x+w]
        cv2.imwrite(output_raw_crop, marker_square)

        gray_marker = cv2.cvtColor(marker_square, cv2.COLOR_BGR2GRAY)
        
        _, mask = cv2.threshold(gray_marker, 160, 255, cv2.THRESH_BINARY)
        cv2.imwrite(output_mask_processing, mask)

        b, g, r = cv2.split(marker_square)
        transparent_marker = cv2.merge([b, g, r, mask])
        
        cv2.imwrite(output_final_cleaned, transparent_marker)