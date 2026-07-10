import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

def yolo_to_binary_mask(yolo_lines, img_width, img_height):
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    for line in yolo_lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
            
        x_center = float(parts[1])
        y_center = float(parts[2])
        box_width = float(parts[3])
        box_height = float(parts[4])
        
        x_min = int((x_center - box_width / 2) * img_width)
        y_min = int((y_center - box_height / 2) * img_height)
        x_max = int((x_center + box_width / 2) * img_width)
        y_max = int((y_center + box_height / 2) * img_height)
        
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_width, x_max)
        y_max = min(img_height, y_max)
        
        mask[y_min:y_max, x_min:x_max] = 255
        
    return mask

image_width = 1024
image_height = 768

# Danh sách chứa nhiều dòng dữ liệu YOLO
yolo_data_lines = [
    "0 0.562912 0.266850 0.020639 0.028393",
    "0 0.518685 0.471279 0.021295 0.024025",
    "0 0.613856 0.414275 0.021622 0.026209",
    "0 0.470362 0.349190 0.020967 0.029703"
]

binary_mask = yolo_to_binary_mask(yolo_data_lines, image_width, image_height)

root = tk.Tk()
root.withdraw() 

file_path = filedialog.asksaveasfilename(
    defaultextension=".png",
    filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
)

if file_path:
    cv2.imwrite(file_path, binary_mask)