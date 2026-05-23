import os
import cv2

def visualize_ground_truth(data_dir, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [f for f in os.listdir(data_dir) if os.path.splitext(f)[1].lower() in valid_extensions]

    count = 0
    for img_name in image_files:
        base_name = os.path.splitext(img_name)[0]
        txt_name = base_name + ".txt"
        
        img_path = os.path.join(data_dir, img_name)
        txt_path = os.path.join(data_dir, txt_name)
   
        if not os.path.exists(txt_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img_h, img_w = img.shape[:2]

        with open(txt_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                x_center = float(parts[1])
                y_center = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

                x1 = int((x_center - w / 2) * img_w)
                y1 = int((y_center - h / 2) * img_h)
                x2 = int((x_center + w / 2) * img_w)
                y2 = int((y_center + h / 2) * img_h)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(img, "Ground Truth: Fibroid", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, img)
        count += 1
        
    print(f"[+] Hoan tat! Da ve khung bao chuan cho {count} anh.")
    print(f"[+] Thu muc ket qua: {output_dir}")

if __name__ == '__main__':
    TEST_DIR = r"D:\AI_Data\test" 
    OUTPUT_DIR = r"D:\AI_Data\test_visualized_ground_truth" 
    
    visualize_ground_truth(TEST_DIR, OUTPUT_DIR)