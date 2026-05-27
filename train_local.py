import os
import xml.etree.ElementTree as ET
import wandb
from ultralytics import YOLO

# def convert_xml_to_txt_high_precision(data_dir):
#     for folder in ['train', 'val']:
#         folder_path = os.path.join(data_dir, folder)
#         if not os.path.exists(folder_path):
#             print(f"[-] CANH BAO: Khong tim thay thu muc: {folder_path}")
#             continue
            
#         all_files = os.listdir(folder_path)
        
#         has_txt_files = any(f.lower().endswith('.txt') for f in all_files)
#         if has_txt_files:
#             print(f"[->] Thu muc [{folder.upper()}]: Da co san file .txt. Tu dong bo qua buoc convert.")
#             continue
            
#         image_basenames = {
#             os.path.splitext(f)[0] for f in all_files 
#             if not f.lower().endswith('.xml') and not f.lower().endswith('.txt')
#         }
        
#         xml_count = 0
#         converted_count = 0
        
#         for file_name in all_files:
#             if not file_name.lower().endswith('.xml'):
#                 continue
                
#             xml_count += 1
#             base_name = os.path.splitext(file_name)[0]
            
#             if base_name not in image_basenames:
#                 continue
                
#             xml_path = os.path.join(folder_path, file_name)
#             try:
#                 tree = ET.parse(xml_path)
#                 root = tree.getroot()
                
#                 size_elem = root.find('size')
#                 if size_elem is None:
#                     continue
#                 width = float(size_elem.find('width').text)
#                 height = float(size_elem.find('height').text)
                
#                 if width == 0 or height == 0:
#                     continue

#                 yolo_boxes = []
#                 valid_tags = ["肌瘤", "Fibroid"]
                
#                 for obj in root.findall('object'):
#                     tag_name = obj.find('name').text.strip()
#                     if tag_name in valid_tags:
#                         bndbox = obj.find('bndbox')
#                         xmin = float(bndbox.find('xmin').text)
#                         ymin = float(bndbox.find('ymin').text)
#                         xmax = float(bndbox.find('xmax').text)
#                         ymax = float(bndbox.find('ymax').text)
                        
#                         x_center = ((xmin + xmax) / 2.0) / width
#                         y_center = ((ymin + ymax) / 2.0) / height
#                         w = (xmax - xmin) / width
#                         h = (ymax - ymin) / height
                        
#                         yolo_boxes.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
                
#                 if yolo_boxes:
#                     txt_path = os.path.join(folder_path, base_name + ".txt")
#                     with open(txt_path, 'w', encoding='utf-8') as f:
#                         f.write("\n".join(yolo_boxes))
#                     converted_count += 1
#             except Exception as e:
#                 print(f"[!] Loi he thong khi doc file {file_name}: {e}")
                
#         print(f"[+] Thu muc [{folder}]: Quet {xml_count} file XML -> Dich thanh cong {converted_count} file .txt")

def main():
    data_base_dir = r"D:\AI_Data"
    
    # print(f"\n=== TIEN TRINH KIEM TRA DU LIEU GOC TAI: {data_base_dir} ===")
    # print(f"Kiem tra thu muc 'train' ton tai: {os.path.exists(os.path.join(data_base_dir, 'train'))}")
    # print(f"Kiem tra thu muc 'val' ton tai: {os.path.exists(os.path.join(data_base_dir, 'val'))}")
    
    # convert_xml_to_txt_high_precision(data_base_dir)
    # print("\n--- Hoan tat kiem tra va chuyen doi file ---")
    # print("Kich hoat tien trinh huan luyen mo hinh YOLOv11...")

    wandb.init(
        entity="set-g64",
        project="AI_SYSTEM_FOR_UTERINE_ULTRASOUND_IMAGE_ANALYSIS_AND_ABNORMALITY_DETECTION",
        name="YOLOv11s_416_Augmented_Execution"
    )
    
    model = YOLO("yolo11s.pt")
    
    model.train(
        data="data.yaml",                    
        project=os.path.join(data_base_dir, "training_runs"),  
        name="YoLov11_Small_416_Augmented",
        epochs=150,                          
        batch=8,                             
        imgsz=416,                           
        device=0,                            
        workers=2,                           
        optimizer='AdamW',
        lr0=1e-4,
        patience=20,
        degrees=15.0,
        flipud=0.5,
        fliplr=0.5,
        scale=0.5,
        perspective=0.0001,
        mosaic=1.0
    )

if __name__ == '__main__':
    main()