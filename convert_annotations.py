import os
import shutil
import random

def main():
    # 1. Định nghĩa đường dẫn nguồn và đích
    # Thay đường dẫn này bằng thư mục hiện tại đang chứa 871 ảnh và 871 xml của bạn
    source_dir = r"D:\AI_Data\all_data_raw"  
    output_base = r"D:\AI_Data"
    
    train_dir = os.path.join(output_base, "train")
    val_dir = os.path.join(output_base, "val")
    
    # Tạo các thư mục đích nếu chưa tồn tại
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # 2. Quét và tìm các tệp xml có sẵn
    all_files = os.listdir(source_dir)
    xml_files = [f for f in all_files if f.lower().endswith('.xml')]
    
    valid_pairs = []
    # Định dạng đuôi ảnh phổ biến (hệ thống tự khớp đuôi thực tế)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.PNG']
    
    for xml_file in xml_files:
        base_name = os.path.splitext(xml_file)[0]
        
        # Tìm tệp ảnh trùng tên tương ứng
        image_file = None
        for ext in image_extensions:
            potenial_img = base_name + ext
            if potenial_img in all_files:
                image_file = potenial_img
                break
                
        if image_file:
            valid_pairs.append((image_file, xml_file))
            
    total_pairs = len(valid_pairs)
    print(f"Tìm thấy tổng cộng: {total_pairs} cặp ảnh và XML hợp lệ trùng tên.")
    
    if total_pairs < 171:
        print("Lỗi: Số lượng cặp dữ liệu ít hơn 171, không thể phân tách tập val.")
        return

    # 3. Xáo trộn ngẫu nhiên để đảm bảo tính khách quan của dữ liệu
    random.seed(42) # Cố định seed để có kết quả đồng nhất nếu chạy lại
    random.shuffle(valid_pairs)
    
    # Tách chính xác 171 cặp cho tập kiểm thử (val)
    val_pairs = valid_pairs[:171]
    train_pairs = valid_pairs[171:]
    
    # 4. Thực hiện sao chép tệp vào các thư mục tương ứng
    print("Đang sao chép dữ liệu vào thư mục tập kiểm thử (val)...")
    for img, xml in val_pairs:
        shutil.copy(os.path.join(source_dir, img), os.path.join(val_dir, img))
        shutil.copy(os.path.join(source_dir, xml), os.path.join(val_dir, xml))
        
    print("Đang sao chép dữ liệu vào thư mục tập huấn luyện (train)...")
    for img, xml in train_pairs:
        shutil.copy(os.path.join(source_dir, img), os.path.join(train_dir, img))
        shutil.copy(os.path.join(source_dir, xml), os.path.join(train_dir, xml))
        
    print(f"Hoàn tất phân chia dữ liệu.")
    print(f"Thư mục '{val_dir}': {len(val_pairs)} ảnh + {len(val_pairs)} XML.")
    print(f"Thư mục '{train_dir}': {len(train_pairs)} ảnh + {len(train_pairs)} XML.")

if __name__ == "__main__":
    main()