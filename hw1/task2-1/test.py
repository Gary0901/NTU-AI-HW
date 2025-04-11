from PIL import Image
import os
import re

def resize_specific_images(input_folder, output_folder, size=(224, 224)):
    """
    調整指定資料夾中的照片大小，並重新命名後保存到輸出資料夾
    
    參數:
    input_folder: 輸入資料夾路徑 (含有 stylized_000001 到 stylized_000100 的圖片)
    output_folder: 輸出資料夾路徑 (將保存為 000001 到 000100 的圖片)
    size: 目標尺寸，預設為(224, 224)
    """
    # 如果輸出資料夾不存在，則創建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"已創建輸出資料夾: {output_folder}")
    
    # 定義檔案名稱的正則表達式模式
    pattern = re.compile(r'stylized_(\d+)')
    
    # 計數器
    processed = 0
    errors = 0
    
    # 遍歷輸入資料夾中的所有文件
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        
        # 檢查文件是否為圖片且符合命名格式
        if os.path.isfile(input_path) and filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
            # 使用正則表達式匹配檔案名稱
            match = pattern.search(filename)
            if match:
                number = match.group(1)  # 提取數字部分
                new_filename = f"{number}"  # 新檔案名稱
                output_path = os.path.join(output_folder, new_filename + os.path.splitext(filename)[1])
                
                try:
                    # 打開圖片
                    with Image.open(input_path) as img:
                        # 調整圖片大小
                        resized_img = img.resize(size, Image.LANCZOS)
                        
                        # 保存調整後的圖片
                        resized_img.save(output_path)
                        processed += 1
                        print(f"已處理: {filename} -> {new_filename}{os.path.splitext(filename)[1]}")
                except Exception as e:
                    print(f"處理 {filename} 時發生錯誤: {e}")
                    errors += 1
    
    print(f"處理完成: 成功處理 {processed} 個圖片, 失敗 {errors} 個")

if __name__ == "__main__":
    # 指定輸入和輸出資料夾
    input_folder = "output_stylized"
    output_folder = "r13922154_stylized_images"
    
    # 執行圖片調整和重命名
    resize_specific_images(input_folder, output_folder)
    
    print("程式執行完畢！")