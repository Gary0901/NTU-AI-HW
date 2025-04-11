import os
import torch
import time
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

# 設置設備和環境
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用設備: {device}")

# 設置路徑
input_dir = "my_pic"  # 原始臉部圖像目錄
output_dir = "output"  # 輸出目錄
os.makedirs(output_dir, exist_ok=True)

# 1. 載入模型：Phi-4-multimodal (MLLM模型)
print("正在載入 Phi-4-multimodal-instruct 模型...")
phi_processor = AutoProcessor.from_pretrained("microsoft/Phi-4-multimodal-instruct")
phi_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-4-multimodal-instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 2. 載入模型：Stable Diffusion v1-5 (T2I模型)
print("正在載入 Stable Diffusion v1-5 模型...")
sd_pipeline = StableDiffusionPipeline.from_pretrained(
    "sd-legacy/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
)
sd_pipeline = sd_pipeline.to(device)

# 優化VRAM使用
sd_pipeline.enable_attention_slicing()  # 減少VRAM使用量

# Snoopy風格的固定提示詞前綴
style_prefix = "In the style of Snoopy from Peanuts, cartoon, cute, rounded character designs, simple black line art, bold colors, minimalist backgrounds, iconic yellow zigzag shirt, white beagle with black ears, Blue Sky Studios 3D rendering, smooth textures, simplified features, clean outlines, bright color palette, large round heads, small bodies"

def generate_image_description(image_path):
    """使用Phi-4生成圖像描述"""
    # 載入圖像
    image = Image.open(image_path)
    
    # 準備提示詞
    prompt = "<|user|><|image_1|>Describe this person's appearance in detail. Focus on facial features, hairstyle, expression, and any notable characteristics.<|end|><|assistant|>"
    
    # 處理輸入
    inputs = phi_processor(
        text=prompt,
        images=[image],
        return_tensors="pt"
    ).to(device)
    
    # 顯示生成進度
    desc_progress = tqdm(total=1, desc="生成圖像描述", position=1, leave=False)
    
    # 生成描述
    start_time = time.time()
    with torch.no_grad():
        outputs = phi_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    end_time = time.time()
    desc_progress.update(1)
    desc_progress.close()
    
    # 解碼輸出
    description = phi_processor.decode(outputs[0], skip_special_tokens=False)
    
    # 提取助手回覆部分
    assistant_part = description.split("<|assistant|>")[1].strip()
    if "<|end|>" in assistant_part:
        assistant_part = assistant_part.split("<|end|>")[0].strip()
    
    elapsed = end_time - start_time
    print(f"  描述生成耗時: {elapsed:.2f}秒")
    print(f"  生成的描述: {assistant_part[:100]}..." if len(assistant_part) > 100 else f"  生成的描述: {assistant_part}")
    
    # 清除CUDA緩存以節省記憶體
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return assistant_part

def create_stylized_image(input_image_path, description):
    """使用StableDiffusion生成風格化圖像"""
    
    input_image = Image.open(input_image_path)
    
    # 結合描述和風格
    prompt = f"{style_prefix}{description}"
    
    # 確保輸入圖像大小為512x512
    input_image_resized = input_image.resize((512, 512), Image.LANCZOS) if input_image.size != (512, 512) else input_image
    
    # 使用tqdm在終端顯示生成進度
    gen_progress = tqdm(total=50, desc="生成Snoopy風格圖像", position=1, leave=False)
    
    def callback_fn(step, timestep, latents):
        gen_progress.update(1)
        return None
    
    # 生成圖像
    try:
        image = sd_pipeline(
            prompt=prompt,
            guidance_scale=7.5,
            num_inference_steps=50,
            callback=callback_fn,
            callback_steps=1,
        ).images[0]
    finally:
        gen_progress.close()
    
    # 調整輸出圖像大小為224x224
    image = image.resize((224, 224), Image.LANCZOS)
    
    # 清除CUDA緩存以節省記憶體
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return image

def process_image(image_path, idx, total):
    """處理單張圖像的完整流程"""
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"snoopy_{filename}")
    
    # 檢查是否已經處理過此圖像
    if os.path.exists(output_path):
        print(f"圖像 {idx}/{total} 已存在，跳過: {filename}")
        # 讀取已存在的描述（如果有）
        desc_path = os.path.join(output_dir, f"snoopy_{filename}.txt")
        if os.path.exists(desc_path):
            with open(desc_path, "r", encoding="utf-8") as f:
                description = f.read()
        else:
            description = "已存在的圖像，描述未保存"
        return description, output_path
    
    # 載入並調整輸入圖像大小為512x512（Stable Diffusion的預設尺寸）
    input_image = Image.open(image_path)
    input_image_resized = input_image.resize((512, 512), Image.LANCZOS)
    
    # 生成描述
    description = generate_image_description(image_path)
    
    # 保存描述
    with open(os.path.join(output_dir, f"snoopy_{filename}.txt"), "w", encoding="utf-8") as f:
        f.write(description)
    
    # 創建風格化圖像
    stylized_image = create_stylized_image(image_path, description)
    
    # 調整輸出圖像為224x224（按作業要求）
    stylized_image = stylized_image.resize((224, 224), Image.LANCZOS)

    # 保存結果
    stylized_image.save(output_path)
    
    return description, output_path

def main():
    """主函數"""
    # 獲取輸入圖像列表
    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                  if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # 處理所有圖像，不設限制
    total_images = len(image_files)
    print(f"找到 {total_images} 張圖像需要處理")
    
    results = []
    
    # 創建主進度條
    main_progress = tqdm(total=total_images, desc="總進度", position=0)
    
    # 處理每張圖像
    for idx, image_path in enumerate(image_files):
        try:
            # 更新進度條描述
            main_progress.set_description(f"處理圖像 {idx+1}/{total_images}")
            
            # 處理圖像
            description, output_path = process_image(image_path, idx+1, total_images)
            
            # 更新結果
            results.append({
                "input": image_path,
                "description": description,
                "output": output_path
            })
            
            # 更新進度條
            main_progress.update(1)
            
            # 顯示估計剩餘時間
            if idx > 0:  # 至少處理一張後才能估計
                time_per_image = main_progress.format_dict["elapsed"] / (idx + 1)
                remaining_images = total_images - (idx + 1)
                eta_seconds = time_per_image * remaining_images
                eta_minutes = eta_seconds / 60
                eta_hours = eta_minutes / 60
                
                if eta_hours >= 1:
                    eta_str = f"預計剩餘時間: {eta_hours:.1f} 小時"
                elif eta_minutes >= 1:
                    eta_str = f"預計剩餘時間: {eta_minutes:.1f} 分鐘"
                else:
                    eta_str = f"預計剩餘時間: {eta_seconds:.1f} 秒"
                
                print(f"\r處理進度: {idx+1}/{total_images} ({(idx+1)/total_images*100:.1f}%) - {eta_str}", end="")
            
        except Exception as e:
            print(f"\n處理圖像 {image_path} 時出錯: {e}")
            
    # 關閉進度條
    main_progress.close()
    
    # 保存處理記錄
    with open(os.path.join(output_dir, "processing_log.txt"), "w", encoding="utf-8") as f:
        for idx, result in enumerate(results):
            f.write(f"圖像 {idx+1}:\n")
            f.write(f"  輸入: {result['input']}\n")
            f.write(f"  描述: {result['description']}\n")
            f.write(f"  輸出: {result['output']}\n\n")
    
    print(f"完成! 共處理 {len(results)} 張圖像。結果保存在 {output_dir} 目錄下。")

if __name__ == "__main__":
    main()