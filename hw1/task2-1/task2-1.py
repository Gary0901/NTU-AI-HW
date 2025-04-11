import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from diffusers import StableDiffusion3Pipeline  # 修改這一行
import torch.nn.functional as F

# 設定裝置
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用裝置: {device}")

# 載入Phi-4多模態模型
print("正在載入Phi-4模型...")
model_name = "microsoft/Phi-4-multimodal-instruct"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)  # 添加trust_remote_code=True
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True).to(device)  # 添加trust_remote_code=True

# 載入Stable Diffusion 3 Medium模型
print("正在載入Stable Diffusion模型...")
sd_model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
pipe = StableDiffusion3Pipeline.from_pretrained(sd_model_id, torch_dtype=torch.float16).to(device)  # 使用正確的Pipeline類別

# 設定輸入和輸出路徑
input_folder = "images"  # 存放CeleFaces子集的文件夾
output_folder = "output_stylized"  # 輸出風格化圖像的文件夾

# 創建輸出文件夾（如果不存在）
os.makedirs(output_folder, exist_ok=True)

def generate_description_with_phi4(image_path):
    """使用Phi-4生成圖像描述"""
    # 載入圖像
    image = Image.open(image_path).convert("RGB")
    
    # 定義提示語結構
    user_prompt = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix = '<|end|>'
    
    # 準備提示
    prompt = f'{user_prompt}<|image_1|>Describe this person in detail, focusing on facial features, hair style, and expression. Describe as if they were a character in Peanuts/Snoopy cartoon style.{prompt_suffix}{assistant_prompt}'
    
    # 處理輸入
    inputs = processor(
        text=prompt, 
        images=image, 
        return_tensors="pt"
    ).to(device)
    
    # 生成描述
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_logits_to_keep=1
        )
    
    # 解碼輸出 - 只保留新生成的token
    decode_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    
    # 解碼為文本
    description = processor.batch_decode(
        decode_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]
    
    return description.strip()

def generate_stylized_image(description):
    """使用描述和Stable Diffusion生成風格化圖像"""
    # 增強描述以強調Snoopy/Peanuts風格
    enhanced_prompt = f"A cartoon character in classic Peanuts comic strip style. {description}. Clean, simple line work with minimal detail. Round heads, small bodies with simplified limbs. Use Charles Schulz's distinctive illustration style with flat colors and no shading. Characters should have the cute, innocent quality of Peanuts with slightly oversized heads. Can include recognizable Peanuts elements like Snoopy's doghouse, Charlie Brown's zigzag shirt, or Woodstock the small yellow bird. The artwork should capture the charming, timeless quality of the original Peanuts comics while maintaining their simple, iconic design aesthetic."
    
    # 生成圖像 - 使用Stable Diffusion 3的參數
    image = pipe(
        prompt=enhanced_prompt,
        negative_prompt="realistic, detailed, photograph, 3d, complex background",
        num_inference_steps=28,  # 根據SD3文檔修改
        guidance_scale=7.0,  # 根據SD3文檔修改
        width=512,
        height=512
    ).images[0]
    
    return image

def process_images():
    """處理所有輸入圖像並生成風格化版本"""
    # 獲取所有輸入圖像
    input_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"找到 {len(input_files)} 個輸入圖像")
    
    for i, filename in enumerate(input_files):
        print(f"處理圖像 {i+1}/{len(input_files)}: {filename}")
        
        # 構建路徑
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"stylized_{filename}")
        
        # 生成描述
        print("  生成圖像描述...")
        description = generate_description_with_phi4(input_path)
        print(f"  描述: {description}")
        
        # 生成風格化圖像
        print("  生成風格化圖像...")
        stylized_image = generate_stylized_image(description)
        
        # 保存風格化圖像
        stylized_image.save(output_path)
        print(f"  已保存到 {output_path}")
        
        # 為避免VRAM問題，定期清理緩存
        if device == "cuda" and (i+1) % 10 == 0:
            torch.cuda.empty_cache()

if __name__ == "__main__":
    print("開始處理圖像風格轉換...")
    process_images()
    print("所有圖像處理完成！")