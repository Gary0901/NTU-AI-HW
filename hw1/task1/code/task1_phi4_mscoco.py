import os
import json
import torch
import requests
import argparse
import time
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from concurrent.futures import ThreadPoolExecutor, as_completed

# NLTK imports for evaluation metrics
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge

# 下載必要的NLTK資源
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punk_tab', quiet=True)

def load_model(model_path="microsoft/Phi-4-multimodal-instruct"):
    """加載模型和處理器"""
    print(f"Loading model from {model_path}...")
    
    # 加載處理器
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # 加載模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation='flash_attention_2' if torch.cuda.is_available() else 'eager',
    )
    
    # 加載生成設定
    generation_config = GenerationConfig.from_pretrained(model_path)
    
    # 將模型移動到適當的裝置
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    print(f"Model loaded successfully on {device}")
    
    return model, processor, generation_config, device

def process_image(image):
    """處理圖像為適當的格式"""
    try:
        # 檢查圖像類型
        if isinstance(image, str):
            # 如果是路徑或URL
            if image.startswith(('http://', 'https://')):
                # 如果是URL，下載圖像
                image = Image.open(requests.get(image, stream=True).raw).convert("RGB")
            else:
                # 如果是本地路徑
                image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            # 如果已經是PIL圖像對象，確保它是RGB模式
            image = image.convert("RGB")
        else:
            print(f"Unsupported image type: {type(image)}")
            return None
        return image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def save_checkpoint(results, output_file, checkpoint_dir, batch_idx, total_batches, processed_count, total_count):
    """儲存檢查點結果到指定資料夾"""
    # 確保檢查點資料夾存在
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 生成檢查點檔案名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    progress_percent = int((processed_count / total_count) * 100)
    checkpoint_filename = f"checkpoint_batch{batch_idx+1}of{total_batches}_{progress_percent}percent_{timestamp}"
    
    # 決定檔案格式並儲存
    is_json_output = output_file.endswith('.json')
    
    if is_json_output:
        # JSON 格式
        checkpoint_file = os.path.join(checkpoint_dir, f"{checkpoint_filename}.json")
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    else:
        # TXT 格式
        checkpoint_file = os.path.join(checkpoint_dir, f"{checkpoint_filename}.txt")
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            f.write(f"Phi-4 Generated Captions Checkpoint - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Progress: {processed_count}/{total_count} images ({progress_percent}%)\n\n")
            for img_id, cap in results.items():
                f.write(f"Image ID: {img_id}\n")
                f.write(f"Caption: {cap}\n\n")
    
    # 同時更新主要輸出檔案
    if is_json_output:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    else:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Phi-4 Generated Captions - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Progress: {processed_count}/{total_count} images ({progress_percent}%)\n\n")
            for img_id, cap in results.items():
                f.write(f"Image ID: {img_id}\n")
                f.write(f"Caption: {cap}\n\n")
    
    return checkpoint_file

def generate_batch_captions(batch_data, model, processor, generation_config, device, batch_size=8, output_file=None, checkpoint_dir="checkpoints", progress_interval=5):
    """批次生成圖像描述，並定期將檢查點儲存到指定資料夾"""
    results = {}
    total_batches = (len(batch_data) + batch_size - 1) // batch_size
    start_time = time.time()
    last_save_time = start_time
    
    # 定義提示語結構
    user_prompt = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix = '<|end|>'
    
    # 確保檢查點資料夾存在
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}/")
    
    # 決定輸出檔案格式
    is_json_output = output_file.endswith('.json') if output_file else False
    
    # 初始化進度條
    progress_bar = tqdm(total=len(batch_data), desc="Generating captions")
    
    # 把數據分成批次
    for batch_idx in range(0, total_batches):
        batch_start_idx = batch_idx * batch_size
        batch_end_idx = min(batch_start_idx + batch_size, len(batch_data))
        current_batch = batch_data[batch_start_idx:batch_end_idx]
        batch_images = []
        batch_ids = []
        batch_prompts = []
        
        # 處理當前批次的圖像和提示
        for image_id, image in current_batch:
            processed_image = process_image(image)
            if processed_image:
                batch_images.append(processed_image)
                batch_ids.append(image_id)
                prompt = f'{user_prompt}<|image_1|>Generate a detailed caption for this image.{prompt_suffix}{assistant_prompt}'
                batch_prompts.append(prompt)
        
        if not batch_images:
            continue
        
        try:
            # 批次處理輸入
            inputs = processor(text=batch_prompts, images=batch_images, return_tensors='pt', padding=True).to(device)
            
            # 批次生成回應
            with torch.no_grad():
                generate_ids = model.generate(
                    **inputs, 
                    max_new_tokens=20, 
                    generation_config=generation_config,
                    num_logits_to_keep=1
                )
            
            # 解碼輸出
            batch_results = {}
            for j, image_id in enumerate(batch_ids):
                # 獲取這個圖像對應的生成ID
                img_generate_ids = generate_ids[j:j+1]
                # 只保留新生成的token
                decode_ids = img_generate_ids[:, inputs['input_ids'][j:j+1].shape[1]:]
                # 解碼為文本
                caption = processor.batch_decode(
                    decode_ids, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]
                results[image_id] = caption.strip()
                batch_results[image_id] = caption.strip()
                
            # 更新進度條
            progress_bar.update(len(batch_images))
            
            # 計算並顯示進度
            elapsed_time = time.time() - start_time
            processed_count = batch_end_idx
            total_count = len(batch_data)
            images_per_second = processed_count / elapsed_time if elapsed_time > 0 else 0
            remaining_time = (total_count - processed_count) / images_per_second if images_per_second > 0 else 0
            
            # 每隔指定批次顯示進度
            if batch_idx % progress_interval == 0 or batch_idx == total_batches - 1:
                # 格式化剩餘時間
                remaining_hours = int(remaining_time // 3600)
                remaining_minutes = int((remaining_time % 3600) // 60)
                remaining_seconds = int(remaining_time % 60)
                
                progress_percent = (processed_count / total_count) * 100
                print(f"\nProgress: {processed_count}/{total_count} images ({progress_percent:.1f}%)")
                print(f"Images per second: {images_per_second:.2f}")
                print(f"Elapsed time: {elapsed_time:.1f} seconds")
                print(f"Estimated time remaining: {remaining_hours}h {remaining_minutes}m {remaining_seconds}s")
                print(f"Current batch: {batch_idx + 1}/{total_batches}")
                
                # 顯示當前批次的一些生成結果
                if batch_results:
                    sample_id = list(batch_results.keys())[0]
                    print(f"Sample caption (ID {sample_id}): {batch_results[sample_id][:100]}...")
            
            # 定期保存中間結果 (每5分鐘或每10批次)
            current_time = time.time()
            save_checkpoint_now = False
            checkpoint_reason = ""
            
            # 檢查是否需要儲存檢查點
            if current_time - last_save_time > 600:  # 每5分鐘
                save_checkpoint_now = True
                checkpoint_reason = "5-minute interval"
            elif batch_idx % 20 == 0 and batch_idx > 0:  # 每10批次
                save_checkpoint_now = True
                checkpoint_reason = "10-batch interval"
            elif batch_idx == total_batches - 1:  # 最後一個批次
                save_checkpoint_now = True
                checkpoint_reason = "final batch"
            
            if output_file and save_checkpoint_now:
                # 儲存檢查點
                checkpoint_file = save_checkpoint(
                    results, 
                    output_file, 
                    checkpoint_dir, 
                    batch_idx, 
                    total_batches, 
                    processed_count, 
                    total_count
                )
                print(f"Checkpoint saved to {checkpoint_file} ({checkpoint_reason})")
                last_save_time = current_time
                
        except Exception as e:
            print(f"Error in batch processing: {e}")
    
    progress_bar.close()
    return results

def load_mscoco_dataset(dataset_name="nlphuji/mscoco_2014_5k_test_image_text_retrieval", split="test"):
    """從Hugging Face加載MSCOCO完整數據集"""
    try:
        # 加載Hugging Face數據集
        dataset = load_dataset(dataset_name)
        
        # 確定使用哪個分割
        if split in dataset:
            subset = dataset[split]
        elif 'validation' in dataset and split == "test":
            # 如果沒有test集但請求的是test，則使用validation
            subset = dataset['validation']
            print(f"No 'test' split found, using 'validation' instead.")
        elif 'train' in dataset and split == "test":
            # 如果都沒有，使用train
            subset = dataset['train']
            print(f"No 'test' or 'validation' split found, using 'train' instead.")
        else:
            # 使用第一個可用的分割
            subset = dataset[list(dataset.keys())[0]]
            print(f"Using '{list(dataset.keys())[0]}' split.")
        
        print(f"Loaded {len(subset)} images from dataset.")
        
        # 準備圖像和參考標題
        image_data = []
        reference_captions = {}
        
        for i, item in enumerate(subset):
            image_id = str(item.get('image_id', i))
            image = item['image']  # 這是PIL圖像對象
            
            # 保存參考標題
            if 'captions' in item:
                reference_captions[image_id] = item['captions']
            elif 'caption' in item:
                reference_captions[image_id] = [item['caption']]
            
            # 添加到圖像數據列表
            image_data.append((image_id, image))
        
        return image_data, reference_captions
    
    except Exception as e:
        print(f"Error loading dataset from Hugging Face: {e}")
        return [], {}

def evaluate_captions(generated_captions, reference_captions):
    """評估生成的標題"""
    if not generated_captions or not reference_captions:
        return {"BLEU-1": 0, "BLEU-2": 0, "BLEU-3": 0, "BLEU-4": 0, "ROUGE-1": 0, "ROUGE-2": 0, "METEOR": 0}
    
    # 確保reference_captions是正確的結構 - 應該是字符串列表的列表
    # 每一項對應一個圖像的所有參考標題
    processed_references = []
    for refs in reference_captions:
        # 處理refs可能是嵌套列表的情況
        if isinstance(refs, list) and all(isinstance(item, list) for item in refs):
            # 已經是預期格式 - 列表的列表
            processed_references.append(refs)
        elif isinstance(refs, list):
            # 單個參考標題列表
            processed_references.append(refs)
        else:
            # 將單個參考轉換為列表
            processed_references.append([refs])
    
    # 初始化評估指標
    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []
    bleu4_scores = []
    meteor_scores = []
    rouge = Rouge()
    all_rouge_scores = {"rouge-1": {"f": []}, "rouge-2": {"f": []}}
    
    # 處理每個圖像的標題
    for gen_caption, refs_list in zip(generated_captions, processed_references):
        # 如果沒有參考標題則跳過
        if not refs_list:
            continue
        
        # 標記化生成的標題
        gen_tokens = nltk.word_tokenize(gen_caption.lower())
        
        # 處理參考標題 - 確保都是字符串
        refs_tokens = []
        for ref_set in refs_list:
            # 處理參考標題的不同潛在格式
            if isinstance(ref_set, list):
                # 多個參考標題的列表
                for ref in ref_set:
                    if isinstance(ref, str):
                        refs_tokens.append(nltk.word_tokenize(ref.lower()))
            elif isinstance(ref_set, str):
                # 單個參考標題
                refs_tokens.append(nltk.word_tokenize(ref_set.lower()))
        
        # 如果沒有有效的參考標題則跳過
        if not refs_tokens:
            print(f"Warning: No valid reference captions for: {gen_caption[:30]}...")
            continue
        
        # 計算BLEU分數
        smoothing = SmoothingFunction().method1
        try:
            # 計算不同n-gram權重的BLEU分數
            bleu1 = 
            (refs_tokens, gen_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
            bleu2 = sentence_bleu(refs_tokens, gen_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
            bleu3 = sentence_bleu(refs_tokens, gen_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
            bleu4 = sentence_bleu(refs_tokens, gen_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
            
            bleu1_scores.append(bleu1)
            bleu2_scores.append(bleu2)
            bleu3_scores.append(bleu3)
            bleu4_scores.append(bleu4)
        except Exception as e:
            print(f"BLEU calculation error: {e}")
        
        # 計算METEOR分數
        try:
            meteor_values = []
            for ref_tokens in refs_tokens:
                meteor_score_value = meteor_score([ref_tokens], gen_tokens)
                meteor_values.append(meteor_score_value)
            
            max_meteor = max(meteor_values) if meteor_values else 0
            meteor_scores.append(max_meteor)
        except Exception as e:
            print(f"METEOR calculation error: {e}")
        
        # 計算ROUGE分數
        try:
            # 將標記轉換回字符串用於ROUGE計算
            gen_str = " ".join(gen_tokens)
            
            rouge1_values = []
            rouge2_values = []
            
            for ref_tokens in refs_tokens:
                ref_str = " ".join(ref_tokens)
                scores = rouge.get_scores(gen_str, ref_str)[0]
                rouge1_values.append(scores["rouge-1"]["f"])
                rouge2_values.append(scores["rouge-2"]["f"])
            
            max_rouge1 = max(rouge1_values) if rouge1_values else 0
            max_rouge2 = max(rouge2_values) if rouge2_values else 0
            
            all_rouge_scores["rouge-1"]["f"].append(max_rouge1)
            all_rouge_scores["rouge-2"]["f"].append(max_rouge2)
        except Exception as e:
            print(f"ROUGE calculation error: {e}")
    
    # 計算平均分數
    avg_bleu1 = sum(bleu1_scores) / len(bleu1_scores) if bleu1_scores else 0
    avg_bleu2 = sum(bleu2_scores) / len(bleu2_scores) if bleu2_scores else 0
    avg_bleu3 = sum(bleu3_scores) / len(bleu3_scores) if bleu3_scores else 0
    avg_bleu4 = sum(bleu4_scores) / len(bleu4_scores) if bleu4_scores else 0
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0
    avg_rouge1 = sum(all_rouge_scores["rouge-1"]["f"]) / len(all_rouge_scores["rouge-1"]["f"]) if all_rouge_scores["rouge-1"]["f"] else 0
    avg_rouge2 = sum(all_rouge_scores["rouge-2"]["f"]) / len(all_rouge_scores["rouge-2"]["f"]) if all_rouge_scores["rouge-2"]["f"] else 0
    
    return {
        "BLEU-1": avg_bleu1,
        "BLEU-2": avg_bleu2,
        "BLEU-3": avg_bleu3,
        "BLEU-4": avg_bleu4,
        "ROUGE-1": avg_rouge1,
        "ROUGE-2": avg_rouge2,
        "METEOR": avg_meteor
    }

def main():
    parser = argparse.ArgumentParser(description='Phi-4 Image Captioning Test with Batch Processing')
    parser.add_argument('--model_path', type=str, default="microsoft/Phi-4-multimodal-instruct", help='Path to the Phi-4 model')
    parser.add_argument('--dataset_name', type=str, default="nlphuji/mscoco_2014_5k_test_image_text_retrieval", help='Hugging Face dataset name')
    parser.add_argument('--output_file', type=str, default="phi4_mscoco.json", help='Output file for generated captions (.json or .txt)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing')
    parser.add_argument('--split', type=str, default="test", help='Dataset split to use (test, validation, or train)')
    parser.add_argument('--progress_interval', type=int, default=20, help='Interval of batches to show progress')
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints", help='Directory to save checkpoints')
    args = parser.parse_args()
    
    # 檢查並創建檢查點資料夾
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        print(f"Created checkpoint directory: {args.checkpoint_dir}")
    
    # 記錄開始時間
    start_time = time.time()
    print(f"Starting Phi-4 image captioning at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加載模型
    model, processor, generation_config, device = load_model(args.model_path)
    
    # 加載完整數據集
    print(f"Loading dataset {args.dataset_name}...")
    image_data, reference_captions = load_mscoco_dataset(args.dataset_name, args.split)
    
    if not image_data:
        print(f"No images found in dataset {args.dataset_name}")
        return
    
    print(f"Processing {len(image_data)} images from MSCOCO dataset with batch size {args.batch_size}")
    
    # 檢查參考標題的結構
    if reference_captions and list(reference_captions.keys()):
        first_key = list(reference_captions.keys())[0]
        print("\nReference caption structure example:")
        print(f"Key: {first_key}")
        print(f"Type: {type(reference_captions[first_key])}")
        print(f"Content: {reference_captions[first_key]}")
        
        if isinstance(reference_captions[first_key], list) and reference_captions[first_key]:
            print(f"First item type: {type(reference_captions[first_key][0])}")
            print(f"First item content: {reference_captions[first_key][0]}")
    
    # 批次生成標題
    print(f"\nStarting batch processing. Results will be saved to {args.output_file}")
    print(f"Checkpoints will be saved to {args.checkpoint_dir}/ directory")
    print(f"Progress updates will be shown every {args.progress_interval} batches")
    results = generate_batch_captions(
        image_data, 
        model, 
        processor, 
        generation_config, 
        device, 
        args.batch_size,
        args.output_file,
        args.checkpoint_dir,
        args.progress_interval
    )
    
    # 顯示生成的標題數量
    print(f"\nGenerated {len(results)} captions")
    
    # 保存最終結果
    if args.output_file.endswith('.json'):
        # 保存為JSON
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    else:
        # 保存為TXT
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(f"Phi-4 Generated Captions - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for img_id, cap in results.items():
                f.write(f"Image ID: {img_id}\n")
                f.write(f"Caption: {cap}\n\n")
    
    print(f"Final generated captions saved to {args.output_file}")
    
    # 評估生成的標題
    if reference_captions:
        print("\nPreparing for evaluation...")
        # 準備評估
        eval_refs = []
        eval_gens = []
        
        for image_id, caption in results.items():
            if image_id in reference_captions:
                eval_refs.append(reference_captions[image_id])
                eval_gens.append(caption)
        
        print(f"Prepared {len(eval_gens)} caption pairs for evaluation")
        
        # 評估生成的標題
        if eval_refs and eval_gens:
            print("Evaluating captions...")
            metrics = evaluate_captions(eval_gens, eval_refs)
            print("\nEvaluation Results:")
            for metric, score in metrics.items():
                print(f"{metric}: {score:.4f}")
            
            # 保存評估結果
            eval_file = args.output_file.replace('.json', '_evaluation.json').replace('.txt', '_evaluation.json')
            with open(eval_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Evaluation results saved to {eval_file}")
        else:
            print("No matching reference captions found for evaluation")
    else:
        print("No reference captions available for evaluation")
    
    # 計算總執行時間
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    print(f"\nTotal execution time: {hours}h {minutes}m {seconds}s")

if __name__ == "__main__":
    main()