import os
import json
import torch
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

# Download necessary NLTK resources
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4', quiet=True)

def load_model(model_path="microsoft/Phi-4-multimodal-instruct"):
    """Load model and processor"""
    print(f"Loading model from {model_path}...")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation='flash_attention_2' if torch.cuda.is_available() else 'eager',
    )
    
    # Load generation config
    generation_config = GenerationConfig.from_pretrained(model_path)
    
    # Move model to appropriate device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    print(f"Model loaded successfully on {device}")
    
    return model, processor, generation_config, device

def process_image(image):
    """Process image to the appropriate format"""
    try:
        # Check image type
        if isinstance(image, str):
            # If it's a path or URL
            if image.startswith(('http://', 'https://')):
                # If it's a URL, download the image
                image = Image.open(requests.get(image, stream=True).raw).convert("RGB")
            else:
                # If it's a local path
                image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            # If it's already a PIL image object, ensure it's in RGB mode
            image = image.convert("RGB")
        else:
            print(f"Unsupported image type: {type(image)}")
            return None
        return image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def save_checkpoint(results, output_file, checkpoint_dir, batch_idx, total_batches, processed_count, total_count):
    """Save checkpoint results to the specified directory"""
    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Generate checkpoint filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    progress_percent = int((processed_count / total_count) * 100)
    checkpoint_filename = f"checkpoint_batch{batch_idx+1}of{total_batches}_{progress_percent}percent_{timestamp}"
    
    # Determine file format and save
    is_json_output = output_file.endswith('.json')
    
    if is_json_output:
        # JSON format
        checkpoint_file = os.path.join(checkpoint_dir, f"{checkpoint_filename}.json")
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    else:
        # TXT format
        checkpoint_file = os.path.join(checkpoint_dir, f"{checkpoint_filename}.txt")
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            f.write(f"Phi-4 Generated Captions Checkpoint - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Progress: {processed_count}/{total_count} images ({progress_percent}%)\n\n")
            for img_id, cap in results.items():
                f.write(f"Image ID: {img_id}\n")
                f.write(f"Caption: {cap}\n\n")
    
    # Also update the main output file
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
    """Generate captions in batches and periodically save checkpoints to the specified directory"""
    results = {}
    total_batches = (len(batch_data) + batch_size - 1) // batch_size
    start_time = time.time()
    last_save_time = start_time
    
    # Define prompt structure
    user_prompt = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix = '<|end|>'
    
    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}/")
    
    # Determine output file format
    is_json_output = output_file.endswith('.json') if output_file else False
    
    # Initialize progress bar
    progress_bar = tqdm(total=len(batch_data), desc="Generating captions")
    
    # Split data into batches
    for batch_idx in range(0, total_batches):
        batch_start_idx = batch_idx * batch_size
        batch_end_idx = min(batch_start_idx + batch_size, len(batch_data))
        current_batch = batch_data[batch_start_idx:batch_end_idx]
        batch_images = []
        batch_ids = []
        batch_prompts = []
        
        # Process current batch of images and prompts
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
            # Batch process inputs
            inputs = processor(text=batch_prompts, images=batch_images, return_tensors='pt', padding=True).to(device)
            
            # Batch generate responses
            with torch.no_grad():
                generate_ids = model.generate(
                    **inputs, 
                    max_new_tokens=20, 
                    generation_config=generation_config,
                    num_logits_to_keep=1
                )
            
            # Decode outputs
            batch_results = {}
            for j, image_id in enumerate(batch_ids):
                # Get the generation IDs for this image
                img_generate_ids = generate_ids[j:j+1]
                # Only keep the newly generated tokens
                decode_ids = img_generate_ids[:, inputs['input_ids'][j:j+1].shape[1]:]
                # Decode to text
                caption = processor.batch_decode(
                    decode_ids, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]
                results[image_id] = caption.strip()
                batch_results[image_id] = caption.strip()
                
            # Update progress bar
            progress_bar.update(len(batch_images))
            
            # Calculate and display progress
            elapsed_time = time.time() - start_time
            processed_count = batch_end_idx
            total_count = len(batch_data)
            images_per_second = processed_count / elapsed_time if elapsed_time > 0 else 0
            remaining_time = (total_count - processed_count) / images_per_second if images_per_second > 0 else 0
            
            # Display progress at specified intervals
            if batch_idx % progress_interval == 0 or batch_idx == total_batches - 1:
                # Format remaining time
                remaining_hours = int(remaining_time // 3600)
                remaining_minutes = int((remaining_time % 3600) // 60)
                remaining_seconds = int(remaining_time % 60)
                
                progress_percent = (processed_count / total_count) * 100
                print(f"\nProgress: {processed_count}/{total_count} images ({progress_percent:.1f}%)")
                print(f"Images per second: {images_per_second:.2f}")
                print(f"Elapsed time: {elapsed_time:.1f} seconds")
                print(f"Estimated time remaining: {remaining_hours}h {remaining_minutes}m {remaining_seconds}s")
                print(f"Current batch: {batch_idx + 1}/{total_batches}")
                
                # Display some generated results from current batch
                if batch_results:
                    sample_id = list(batch_results.keys())[0]
                    print(f"Sample caption (ID {sample_id}): {batch_results[sample_id][:100]}...")
            
            # Periodically save intermediate results (every 10 minutes or every 20 batches)
            current_time = time.time()
            save_checkpoint_now = False
            checkpoint_reason = ""
            
            # Check if checkpoint needs to be saved
            if current_time - last_save_time > 600:  # Every 10 minutes
                save_checkpoint_now = True
                checkpoint_reason = "10-minute interval"
            elif batch_idx % 20 == 0 and batch_idx > 0:  # Every 20 batches
                save_checkpoint_now = True
                checkpoint_reason = "20-batch interval"
            elif batch_idx == total_batches - 1:  # Last batch
                save_checkpoint_now = True
                checkpoint_reason = "final batch"
            
            if output_file and save_checkpoint_now:
                # Save checkpoint
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

def load_flickr30k_dataset(dataset_name="nlphuji/flickr30k", split="test"):
    """Load Flickr30k dataset from Hugging Face"""
    try:
        # Load Hugging Face dataset
        dataset = load_dataset(dataset_name)
        
        # Determine which split to use
        if split in dataset:
            subset = dataset[split]
        elif 'validation' in dataset and split == "test":
            # If there's no test set but test is requested, use validation
            subset = dataset['validation']
            print(f"No 'test' split found, using 'validation' instead.")
        elif 'train' in dataset and split == "test":
            # If neither is available, use train
            subset = dataset['train']
            print(f"No 'test' or 'validation' split found, using 'train' instead.")
        else:
            # Use the first available split
            subset = dataset[list(dataset.keys())[0]]
            print(f"Using '{list(dataset.keys())[0]}' split.")
        
        print(f"Loaded {len(subset)} images from dataset.")
        
        # Prepare images and reference captions
        image_data = []
        reference_captions = {}
        
        for i, item in enumerate(subset):
            # In Flickr30k, images are in the 'image' field and captions in the 'caption' field
            image_id = str(item.get('image_id', i))
            #image_id = str(i)
            image = item['image']  # This is a PIL image object
            
            # Save reference captions
            if 'captions' in item:
                reference_captions[image_id] = item['captions']
            elif 'caption' in item:
                reference_captions[image_id] = [item['caption']]
            
            # Add to image data list
            image_data.append((image_id, image))
        
        return image_data, reference_captions
    
    except Exception as e:
        print(f"Error loading dataset from Hugging Face: {e}")
        return [], {}

def evaluate_captions_optimized(generated_captions, reference_captions, show_examples=3, verbose=False):
    """
    评估生成的标题，减少输出的详细信息
    
    参数:
    - generated_captions: 生成的标题字典 {image_id: caption}
    - reference_captions: 参考标题字典 {image_id: [captions]}
    - show_examples: 显示的样例数量
    - verbose: 是否显示详细的处理信息
    """
    print(f"评估 {len(generated_captions)} 个生成的标题，参考标题集合大小: {len(reference_captions)}")
    
    matched_ids = set(generated_captions.keys()) & set(reference_captions.keys())
    print(f"匹配的ID数量: {len(matched_ids)}")
    
    if not matched_ids:
        print("没有匹配的ID，无法进行评估")
        return {"BLEU-1": 0, "BLEU-2": 0, "BLEU-3": 0, "BLEU-4": 0, "ROUGE-1": 0, "ROUGE-2": 0, "METEOR": 0}
    
    # 准备评估数据
    processed_references = []
    generated_list = []
    image_ids = []
    
    # 处理进度计数
    processed_count = 0
    total_count = len(matched_ids)
    
    # 处理参考标题
    for img_id in matched_ids:
        caption = generated_captions[img_id]
        refs = reference_captions[img_id]
        
        # 更新进度（每处理100个显示一次）
        processed_count += 1
        if verbose or processed_count % 100 == 0:
            print(f"处理进度: {processed_count}/{total_count} ({processed_count/total_count*100:.1f}%)")
        
        # 修复处理嵌套列表的问题
        if isinstance(refs, list):
            if len(refs) > 0 and isinstance(refs[0], list):
                # 已经是嵌套列表，直接使用第一个内部列表
                processed_references.append(refs[0])
            else:
                # 单层列表
                processed_references.append(refs)
        else:
            # 不是列表，转换为单项列表
            processed_references.append([refs])
            
        generated_list.append(caption)
        image_ids.append(img_id)
    
    # 初始化评估指标
    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []
    bleu4_scores = []
    meteor_scores = []
    rouge = Rouge()
    all_rouge_scores = {"rouge-1": {"f": []}, "rouge-2": {"f": []}}
    
    # 存储比较样例
    comparison_examples = []
    
    # 处理每个图像的标题
    print("开始计算评估指标...")
    for i, (gen_caption, refs_list, img_id) in enumerate(zip(generated_list, processed_references, image_ids)):
        # 显示进度（每计算1000个显示一次）
        if (i+1) % 1000 == 0:
            print(f"评估进度: {i+1}/{len(generated_list)} ({(i+1)/len(generated_list)*100:.1f}%)")
        
        # 跳过没有参考标题的情况
        if not refs_list:
            continue
        
        # 标记化生成的标题
        gen_tokens = nltk.word_tokenize(gen_caption.lower())
        
        # 处理参考标题 - 确保它们都是字符串
        refs_tokens = []
        refs_str = []
        for ref in refs_list:
            if isinstance(ref, str):
                refs_tokens.append(nltk.word_tokenize(ref.lower()))
                refs_str.append(ref)
        
        # 跳过没有有效参考标题的情况
        if not refs_tokens:
            continue
        
        # 用于详细比较的参考标题（第一个）
        ref_tokens = refs_tokens[0]
        ref_str = refs_str[0] if refs_str else ""
        
        # 计算BLEU分数
        smoothing = SmoothingFunction().method1
        try:
            # 使用不同的n-gram权重计算BLEU分数
            bleu1 = sentence_bleu(refs_tokens, gen_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
            bleu2 = sentence_bleu(refs_tokens, gen_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
            bleu3 = sentence_bleu(refs_tokens, gen_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
            bleu4 = sentence_bleu(refs_tokens, gen_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
            
            bleu1_scores.append(bleu1)
            bleu2_scores.append(bleu2)
            bleu3_scores.append(bleu3)
            bleu4_scores.append(bleu4)
        except Exception:
            # 出错时不输出详细错误信息，只是跳过
            continue
        
        # 计算METEOR分数
        meteor_value = 0
        try:
            meteor_values = []
            for ref_tokens in refs_tokens:
                meteor_score_value = meteor_score([ref_tokens], gen_tokens)
                meteor_values.append(meteor_score_value)
            
            meteor_value = max(meteor_values) if meteor_values else 0
            meteor_scores.append(meteor_value)
        except Exception:
            # 出错时不输出详细错误信息
            pass
        
        # 计算ROUGE分数
        try:
            # 将标记转换回字符串以进行ROUGE计算
            gen_str = " ".join(gen_tokens)
            
            rouge1_values = []
            rouge2_values = []
            
            for ref in refs_str:
                scores = rouge.get_scores(gen_str, ref)[0]
                rouge1_values.append(scores["rouge-1"]["f"])
                rouge2_values.append(scores["rouge-2"]["f"])
            
            rouge1_value = max(rouge1_values) if rouge1_values else 0
            rouge2_value = max(rouge2_values) if rouge2_values else 0
            
            all_rouge_scores["rouge-1"]["f"].append(rouge1_value)
            all_rouge_scores["rouge-2"]["f"].append(rouge2_value)
        except Exception:
            # 出错时不输出详细错误信息
            pass
        
        # 收集比较样例（只保留少数几个）
        if len(comparison_examples) < show_examples:
            # 寻找匹配和不匹配的标记
            matching_tokens = set(gen_tokens) & set(ref_tokens)
            gen_only_tokens = set(gen_tokens) - set(ref_tokens)
            ref_only_tokens = set(ref_tokens) - set(gen_tokens)
            
            # 收集样例
            comparison_examples.append({
                "image_id": img_id,
                "generated": gen_caption,
                "reference": ref_str,
                "matching_tokens": list(matching_tokens),
                "metrics": {
                    "BLEU-1": bleu1,
                    "BLEU-2": bleu2,
                    "BLEU-3": bleu3,
                    "BLEU-4": bleu4,
                    "METEOR": meteor_value,
                    "ROUGE-1": rouge1_value if 'rouge1_value' in locals() else 0,
                    "ROUGE-2": rouge2_value if 'rouge2_value' in locals() else 0
                }
            })
    
    # 计算平均分数
    print("计算平均评估指标...")
    avg_bleu1 = sum(bleu1_scores) / len(bleu1_scores) if bleu1_scores else 0
    avg_bleu2 = sum(bleu2_scores) / len(bleu2_scores) if bleu2_scores else 0
    avg_bleu3 = sum(bleu3_scores) / len(bleu3_scores) if bleu3_scores else 0
    avg_bleu4 = sum(bleu4_scores) / len(bleu4_scores) if bleu4_scores else 0
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0
    avg_rouge1 = sum(all_rouge_scores["rouge-1"]["f"]) / len(all_rouge_scores["rouge-1"]["f"]) if all_rouge_scores["rouge-1"]["f"] else 0
    avg_rouge2 = sum(all_rouge_scores["rouge-2"]["f"]) / len(all_rouge_scores["rouge-2"]["f"]) if all_rouge_scores["rouge-2"]["f"] else 0
    
    # 打印比较样例
    print("\n===== 标题比较样例 =====")
    for i, example in enumerate(comparison_examples):
        print(f"\n样例 {i+1} (图像ID: {example['image_id']}):")
        print(f"生成的标题: {example['generated']}")
        print(f"参考标题: {example['reference']}")
        print(f"匹配的词汇: {example['matching_tokens']}")
        print("评估指标:")
        for metric, value in example['metrics'].items():
            print(f"  {metric}: {value:.4f}")
        print("-" * 40)
    
    # 准备返回的指标字典
    metrics = {
        "BLEU-1": avg_bleu1,
        "BLEU-2": avg_bleu2,
        "BLEU-3": avg_bleu3,
        "BLEU-4": avg_bleu4,
        "ROUGE-1": avg_rouge1,
        "ROUGE-2": avg_rouge2,
        "METEOR": avg_meteor,
        "Processed": len(bleu1_scores)  # 记录成功处理的数量
    }
    
    print(f"\n成功评估 {len(bleu1_scores)} 个标题")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Phi-4 Image Captioning on Flickr30k Dataset')
    parser.add_argument('--model_path', type=str, default="microsoft/Phi-4-multimodal-instruct", help='Path to the Phi-4 model')
    parser.add_argument('--dataset_name', type=str, default="nlphuji/flickr30k", help='Hugging Face dataset name')
    parser.add_argument('--output_file', type=str, default="phi4_flickr30k.json", help='Output file for generated captions (.json or .txt)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing')
    parser.add_argument('--split', type=str, default="test", help='Dataset split to use (test, validation, or train)')
    parser.add_argument('--progress_interval', type=int, default=20, help='Interval of batches to show progress')
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints", help='Directory to save checkpoints')
    parser.add_argument('--subset_size', type=int, default=None, help='Number of images to process (default: entire dataset)')
    args = parser.parse_args()
    
    # Check and create checkpoint directory
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        print(f"Created checkpoint directory: {args.checkpoint_dir}")
    
    # Record start time
    start_time = time.time()
    print(f"Starting Phi-4 image captioning on Flickr30k at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load model
    model, processor, generation_config, device = load_model(args.model_path)
    
    # Load Flickr30k dataset
    print(f"Loading dataset {args.dataset_name}...")
    image_data, reference_captions = load_flickr30k_dataset(args.dataset_name, args.split)
    
    if not image_data:
        print(f"No images found in dataset {args.dataset_name}")
        return
    
    # If subset_size is specified, use only that many images
    if args.subset_size and args.subset_size < len(image_data):
        print(f"Using subset of {args.subset_size} images from the dataset")
        image_data = image_data[:args.subset_size]
    
    print(f"Processing {len(image_data)} images from Flickr30k dataset with batch size {args.batch_size}")
    
    # Check reference caption structure
    if reference_captions and list(reference_captions.keys()):
        first_key = list(reference_captions.keys())[0]
        print("\nReference caption structure example:")
        print(f"Key: {first_key}")
        print(f"Type: {type(reference_captions[first_key])}")
        print(f"Content: {reference_captions[first_key]}")
        
        if isinstance(reference_captions[first_key], list) and reference_captions[first_key]:
            print(f"First item type: {type(reference_captions[first_key][0])}")
            print(f"First item content: {reference_captions[first_key][0]}")
    
    # Generate captions in batches
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
    
    # Display number of generated captions
    print(f"\nGenerated {len(results)} captions")
    
    # Save final results
    if args.output_file.endswith('.json'):
        # Save as JSON
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    else:
        # Save as TXT
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(f"Phi-4 Generated Captions - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for img_id, cap in results.items():
                f.write(f"Image ID: {img_id}\n")
                f.write(f"Caption: {cap}\n\n")
    
    print(f"Final generated captions saved to {args.output_file}")
    
    # Evaluate generated captions
    if reference_captions:
        print("\nEvaluating captions...")
        metrics = evaluate_captions_optimized(results, reference_captions)
        print("\nEvaluation Results:")
        for metric, score in metrics.items():
            print(f"{metric}: {score:.4f}")
        
        # Save evaluation results
        eval_file = args.output_file.replace('.json', '_evaluation.json').replace('.txt', '_evaluation.json')
        with open(eval_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Evaluation results saved to {eval_file}")
    else:
        print("No reference captions available for evaluation")
    
    # Calculate total execution time
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    print(f"\nTotal execution time: {hours}h {minutes}m {seconds}s")

if __name__ == "__main__":
    main()