import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from datasets import load_dataset
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import numpy as np
from tqdm import tqdm
import time
import os

# 下載必要的NLTK數據
print("下載必要的NLTK數據...")
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4', quiet=True)  # 確保METEOR正常工作

# 開啟結果文件
result_file = "flickr30k_full_results.txt"
f = open(result_file, "w", encoding="utf-8")

def write_log(message):
    """寫入日誌到檔案並同時列印到控制台"""
    print(message)
    f.write(message + "\n")
    f.flush()  # 立即寫入檔案，不等待緩衝區填滿

# 設定進度報告的頻率
REPORT_FREQUENCY = 300  # 每處理300個樣本報告一次進度

write_log("開始時間: " + time.strftime("%Y-%m-%d %H:%M:%S"))
write_log("正在載入BLIP模型...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
write_log(f"使用裝置: {device}")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

write_log("正在載入完整的Flickr30k資料集...")
try:
    # 載入完整的Flickr30k資料集
    dataset = load_dataset("nlphuji/flickr30k")
    
    # 使用所有可用的資料
    write_log("獲取所有可用的資料...")
    
    # 合併所有的split (train, validation, test)
    all_data = []
    
    if "train" in dataset:
        write_log(f"訓練集大小: {len(dataset['train'])}")
        all_data.append(dataset["train"])
    
    if "validation" in dataset:
        write_log(f"驗證集大小: {len(dataset['validation'])}")
        all_data.append(dataset["validation"])
        
    if "test" in dataset:
        write_log(f"測試集大小: {len(dataset['test'])}")
        all_data.append(dataset["test"])
    
    # 檢查是否有任何分割
    if not all_data:
        write_log("未找到任何分割，使用原始資料集")
        # 使用可能已經合併的資料集
        if hasattr(dataset, "__len__"):
            write_log(f"資料集大小: {len(dataset)}")
            full_dataset = dataset
        else:
            # 如果dataset是DatasetDict而不是Dataset
            for key in dataset.keys():
                write_log(f"使用 '{key}' 作為資料集，大小: {len(dataset[key])}")
                full_dataset = dataset[key]
                break
    else:
        from datasets import concatenate_datasets
        full_dataset = concatenate_datasets(all_data)
        write_log(f"合併後的資料集大小: {len(full_dataset)}")
    
except Exception as e:
    write_log(f"載入Flickr30k失敗: {e}")
    import traceback
    write_log(traceback.format_exc())
    write_log("嘗試載入替代資料集...")
    try:
        # 嘗試替代資料集
        dataset = load_dataset("sbu_captions")
        full_dataset = dataset["train"]
        write_log(f"使用替代資料集，大小: {len(full_dataset)}")
    except Exception as e2:
        write_log(f"載入替代資料集也失敗: {e2}")
        raise Exception("無法載入任何可用的資料集")

# 輸出資料集的結構資訊
write_log(f"資料集特徵: {full_dataset.features}")
sample = full_dataset[0]
write_log(f"樣本鍵: {list(sample.keys())}")

# 識別圖像和文字的欄位名稱
image_field = None
text_field = None

possible_image_fields = ["image", "img", "images", "picture", "photo"]
possible_text_fields = ["caption", "captions", "text", "texts", "description", "descriptions"]

for field in possible_image_fields:
    if field in sample:
        image_field = field
        break

for field in possible_text_fields:
    if field in sample:
        text_field = field
        break

if not image_field:
    write_log("無法找到圖像欄位，嘗試假設image是圖像欄位")
    image_field = "image"

if not text_field:
    write_log("無法找到文字欄位，嘗試假設caption是文字欄位")
    text_field = "caption"

write_log(f"使用圖像欄位: {image_field}")
write_log(f"使用文字欄位: {text_field}")

# 生成字幕並評估
generated_captions = []
reference_captions = []

write_log("\n開始處理圖像和生成字幕...")
start_process_time = time.time()

# 單張處理圖像
for idx, item in enumerate(tqdm(full_dataset)):
    try:
        # 獲取圖像
        image = item[image_field]
        
        # 獲取參考字幕
        if isinstance(item.get(text_field), list):
            refs = item[text_field]
        else:
            # 使用單一caption
            refs = [item[text_field]]
        
        # 處理圖像用於BLIP模型
        inputs = processor(image, return_tensors="pt").to(device)
        
        # 生成字幕
        with torch.no_grad():
            out = model.generate(**inputs)
        
        # 解碼生成的字幕
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        # 儲存生成的字幕和參考字幕
        generated_captions.append(caption)
        reference_captions.append(refs)
        
        # 根據頻率報告進度
        if (idx + 1) % REPORT_FREQUENCY == 0 or idx == 0:
            elapsed_time = time.time() - start_process_time
            avg_time_per_sample = elapsed_time / (idx + 1)
            remaining_samples = len(full_dataset) - (idx + 1)
            estimated_time_remaining = remaining_samples * avg_time_per_sample
            
            hours, remainder = divmod(estimated_time_remaining, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            write_log(f"\n進度: {idx+1}/{len(full_dataset)} ({(idx+1)/len(full_dataset)*100:.2f}%)")
            write_log(f"預估剩餘時間: {int(hours)}小時 {int(minutes)}分鐘 {int(seconds)}秒")
            write_log(f"最新生成的字幕: {caption}")
            write_log(f"參考字幕: {refs}")
            
            # 每300筆存檔一次中間結果，以防程式中斷
            if len(generated_captions) > 0:
                temp_results = f"temp_results_{idx+1}.txt"
                with open(temp_results, "w", encoding="utf-8") as temp_file:
                    temp_file.write(f"處理的樣本數量: {len(generated_captions)}\n")
                    for i in range(min(5, len(generated_captions))):
                        temp_file.write(f"樣本 {i+1}:\n")
                        temp_file.write(f"生成的字幕: {generated_captions[i]}\n")
                        temp_file.write(f"參考字幕: {reference_captions[i]}\n\n")
                write_log(f"已儲存中間結果到 {temp_results}")
        
    except Exception as e:
        write_log(f"\n處理樣本 {idx+1} 時出錯: {e}")
        continue

total_process_time = time.time() - start_process_time
hours, remainder = divmod(total_process_time, 3600)
minutes, seconds = divmod(remainder, 60)
write_log(f"\n處理完成! 總共處理 {len(generated_captions)}/{len(full_dataset)} 個樣本")
write_log(f"總處理時間: {int(hours)}小時 {int(minutes)}分鐘 {int(seconds)}秒")

# 儲存所有生成的字幕
captions_file = "flickr30k_all_captions.txt"
write_log(f"\n正在儲存所有生成的字幕到 {captions_file}...")
with open(captions_file, "w", encoding="utf-8") as cap_file:
    for i, (gen, ref) in enumerate(zip(generated_captions, reference_captions)):
        cap_file.write(f"樣本 {i+1}:\n")
        cap_file.write(f"生成的字幕: {gen}\n")
        cap_file.write(f"參考字幕: {ref}\n\n")

write_log("\n準備評估...")
if not generated_captions:
    write_log("沒有成功生成任何字幕，無法進行評估。")
else:
    # 準備數據用於評估
    flat_references = []
    for refs in reference_captions:
        if isinstance(refs, list):
            flat_references.append(refs)  # 保留整個列表
        else:
            flat_references.append([refs])  # 將單個字串包裝成列表

    # 建立平滑函數和Rouge評估器
    smoothing = SmoothingFunction().method1
    rouge_evaluator = Rouge()

    # 計算BLEU分數
    def calculate_bleu(references, hypotheses):
        write_log("\n計算BLEU評分...")
        
        # 準備用於BLEU評分的格式
        references_for_bleu = []
        for ref in references:
            tokenized_refs = []
            
            if isinstance(ref, list):
                for r in ref:
                    if isinstance(r, str):
                        tokenized_refs.append(r.split())
                    elif isinstance(r, list):
                        for subr in r:
                            if isinstance(subr, str):
                                tokenized_refs.append(subr.split())
            else:
                tokenized_refs.append(ref.split())
            
            if tokenized_refs:
                references_for_bleu.append(tokenized_refs)
            else:
                references_for_bleu.append([[]]) 
        
        # 對生成的字幕進行分詞
        hypotheses_tokenized = [hyp.split() for hyp in hypotheses]
        
        # 計算BLEU-1分數
        weights_1gram = (1.0, 0, 0, 0)
        bleu_1 = corpus_bleu(
            references_for_bleu, 
            hypotheses_tokenized, 
            weights=weights_1gram,
            smoothing_function=smoothing
        )
        
        # 計算BLEU-4分數
        weights_4gram = (0.25, 0.25, 0.25, 0.25)
        bleu_4 = corpus_bleu(
            references_for_bleu, 
            hypotheses_tokenized, 
            weights=weights_4gram,
            smoothing_function=smoothing
        )
        
        return bleu_1, bleu_4

    # 計算METEOR分數
    def calculate_meteor(references, hypotheses):
        write_log("\n計算METEOR評分...")
        meteor_scores = []
        
        # 為了處理大型資料集，分批計算METEOR
        batch_size = 1000
        for i in range(0, len(hypotheses), batch_size):
            batch_end = min(i + batch_size, len(hypotheses))
            write_log(f"處理METEOR批次 {i//batch_size + 1}/{(len(hypotheses) + batch_size - 1)//batch_size}")
            
            for j in range(i, batch_end):
                try:
                    # 對假設進行分詞
                    hyp_tokens = hypotheses[j].split()
                    
                    # 對參考字幕進行分詞
                    refs_tokens = []
                    for ref in references[j]:
                        if isinstance(ref, str):
                            refs_tokens.append(ref.split())
                        elif isinstance(ref, list):
                            for subr in ref:
                                if isinstance(subr, str):
                                    refs_tokens.append(subr.split())
                    
                    if refs_tokens:
                        # 使用分詞後的文本計算METEOR
                        m_score = meteor_score(refs_tokens, hyp_tokens)
                        meteor_scores.append(m_score)
                except Exception as e:
                    write_log(f"計算第 {j+1} 個樣本的METEOR時出錯: {e}")
                    continue
        
        if meteor_scores:
            return np.mean(meteor_scores)
        else:
            return 0.0

    # 計算ROUGE分數
    def calculate_rouge(references, hypotheses):
        write_log("\n計算ROUGE評分...")
        
        # 為了處理大型資料集，分批計算ROUGE
        batch_size = 1000
        rouge_1_scores = []
        rouge_2_scores = []
        
        for i in range(0, len(hypotheses), batch_size):
            batch_end = min(i + batch_size, len(hypotheses))
            write_log(f"處理ROUGE批次 {i//batch_size + 1}/{(len(hypotheses) + batch_size - 1)//batch_size}")
            
            # 準備這個批次的數據
            batch_hyps = hypotheses[i:batch_end]
            batch_refs = []
            
            for j in range(i, batch_end):
                ref_list = references[j]
                if isinstance(ref_list, list) and ref_list:
                    if isinstance(ref_list[0], str):
                        batch_refs.append(ref_list[0])
                    elif isinstance(ref_list[0], list) and ref_list[0]:
                        batch_refs.append(ref_list[0][0] if isinstance(ref_list[0][0], str) else "")
                    else:
                        batch_refs.append("")
                else:
                    batch_refs.append("")
            
            try:
                # 計算這個批次的ROUGE分數
                rouge_scores = rouge_evaluator.get_scores(batch_hyps, batch_refs, avg=True)
                rouge_1_scores.append(rouge_scores["rouge-1"]["f"])
                rouge_2_scores.append(rouge_scores["rouge-2"]["f"])
            except Exception as e:
                write_log(f"計算ROUGE批次 {i//batch_size + 1} 時出錯: {e}")
                continue
        
        # 計算平均分數
        if rouge_1_scores and rouge_2_scores:
            avg_rouge_1 = np.mean(rouge_1_scores)
            avg_rouge_2 = np.mean(rouge_2_scores)
            return avg_rouge_1, avg_rouge_2
        else:
            return 0.0, 0.0

    # 計算所有指標
    try:
        # BLEU
        bleu_1, bleu_4 = calculate_bleu(flat_references, generated_captions)
        
        # METEOR
        meteor = calculate_meteor(flat_references, generated_captions)
        
        # ROUGE
        rouge_1, rouge_2 = calculate_rouge(flat_references, generated_captions)
        
        # 印出結果
        write_log("\n評估結果:")
        write_log(f"處理的樣本數量: {len(generated_captions)}")
        write_log(f"BLEU-1 分數: {bleu_1:.4f}")
        write_log(f"BLEU-4 分數: {bleu_4:.4f}")
        write_log(f"METEOR 分數: {meteor:.4f}")
        write_log(f"ROUGE-1 分數: {rouge_1:.4f}")
        write_log(f"ROUGE-2 分數: {rouge_2:.4f}")
        
    except Exception as e:
        write_log(f"\n評估過程中出錯: {e}")
        import traceback
        error_trace = traceback.format_exc()  # 獲取詳細的錯誤追蹤信息
        write_log(error_trace)

write_log("\n結束時間: " + time.strftime("%Y-%m-%d %H:%M:%S"))
write_log("\n測試完成!")

# 關閉檔案
f.close()
print(f"測試結果已保存到 {result_file}")
print(f"所有生成的字幕已保存到 {captions_file}")