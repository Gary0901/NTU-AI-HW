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
result_file = "blip2_mscoco.txt"
f = open(result_file, "w", encoding="utf-8")

def write_log(message):
    """寫入日誌到檔案並同時列印到控制台"""
    print(message)
    f.write(message + "\n")
    f.flush()  # 立即寫入檔案，不等待緩衝區填滿

write_log("開始時間: " + time.strftime("%Y-%m-%d %H:%M:%S"))
write_log("正在載入BLIP模型...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
write_log(f"使用裝置: {device}")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

write_log("正在載入MSCOCO測試集...")
# 選擇MSCOCO測試集
# dataset = load_dataset("nlphuji/mscoco_2014_5k_test_image_text_retrieval")

# 選擇flickr30k 測試集

# 使用全部資料集
test_dataset = dataset["test"]
write_log(f"測試數據集樣本數量: {len(test_dataset)}")

# 生成字幕並評估
generated_captions = []
reference_captions = []

write_log("\n開始處理圖像和生成字幕...")
# 單張處理圖像
for idx, item in enumerate(tqdm(test_dataset)):
    try:
        # 獲取圖像
        image = item["image"]
        
        # 獲取參考字幕
        if "captions" in item:
            refs = item["captions"]
        else:
            # 使用caption欄位
            refs = [item["caption"]]
        
        # 處理圖像用於BLIP模型
        inputs = processor(image, return_tensors="pt").to(device)
        
        # 生成字幕
        with torch.no_grad():
            out = model.generate(**inputs)
        
        # 解碼生成的字幕
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        # 每100個樣本記錄一次進度
        if (idx + 1) % 100 == 0 or idx == 0:
            write_log(f"\n處理進度: {idx+1}/{len(test_dataset)}")
            write_log(f"樣本 {idx+1}:")
            write_log(f"生成的字幕: {caption}")
            write_log(f"參考字幕: {refs}")
        
        # 儲存生成的字幕和參考字幕
        generated_captions.append(caption)
        reference_captions.append(refs)
        
    except Exception as e:
        write_log(f"\n處理樣本 {idx+1} 時出錯: {e}")
        continue

write_log("\n準備評估...")
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

# 計算METEOR分數 - 修正版
def calculate_meteor(references, hypotheses):
    write_log("\n計算METEOR評分...")
    meteor_scores = []
    
    for i in range(len(hypotheses)):
        # 對假設進行分詞
        hyp_tokens = hypotheses[i].split()
        
        # 對參考字幕進行分詞
        refs_tokens = []
        for ref in references[i]:
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
    
    if meteor_scores:
        return np.mean(meteor_scores)
    else:
        return 0.0

# 計算ROUGE分數
def calculate_rouge(references, hypotheses):
    write_log("\n計算ROUGE評分...")
    # 準備數據 - 使用第一個參考
    rouge_refs = []
    for ref_list in references:
        if isinstance(ref_list, list) and ref_list:
            if isinstance(ref_list[0], str):
                rouge_refs.append(ref_list[0])
            elif isinstance(ref_list[0], list) and ref_list[0]:
                rouge_refs.append(ref_list[0][0] if isinstance(ref_list[0][0], str) else "")
            else:
                rouge_refs.append("")
        else:
            rouge_refs.append("")
    
    try:
        # 計算ROUGE分數
        rouge_scores = rouge_evaluator.get_scores(hypotheses, rouge_refs, avg=True)
        rouge_1 = rouge_scores["rouge-1"]["f"]
        rouge_2 = rouge_scores["rouge-2"]["f"]
        return rouge_1, rouge_2
    except Exception as e:
        write_log(f"計算ROUGE時出錯: {e}")
        return 0.0, 0.0

# 計算所有指標
try:
    # 保存生成的字幕和參考字幕到檔案中
    write_log("\n儲存生成的字幕和參考字幕...")
    with open("blip2_generated_captions.txt", "w", encoding="utf-8") as cap_file:
        for idx, (gen, ref) in enumerate(zip(generated_captions, reference_captions)):
            cap_file.write(f"樣本 {idx+1}:\n")
            cap_file.write(f"生成的字幕: {gen}\n")
            cap_file.write(f"參考字幕: {ref}\n\n")
    
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
print(f"結果已保存到 {result_file}")
print(f"生成的字幕詳細資訊已保存到 blip2_generated_captions.txt")