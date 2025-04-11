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

# 測試樣本數
#TEST_SAMPLES = 5  # 只測試5個樣本來檢查功能是否正常

# 開啟結果文件
result_file = "flickr30k_test_results(1000).txt"
f = open(result_file, "w", encoding="utf-8")

def write_log(message):
    """寫入日誌到檔案並同時列印到控制台"""
    print(message)
    f.write(message + "\n")
    f.flush()  # 立即寫入檔案，不等待緩衝區填滿

write_log("開始時間: " + time.strftime("%Y-%m-%d %H:%M:%S"))
write_log("正在測試 Flickr30k 資料集與評估指標...")

try:
    write_log("正在載入BLIP模型...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    write_log(f"使用裝置: {device}")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    write_log("正在載入Flickr30k測試集...")
    # 選擇Flickr30k測試集
    try:
        # 先嘗試直接載入Flickr30k
        dataset = load_dataset("nlphuji/flickr30k")
        if "test" in dataset:
            # 按照助教要求，選取 split 欄位為 "test" 的子集
            flickr30k_dataset = dataset
            test_dataset = flickr30k_dataset["test"]
            
            # 取得 split 為 "test" 的子集
            write_log("正在篩選 split 為 'test' 的子集...")
            flickr30k_subset = test_dataset.filter(lambda x: x["split"] == "test")
            write_log(f"篩選後的子集數量: {len(flickr30k_subset)}")
            
            # 如果子集為空，則使用原測試集
            if len(flickr30k_subset) == 0:
                write_log("篩選後的子集為空，使用原始測試集")
                test_dataset = test_dataset
            else:
                test_dataset = flickr30k_subset
                
        elif "validation" in dataset:
            test_dataset = dataset["validation"]
        else:
            test_dataset = dataset["train"].select(range(1000))  # 如果沒有測試集，使用部分訓練集
    except Exception as e:
        write_log(f"載入nlphuji/flickr30k失敗，嘗試替代資料集: {e}")
        try:
            # 嘗試其他可能的Flickr30k資料集
            dataset = load_dataset("sbu_captions")
            test_dataset = dataset["train"].select(range(1000))  # 使用前1000個樣本作為測試集
        except Exception as e2:
            write_log(f"載入替代資料集也失敗: {e2}")
            write_log("嘗試使用Flickr8k作為備選...")
            dataset = load_dataset("nlphuji/flickr8k")
            test_dataset = dataset["test"]

    # 只使用少量樣本進行測試
    #limited_test_dataset = test_dataset.select(range(min(TEST_SAMPLES, len(test_dataset))))
    limited_test_dataset = test_dataset  # 使用全部子集
    write_log(f"測試數據集樣本數量: {len(limited_test_dataset)}")
    write_log(f"資料集結構: {limited_test_dataset.features}")
    
    # 輸出資料集的第一個樣本，檢查結構
    sample = limited_test_dataset[0]
    write_log(f"第一個樣本的鍵: {list(sample.keys())}")

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
    # 單張處理圖像
    for idx, item in enumerate(tqdm(limited_test_dataset)):
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
            
            # 列印生成的字幕和參考字幕
            write_log(f"\n樣本 {idx+1}:")
            write_log(f"生成的字幕: {caption}")
            write_log(f"參考字幕: {refs}")
            
            # 儲存生成的字幕和參考字幕
            generated_captions.append(caption)
            reference_captions.append(refs)
            
        except Exception as e:
            write_log(f"\n處理樣本 {idx+1} 時出錯: {e}")
            import traceback
            write_log(traceback.format_exc())
            continue

    write_log("\n準備評估...")
    if not generated_captions:
        write_log("沒有成功生成任何字幕，無法進行評估。")
        raise Exception("無字幕生成")
    
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
        
        for i in range(len(hypotheses)):
            try:
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
            except Exception as e:
                write_log(f"計算第 {i+1} 個樣本的METEOR時出錯: {e}")
                continue
        
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

except Exception as main_error:
    write_log(f"主程序執行過程中發生錯誤: {main_error}")
    import traceback
    write_log(traceback.format_exc())

write_log("\n結束時間: " + time.strftime("%Y-%m-%d %H:%M:%S"))
write_log("\n測試完成!")

# 關閉檔案
f.close()
print(f"測試結果已保存到 {result_file}")