from datasets import load_dataset

# 載入 MSCOCO 測試集
mscoco_dataset = load_dataset("nlphuji/mscoco_2014_5k_test_image_text_retrieval")

# 載入 Flickr30k 
flickr30k_dataset = load_dataset("nlphuji/flickr30k")

print(mscoco_dataset)
print(flickr30k_dataset)