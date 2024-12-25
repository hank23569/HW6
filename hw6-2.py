import torch
from transformers import pipeline
from diffusers import StableDiffusionPipeline

def initialize_models():
    try:
        # 使用新的翻譯模型
        print("正在初始化翻譯模型...")
        translator = pipeline("translation", model="facebook/nllb-200-distilled-600M", src_lang="zho_Hans", tgt_lang="eng_Latn")
        print("翻譯模型初始化成功！")

        # Stable Diffusion 圖片生成模型
        print("正在初始化 Stable Diffusion 模型...")
        sd_pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        sd_pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
        print("Stable Diffusion 模型初始化成功！")
        
        return translator, sd_pipeline
    except Exception as e:
        print(f"模型初始化失敗: {e}")
        return None, None

def translate_and_generate_image(chinese_text, translator, sd_pipeline):
    try:
        # 翻譯文本
        translation_result = translator(chinese_text)
        english_text = translation_result[0]['translation_text']
        print(f"翻譯結果: {english_text}")

        # 使用 Stable Diffusion 生成圖片
        print("正在生成圖片，請稍候...")
        image = sd_pipeline(english_text).images[0]
        image.show()  # 顯示圖片
        return image
    except Exception as e:
        print(f"處理失敗: {e}")
        return None

if __name__ == "__main__":
    translator, sd_pipeline = initialize_models()
    if translator and sd_pipeline:
        # 測試翻譯和圖片生成
        chinese_text = "一個狗和一隻貓"
        translate_and_generate_image(chinese_text, translator, sd_pipeline)
