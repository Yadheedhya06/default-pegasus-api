# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("BrainStormersHakton/question-gen-T5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("BrainStormersHakton/question-gen-T5-base")

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    pipeline(model="BrainStormersHakton/question-gen-T5-base")

if __name__ == "__main__":
    download_model()
