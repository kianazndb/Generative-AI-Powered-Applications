import torch
from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny.en",
    chunk_length_s=30, 
)

sample = 'downloaded_audio.mp3'
prediction = pipe(sample, batch_size=8)["text"]
print(prediction)
