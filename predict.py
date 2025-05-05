import torch
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import os

# Set directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "vilt-daquar-epoch_10_acc_0.4056")
ANSWER_PATH = os.path.join(BASE_DIR, "data", "answer_space.txt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load processor and model
processor = ViltProcessor.from_pretrained(MODEL_DIR)
model = ViltForQuestionAnswering.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

# Load answer space
with open(ANSWER_PATH, "r") as f:
    label2answer = [line.strip() for line in f.readlines()]

@torch.no_grad()
def predict_answer(image: Image.Image, question: str) -> str:
    # Preprocess
    encoding = processor(
        images=image,
        text=question,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=40,
    ).to(device)

    # Forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()
    predicted_answer = label2answer[predicted_label]

    return predicted_answer
