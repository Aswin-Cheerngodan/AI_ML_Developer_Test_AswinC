from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pathlib import Path

MODEL_PATH = Path("model/distilbert_finetuned_sentiment")
LABELS = ['Negative','Positive']

# Load fine-tuned DistilBERT model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
except Exception as e:
    raise

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def categorizer(review):
    """
    Find the sentiment of the review

    Args:
        review : review from the customer

    Returns:
        sentiment: predicted sentiment of that review
    """
    try:
        # Tokenize input
        inputs = tokenizer(
            review,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().tolist()[0]
            predicted_idx = torch.argmax(logits, dim=1).item()
            sentiment =  LABELS[predicted_idx]

        prob_dict = {label: round(prob, 4) for label, prob in zip(LABELS, probs)}

        return sentiment
    
    except Exception as e:
        print(e)  
        return 


if __name__=="__main__":
    review = "Overall it was a good experience, I Loved it" 
    sentiment = categorizer(review)
    print(sentiment)



    