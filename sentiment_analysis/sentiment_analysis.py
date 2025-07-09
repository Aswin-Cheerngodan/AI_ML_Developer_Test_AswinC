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
    reviews = [
    "Absolutely loved the food and the service was excellent!",
    "The food was cold and took forever to arrive.",
    "The ambiance was perfect and the staff were very friendly.",
    "Worst experience ever — completely ruined my evening.",
    "I had a great experience — will definitely come back!",
    "The waiter was rude and inattentive.",
    "The pizza was amazing, fresh, and full of flavor.",
    "Overpriced and underwhelming. Definitely not worth it.",
    "Highly recommend this place for a cozy dinner night.",
    "The place was dirty and smelled bad.",
    "The customer service was top-notch and the coffee was delicious.",
    "I got sick after eating here. Never coming back.",
    "Quick service, clean environment, and very tasty meals.",
    "Terrible customer service and bland food.",
    "I enjoyed every bite! This place never disappoints.",
    "The portions were tiny and the taste was awful.",
    "Five stars for the great food and warm hospitality!",
    "Waited 40 minutes only to get the wrong order.",
    "One of the best dining experiences I’ve had in a long time.",
    "The food looked good but tasted horrible."
] 
    for review in reviews:
        sentiment = categorizer(review)
        print(review," : ",sentiment)



    