# RAG Pipeline - Retrieval Augmented Generation

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to generate responses based on context retrieved from uploaded documents.

---

##  Overview

The RAG pipeline combines the power of information retrieval with generative language models to answer user queries based on uploaded `.pdf` or `.txt` documents.

###  Pipeline Flow

1. **File Upload**: Accept `.pdf` or `.txt` files.
2. **Document Loading**: Load and parse text content.
3. **Text Splitting**: Chunk documents using semantic-aware splitting.
4. **Embedding**: Convert chunks into vectors using sentence transformers.
5. **Vector Store**: Store embeddings in a FAISS index.
6. **Query Retrieval**: Find top-k relevant chunks for a user query.
7. **Answer Generation**: Generate final answer using a language model.

---

##  Pipeline Components

###  Document Loading
- **File Types Supported**: `.pdf`, `.txt`
- **Libraries Used**: `PyPDFLoader`, `TextLoader` from `langchain_community`

###  Text Splitting
- **Tool**: `RecursiveCharacterTextSplitter`
- **Config**:
  - `chunk_size = 500`
  - `chunk_overlap = 50`
- **Reason**: Preserves semantic meaning and supports large documents

###  Embedding Model
- **Model**: `sentence-transformers/all-MiniLM-l6-v2`
- **Library**: `HuggingFaceEmbeddings` (LangChain)
- **Device**: CPU
- **Why This Model?**
  - Lightweight & fast
  - Trained for semantic similarity
  - Works well in low-resource environments

###  Vector Store
- **Library**: `FAISS`
- **Purpose**: Efficient similarity search over embedded document chunks

---

##  LLM for Answer Generation

### Model Used
- **Primary**: `openai-community/gpt2` (Text generation)
Smaller model with better generation that distilbert.

### Framework
- **Tool**: `HuggingFacePipeline` from `langchain_huggingface`
- **Parameters**:
  - `temperature = 0.75`
  - `max_new_tokens = 300`

---

##  Chain Configuration

### Used Chain: `RetrievalQA`

### Example Prompt Template

```text
Use the following pieces of context to answer the question at the end. Please follow the following rules:
1. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer but you may want to check the following links".
2. If you find the answer, write the answer in a concise way with five sentences maximum.

{context}

Question: {question}

Helpful Answer:
```

##  Local Setup & Running the Pipeline

Follow these steps to set up the RAG pipeline locally and run it with your own documents.

---

###  Prerequisites

- Python 3.12 
- Virtual Environment (recommended)

---

###  1. Clone the Repository

```bash
git clone https://github.com/Aswin-Cheerngodan/AI_ML_Developer_Test_AswinC.git
cd AI_ML_Developer_Test_AswinC/RAG-System
```

### 2. Create and Activate Virtual Environment
```bash
# Create venv
python -m venv venv

# Activate it
# On Windows
venv\Scripts\activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Add Your Input File
Put your .pdf or .txt file inside the RAG-System/uploads/ directory.

Update the path in the script (rag_pipeline.py):

```bash
ragpipe.file_path = Path('RAG-System/uploads/your_file_name.pdf')
```

 ### 5. Run the Pipeline
```bash
python -m RAG-System.rag_pipeline
```
### 6. Ask Questions from Document
Once running, the script will:

Create the vector DB from your file

Load the language model

Ask a query

Print the final answer based on context from the document

You can modify this part to change the query:

```bash
query = "What is the main objective of the project?"
```
## results
![Alt text](RAG-System/results/Screenshot%20(343).png)
![Alt text](RAG-System/results/Screenshot%20(344).png)
![Alt text](RAG-System/results/Screenshot%20(345).png)
![Alt text](RAG-System/results/Screenshot%20(346).png)
![Alt text](RAG-System/results/Screenshot%20(347).png)
![Alt text](RAG-System/results/Screenshot%20(348).png)
![Alt text](RAG-System/results/Screenshot%20(349).png)

### For Referance Training notebooks added

---
---
---
---


#  Stock Chart Trend Classifier

This FastAPI application classifies stock chart images as either **Uptrend** or **Downtrend** using a CNN model. The model is loaded from a trained `.h5` file. Users can upload stock chart images via a web interface to receive instant trend predictions.

---

##  Features

- Upload stock chart images via a simple web UI
- Preprocessing with OpenCV
- CNN-based classification using TensorFlow
- FastAPI backend with Jinja2 templated frontend
- Dockerized and deployable on Google Cloud Run

---

---

##  Local Setup Instructions (Without Docker)

### 1. Clone the repo

```bash
git clone https://github.com/Aswin-Cheerngodan/AI_ML_Developer_Test_AswinC.git
cd stock_chart
```
### 2. Set up virtual environment Install Python dependencies
```bash
python -m venv venv
venv\Scripts\activate (windows)
pip install -r requirements.txt
```
### 3. Run the FastAPI app locally
```bash
python -m app.main
```   
Open in browser: http://localhost:8000

## Docker Setup (Local)
You can run the app locally using Docker :

### 2. Build the Docker Image
```bash
docker build -t trend-classifier-app .
```
### 3. Run the Container
```bash
docker run -p 8000:8000 trend-classifier-app
```
This maps the container's port 8000 to your local machine's port 8000.

### 4. Access the App in Your Browser
Visit:
```bash
http://localhost:8000
```
You’ll see the UI to upload stock chart images and get predictions (Uptrend / Downtrend).

## Test App

Upload a stock chart image (.png, .jpg, etc.)

Click the "Predict Trend" button.

You'll see a prediction like:
```bash
Prediction: Uptrend
```

#### 3. Test via Postman
Open Postman.

Create a new POST request to:

```bash
http://localhost:8000/predict
```
Under the Body tab:

Select form-data

Add a key: file, type: File

Upload an image file

Click Send.
You will receive an HTML response showing either Uptrend or Downtrend.

## Training and Testing results
![Alt text](stock_chart/results/Screenshot%20(352).png)
![Alt text](stock_chart/results/Screenshot%20(353).png)
![Alt text](stock_chart/results/Screenshot%20(354).png)
![Alt text](stock_chart/results/Screenshot%202025-07-09%20202412.png)


![Alt text](stock_chart/results/Screenshot%202025-07-10%20013244.png)
![Alt text](stock_chart/results/Screenshot%202025-07-10%20013514.png)

## Postman API Testing
![Alt text](stock_chart/results/Screenshot%202025-07-09%20201646.png)

### For Referance Training notebooks added

---
---
---
---

# Yelp Review Sentiment Classification using DistilBERT

This project fine-tunes a pre-trained DistilBERT model for binary sentiment classification (positive or negative) on Yelp customer reviews.

---

## Dataset

A custom Yelp review dataset was used. Each row includes:
- `text`: Customer review content
- `label`: 1 (negative), 2 (positive), remapped to 0 and 1 respectively

---

## Preprocessing Steps

- Loaded the CSV without headers and manually assigned column names.
- Mapped sentiment labels: `1 → 0` (negative), `2 → 1` (positive)
- Removed missing values
- Tokenized using Hugging Face's `AutoTokenizer` with:
  - `max_length=256`
  - `padding='max_length'`
  - `truncation=True`
- Split dataset: 80% training, 20% testing
- Converted to Hugging Face `Dataset` format

---

##  Model Configuration

- **Base Model**: `distilbert-base-uncased`
- **Framework**: PyTorch + Hugging Face Transformers
- **Trainer API** used for efficient training
Why?
DistilBERT is a smaller, faster, and lighter version of BERT that still retains ~97% of BERT's performance

### Training Parameters
| Parameter         | Value     |
|------------------|-----------|
| Epochs           | 1         |
| Learning Rate    | 2e-5      |
| Train Batch Size | 16        |
| Eval Batch Size  | 16        |
| Evaluation       | Per Epoch |

---

## Evaluation Metrics

- **Accuracy**
- **F1-score**

Evaluation
```text
Accuracy: 0.948500
F1 Score: 0.948538
```

# Summary
DistilBERT was fine-tuned for binary sentiment classification on Yelp reviews.

Achieved ~94.9% accuracy on the test set.

Balanced precision and recall, with a good F1-score.

## sample results
## results
![Alt text](sentiment_analysis/results/Screenshot%20(355).png)
![Alt text](sentiment_analysis/results/Screenshot%20(356).png)

## Training notebook added for reference




