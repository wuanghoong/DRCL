import numpy as np
import evaluate
import os

os.environ["http_proxy"] = "http://127.0.0.1:8889"
os.environ["https_proxy"] = "http://127.0.0.1:8889"

import requests
r = requests.get("https://huggingface.co")
print(r.status_code)  # 返回 200 表示连接成功
# Load the "accuracy" module from the evaluate library.
accuracy = evaluate.load("accuracy")

# Create a preprocessing function to encode text and truncate strings longer than the maximum input token length.
def preprocess_function(tokenizer, examples):
    samples = tokenizer(examples["text"], truncation=True)
    samples.pop('attention_mask')
    return samples

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Get the index of the class with the highest probability in predictions.
    predictions = np.argmax(predictions, axis=1)
    
    # Use the "accuracy" module to compute accuracy based on predictions and labels.
    # return accuracy.compute(predictions=predictions, references=labels)
    return {"eval_accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"]}

import evaluation
from sklearn.metrics import f1_score

def result(pred, labels):
    nmi = evaluation.NMI_helper(pred, labels)
    ac = evaluation.matched_ac(pred, labels)
    # f1 = evaluation.cal_F_score(pred, labels)[0]
    f1 = f1_score(labels, pred, average='macro')
    f1_micro = f1_score(labels, pred, average='micro')
    ari = evaluation.adjusted_rand_score(pred,labels)
    return nmi,ac,f1,f1_micro,ari