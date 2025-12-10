from transformers import AutoModel,AutoTokenizer,AutoConfig
from mamba.model import MambaTextClassification
from util import result
import numpy as np

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

import torch
import os


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

from datasets import load_from_disk
from huggingface_hub import login

# token = os.getenv("HUGGINGFACE_TOKEN")
# login(token=token, write_permission=True)
os.environ["http_proxy"] = "http://127.0.0.1:8889"
os.environ["https_proxy"] = "http://127.0.0.1:8889"
def semantic_feat_extra(dataset, dataset_name):
    # selected_dataset = load_from_disk('./text_datasets/cora_text')
    texts = dataset['text']  # 获取所有选定样本的文本
    # labels = selected_dataset['label']


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MambaTextClassification.from_pretrained(f"./papertext_{dataset_name}_after_comm")
    tokenizer = AutoTokenizer.from_pretrained(f"./papertext_{dataset_name}_after_comm")

    model = model.to(device)

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id, tokenizer.pad_token = tokenizer.eos_token_id, tokenizer.eos_token
    # 使用示例
    # texts = ["Nice to see you.", "Nice to see you too", "how old are you"]
    embeddings = []
    pre_list = []
    # n=1
    for text in texts:
        # text=text+tokenizer.eos_token
        if text == "":
            text = "None"
        inputs = tokenizer(text, return_tensors="pt")
        inputs = inputs.to(device)
        input_ids = inputs["input_ids"]
        if dataset_name == 'instagram':
            input_ids = input_ids.to(dtype=torch.long)
        # eos_token_id = tokenizer.eos_token_id
        # input_ids = torch.cat((input_ids, torch.tensor([[eos_token_id]])), dim=1)  # 添加 EOS token
        # 获取最后一层的隐藏状态
        with torch.no_grad():  # 禁用梯度计算
            hidden_states = model.backbone(input_ids)
            text_feature = hidden_states.mean(dim=1)
            logit = model.classification_head(text_feature)
            pred = logit.argmax(dim=-1)
            # outputs = model(**inputs)  # 需要设置 output_hidden_states=True
            # text_feature = outputs.text_feature
            # print(text_feature.shape)
        # print(f"{n}: {text_feature.shape}")
        # n+=1
        # doc_embedding = last_hidden_state.mean(dim=1)
        # print(doc_embedding.shape)


        # eos_token_index = torch.where(inputs == tokenizer.eos_token_id)[1].item()  # 获取 EOS token 的索引

        # eos_embedding = last_hidden_state[0, -1, :]
        embeddings.append(text_feature.squeeze(0))
        pre_list.append(pred.squeeze(0))
    embeddings_tensor = torch.stack(embeddings)
    pre_labels = torch.stack(pre_list)
    print(embeddings_tensor.shape)
    return embeddings_tensor,pre_labels


