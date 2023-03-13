from transformers import AutoTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer
import torch.optim as optim

import math
import time
import sys
import json
import numpy as np
import torch


def main():
    answers = ['A', 'B', 'C', 'D']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    display(device)

    train = []
    test = []
    valid = []

    file_name = '/kaggle/input/mcqa-dataset-cs497/test_complete.jsonl'
    with open(file_name) as json_file:
        json_list = list(json_file)
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)
        # print(result)
        base = result['fact1'] + ' [SEP] ' + result['question']['stem']
        ans = answers.index(result['answerKey'])

        obs = []
        for j in range(4):
            text = base + result['question']['choices'][j]['text'] + ' [SEP]'
            if j == ans:
                label = 1
            else:
                label = 0
            obs.append([text, label])
        train.append(obs)

        # print(obs)
        # print(' ')
        #
        # print(result['question']['stem'])
        #
        # print(' ',result['question']['choices'][0]['label'],result['question']['choices'][0]['text'])
        # print(' ',result['question']['choices'][1]['label'],result['question']['choices'][1]['text'])
        # print(' ',result['question']['choices'][2]['label'],result['question']['choices'][2]['text'])
        # print(' ',result['question']['choices'][3]['label'],result['question']['choices'][3]['text'])
        # print('  Fact: ',result['fact1'])
        # print('  Answer: ',result['answerKey'])
        # print('  ')

    file_name = '/kaggle/input/mcqa-dataset-cs497/dev_complete.jsonl'
    with open(file_name) as json_file:
        json_list = list(json_file)
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)

        base = result['fact1'] + ' [SEP] ' + result['question']['stem']
        ans = answers.index(result['answerKey'])

        obs = []
        for j in range(4):
            text = base + result['question']['choices'][j]['text'] + ' [SEP]'
            if j == ans:
                label = 1
            else:
                label = 0
            obs.append([text, label])
        valid.append(obs)

    file_name = '/kaggle/input/mcqa-dataset-cs497/train_complete.jsonl'
    with open(file_name) as json_file:
        json_list = list(json_file)
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)

        base = result['fact1'] + ' [SEP] ' + result['question']['stem']
        ans = answers.index(result['answerKey'])

        obs = []
        for j in range(4):
            text = base + result['question']['choices'][j]['text'] + ' [SEP]'
            if j == ans:
                label = 1
            else:
                label = 0
            obs.append([text, label])
        test.append(obs)
    best_pretrain_acc = 0
    #     for model_lr in [1e-5, 3e-5, 7e-5]:
    #         for lin_lr in [1e-5, 3e-5, 7e-5]:
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    linear = torch.nn.Linear(768, 1).to(device)
    optimizer = optim.Adam([{"params": model.parameters()}, {"params": linear.parameters(), 'lr': 3e-5}], lr=3e-5)
    #     optimizer = optim.Adam(linear.parameters(), lr=3e-5)
    #     optimizer = optim.Adam(model.parameters(), lr=3e-5)
    #     linear = torch.rand(768,1)

    #    Add code to fine-tune and test your MCQA classifier.

    epochs = 6
    batch_size = 64
    zero_shot = False

    # Init Training data loader
    input_x = torch.zeros(len(train) * 4, 128, dtype=torch.int)
    input_y = torch.zeros(len(train) * 4)
    for idx_q, question in enumerate(train):
        for idx_a, answer in enumerate(question):
            input_x[idx_q * 4 + idx_a] = torch.tensor(
                tokenizer(answer[0], padding="max_length", truncation=True, max_length=128)["input_ids"],
                dtype=torch.int)
            input_y[idx_q * 4 + idx_a] = answer[1]
    train_dataset = torch.utils.data.TensorDataset(input_x.to(device), input_y.to(device))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    # init val set
    val_input_x = torch.zeros(len(valid) * 4, 128, dtype=torch.int)
    val_input_y = torch.zeros(len(valid) * 4)
    for idx_q, question in enumerate(valid):
        for idx_a, answer in enumerate(question):
            val_input_x[idx_q * 4 + idx_a] = torch.tensor(
                tokenizer(answer[0], padding="max_length", truncation=True, max_length=128)["input_ids"],
                dtype=torch.int)
            val_input_y[idx_q * 4 + idx_a] = answer[1]
    val_dataset = torch.utils.data.TensorDataset(val_input_x.to(device), val_input_y.to(device))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4)

    # init test set
    test_input_x = torch.zeros(len(test) * 4, 128, dtype=torch.int)
    test_input_y = torch.zeros(len(test) * 4)
    for idx_q, question in enumerate(test):
        for idx_a, answer in enumerate(question):
            test_input_x[idx_q * 4 + idx_a] = torch.tensor(
                tokenizer(answer[0], padding="max_length", truncation=True, max_length=128)["input_ids"],
                dtype=torch.int)
            test_input_y[idx_q * 4 + idx_a] = answer[1]
    test_dataset = torch.utils.data.TensorDataset(test_input_x.to(device), test_input_y.to(device))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4)

    acc = 0
    for question_idx, (X, y) in enumerate(val_dataloader):
        model.eval()
        with torch.no_grad():
            BERT_out = model(X)[1]
            lin_out = linear(BERT_out)
            preds = torch.nn.Sigmoid()(lin_out).squeeze()
            if torch.argmax(preds).item() == torch.argmax(y).item():
                acc += 1
    print(f"Zero Shot val Accuracy:{acc / len(valid)}")
    acc = 0
    for question_idx, (X, y) in enumerate(test_dataloader):
        model.eval()
        with torch.no_grad():
            BERT_out = model(X)[1]
            lin_out = linear(BERT_out)
            preds = torch.nn.Sigmoid()(lin_out).squeeze()
            if torch.argmax(preds).item() == torch.argmax(y).item():
                acc += 1
    print(f"Zero Shot test Accuracy:{acc / len(test)}")

    #             print(len(valid))
    for epoch in range(epochs):
        for batch_idx, (X, y) in enumerate(train_dataloader):
            model.train()
            optimizer.zero_grad()
            BERT_out = model(X)[1]
            lin_out = linear(BERT_out)
            # print(lin_out)
            preds = torch.nn.Sigmoid()(lin_out).squeeze()
            # print(preds)
            loss = torch.nn.BCELoss()(preds, y)
            #                 print(loss)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        acc = 0
        for question_idx, (X, y) in enumerate(val_dataloader):
            model.eval()
            with torch.no_grad():
                BERT_out = model(X)[1]
                lin_out = linear(BERT_out)
                preds = torch.nn.Sigmoid()(lin_out).squeeze()
                if torch.argmax(preds).item() == torch.argmax(y).item():
                    acc += 1
        print(f"Val Accuracy:{acc / len(valid)}")
    acc = 0
    for question_idx, (X, y) in enumerate(test_dataloader):
        model.eval()
        with torch.no_grad():
            BERT_out = model(X)[1]
            lin_out = linear(BERT_out)
            preds = torch.nn.Sigmoid()(lin_out).squeeze()
            if torch.argmax(preds).item() == torch.argmax(y).item():
                acc += 1
    print(f"Test Accuracy:{acc / len(test)}")


if __name__ == "__main__":
    main()