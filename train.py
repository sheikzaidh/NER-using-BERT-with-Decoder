import pandas as pd
import numpy as np

import joblib
import torch

from sklearn import preprocessing
from sklearn import model_selection

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config
import dataset
import engine
from model import EntityModel


def process_data(data_path):
    df = pd.read_csv(data_path, encoding="latin-1")
    df.loc[:, "Sentence #"] = df["Sentence #"].fillna(method="ffill")

    enc_tag = preprocessing.LabelEncoder()

    df.loc[:, "Tag"] = enc_tag.fit_transform(df["Tag"])

    sentences = df.groupby("Sentence #")["Word"].apply(list).values
    tag = df.groupby("Sentence #")["Tag"].apply(list).values
    return sentences, tag, enc_tag


if __name__ == "__main__":
    # sentences, tag, enc_tag = process_data(config.TRAINING_FILE)
    listName = []
    with open(config.TRAINING_FILE) as f:
        for i in f.readlines():
            listName.append(i)

    listName = [i.replace('\n',"") for i in listName]


    sentenceAll = []
    intentALl = []

    tempSentence = []
    tempIntent = []

    for i in listName:
        if not i.strip() == '':
            # print(f'i split {i.split(" ")}')
            sent, inte = i.strip().split(' ')
            tempSentence.append(sent)
            tempIntent.append(inte)

        else:
            sentenceAll.append(tempSentence)
            intentALl.append(tempIntent)
            tempSentence = []
            tempIntent = []
    

    sentences = np.array(sentenceAll)
    # tag = np.array(intentALl)
    intentUnique = list(set([i for intent in intentALl for i in intent]))

    enc = preprocessing.LabelEncoder()

    enc.fit(intentUnique)

    tag = np.array([enc.transform(intent) for intent in intentALl ])
    # meta_data = {
    #     "enc_tag": enc_tag
    # }

    meta_data = {
      "enc_tag": enc
    }

    joblib.dump(meta_data, config.METADATA_PATH)

    num_tag = len(list(enc.classes_))

    (
        train_sentences,
        test_sentences,
        train_tag,
        test_tag
    ) = model_selection.train_test_split(sentences, tag, random_state=42, test_size=0.1)

    train_dataset = dataset.EntityDataset(
        texts=train_sentences, tags=train_tag
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_dataset = dataset.EntityDataset(
        texts=test_sentences, tags=test_tag
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )

    device = torch.device("cuda")
    model = EntityModel(num_tag=num_tag)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(train_sentences) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_loss = np.inf
    stopNum = 0
    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        test_loss = engine.eval_fn(valid_data_loader, model, device)
        print(f"Train Loss = {train_loss} Valid Loss = {test_loss}")
        if test_loss < best_loss:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_loss = test_loss
            stopNum = 0
        else:
            if stopNum == config.EARLYSTOPPING:
                break
            print(f'Early stooping {stopNum}')
            stopNum = stopNum +1

