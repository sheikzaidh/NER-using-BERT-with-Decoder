import numpy as np

import joblib
import torch

import config
import dataset
import engine
from model import EntityModel


def predic(sentence):
    meta_data = joblib.load(config.METADATA_PATH)
    enc_tag = meta_data["enc_tag"]

    num_tag = len(list(enc_tag.classes_))

    tokenized_sentence = config.TOKENIZER.encode(sentence)
    ofToken = config.tok
    offSets = ofToken.encode(sentence)
    sentence = sentence.split()
    text = offSets.tokens

    print(sentence)
    print(tokenized_sentence)
    print(text)
    

    test_dataset = dataset.EntityDataset(
        texts=[sentence], 
        tags=[[0] * len(sentence)]
    )

    device = torch.device("cuda")
    model = EntityModel(num_tag=num_tag)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(device)

    with torch.no_grad():
        data = test_dataset[0]
        for k, v in data.items():
            data[k] = v.to(device).unsqueeze(0)
        tag, _ = model(**data)
        decodedText = enc_tag.inverse_transform(
                tag.argmax(2).cpu().numpy().reshape(-1)
            )[:len(tokenized_sentence)]
        print(decodedText)
        # print(config.TOKENIZER.encode())
    text = text[1:-1]
    decodedText = decodedText[1:-1]

    finalText = ''
    listNum = []
    for i, tex in enumerate(text):
        if tex.startswith('##'):
            finalText = finalText + tex[2:]
            listNum.append([(i-1),i])
        else:
            finalText = finalText + ' ' + tex
    if len(listNum) > 0:
      listNum.append([0, 0])
      print(f'finalText {finalText}')
      print(f'listNum {listNum}')
      finalNum = []

      for eachListNum in range(len(listNum)-1):
          if listNum[eachListNum][1] == listNum[eachListNum+1][0]:
              tempList = listNum[eachListNum]
              tempList.extend(listNum[eachListNum+1])
              finalNum.append(tempList)
          else:
              finalNum.append(listNum[eachListNum])

      finalNum = [list(set(i)) for i in finalNum]

      finalNumList = [j for i in finalNum for j in i]
      print(f'finalNum {finalNum}')

      for i in finalNum[-2]:
          if i in finalNum[-1]:
              finalNum = finalNum[:-1]
              break

      finalIntent = []
      for i in range(len(decodedText)):
          if not i in finalNumList:
              finalIntent.append(decodedText[i])
          else:
              # index = (eachList if i == eachList[0] else False for enu, eachList in enumerate(finalNum))
              index = []
              for enu, eachList in enumerate(finalNum):
                  if i == eachList[0]:
                      index=eachList
              if index:
                  tempToken = decodedText[index[0]:(index[-1]+1)]
                  print(f'temp token {tempToken}')
                  tempToken = list(set(tempToken))
                  if len(tempToken) > 1:
                    if 'O' in tempToken:
                        tempToken = ' '.join(tempToken)
                        tempToken = tempToken.replace("O",'').strip().split()
                  tempToken = tempToken[-1]
                  finalIntent.append(tempToken)
    else:
      finalText= ' '.join(text)
      finalIntent = decodedText
    


    
    intentDict = {}

    for i, inte in enumerate(finalIntent):
        if not inte == 'O':
            intentDict[finalText.strip().split(' ')[i]] = inte

    withOutZeroList = ' '.join(finalIntent)
    withOutZeroList = withOutZeroList.replace('O','').strip().split()

    return withOutZeroList, intentDict


if __name__ == "__main__":
  sentence = """
    i am having cold
    """
  lis, dic = predic(sentence)

  print(f'lis {lis} and dic {dic}')
        
