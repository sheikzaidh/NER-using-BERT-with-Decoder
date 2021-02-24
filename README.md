# NER-using-BERT-with-Decoder
# Explanation
 <h2>dataset.py</h2> <br/> File contains the custom data sampler for the pytorch model <br/> which takes the text and the entity  and return the ids , mask , the taken type <br/> which is required for the bert model and the entity in the tensor format 
 <h2>engine.py</h2> <br/> File contains the training and evaluvation steps <br/>
 <h2>model.py</h2> <br/> Contains the bert model for NER prediction <br/> It uses the CrossEntropy with the active logits loss 
 <h2>sample.txt</h2> <br/> Contains the sample dataset for the model <br/> In this each line contains the the word and the respective Entity <br> the senetnce is seperated by the space between the line in the txt file 
 <h2>train.py</h2> <br/> Contains the training procedure of the model <br/>
