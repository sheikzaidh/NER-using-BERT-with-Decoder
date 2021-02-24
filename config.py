import transformers
import tokenizers

tok = tokenizers.BertWordPieceTokenizer('/content/drive/MyDrive/notebooks/bertNER1/vacob/bert-base-uncased.txt', lowercase=True)

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 10
EARLYSTOPPING = 6
METADATA_PATH = "D:/bertNER1/model/meta.bin"
MODEL_PATH = "/content/drive/MyDrive/notebooks/bertNER1/model/model.bin"
TRAINING_FILE = '/content/drive/MyDrive/notebooks/bertNER1/sample.txt'
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    'bert-base-uncased',
    do_lower_case=True
)
