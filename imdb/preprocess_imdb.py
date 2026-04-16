from tqdm import tqdm
import os
import json

IMDB_DATASET_DIR = "dataset/aclImdb"

def load_data(data_dir):
    x , y = [], []
    for file in tqdm(os.listdir(os.path.join(data_dir, "pos"))):
        with open(os.path.join(data_dir, "pos", file), "r") as f:
            x.append(f.read())
            y.append(1)
    for file in tqdm(os.listdir(os.path.join(data_dir, "neg"))):
        with open(os.path.join(data_dir, "neg", file), "r") as f:
            x.append(f.read())
            y.append(0)
    return x, y


x_train, y_train = load_data(os.path.join(IMDB_DATASET_DIR, "train"))
x_test, y_test   = load_data(os.path.join(IMDB_DATASET_DIR, "test"))

train_dic = []
for i in tqdm(range(len(x_train))):
    train_dic.append((x_train[i], y_train[i]))
with open(os.path.join(IMDB_DATASET_DIR, "train.json"), "w") as f:
    json.dump(train_dic, f, ensure_ascii=False, indent=4)

test_dic = []
for i in tqdm(range(len(x_test))):
    test_dic.append((x_test[i], y_test[i]))
with open(os.path.join(IMDB_DATASET_DIR, "test.json"), "w") as f:
    json.dump(test_dic, f, ensure_ascii=False, indent=4)

print("train_dic len:", len(train_dic))
print("test_dic len:", len(test_dic))
