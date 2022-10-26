import json
import numpy as np


def read_text(data_path: str):
    with open(data_path, "r", encoding='utf-8') as f:
        reader = f.readlines()
        X, y = [], []
        for line in reader:
            line_dict = json.loads(line.strip())
            line_dict['labels'] = line_dict['labels'][0]
            X.append(line_dict['text'])
            y.append(line_dict["labels"])
    labels = set(y)
    return np.array(X), np.array(y), labels


X, y, labels = read_text("/Users/liguodong/work/data/liantong/2000144.json")


def get_label2id(labels):
    return {label: idx for idx, label in enumerate(labels)}


def get_id2label(labels):
    label2id = get_label2id(labels)
    return {idx: label for label, idx in label2id.items()}


print(get_label2id(labels))
print(get_id2label(labels))
label2id = get_label2id(labels)


def convert_label2id(y: list, label2id: dict):
    return np.array([label2id.get(temp) for temp in y])


y_id = convert_label2id(y, label2id)

print(X[:5])


# print(convert_label2id(y, label2id))


def random_sample(X: np.ndarray, y_id: np.ndarray, sample_num: int):
    print(y_id.shape, len(y_id.shape))
    idx_0 = np.where(y_id == 0)[0]
    idx_1 = np.where(y_id == 1)[0]
    idx_2 = np.where(y_id == 2)[0]

    idx_0_random = np.random.choice(idx_0, sample_num, replace=False)
    idx_1_random = np.random.choice(idx_1, sample_num, replace=False)
    idx_2_random = np.random.choice(idx_2, sample_num, replace=False)
    y_out = np.concatenate([y_id[idx_0_random], y_id[idx_1_random], y_id[idx_2_random]])
    X_out = np.concatenate([X[idx_0_random], X[idx_1_random], X[idx_2_random]])

    return X_out, y_out


X_ref = random_sample(X, y_id, 400)

print("~~~")
