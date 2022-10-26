from datasets import load_dataset, load_metric

task = "imdb"

# dataset = load_dataset(task)

# ~/.cache/huggingface/datasets
dataset = load_dataset(path=task)

print(dataset)

# ~/.cache/huggingface/metrics/
metric = load_metric("accuracy")

result = metric.compute(predictions=[0,0,1,1], references=[0,1,1,1])

print(result)

splitted_datasets = dataset["train"].train_test_split(test_size=0.3)
print(splitted_datasets)

from transformers import AutoTokenizer

model_checkpoint = "distilbert-base-uncased"

# use_fast: Whether or not to try to load the fast version of the tokenizer.
# Most of the tokenizers are available in two flavors: a full python
# implementation and a “Fast” implementation based on the Rust library 🤗 Tokenizers.
# The “Fast” implementations allows a significant speed-up in particular
# when doing batched tokenization, and additional methods to map between the
# original string (character and words) and the token space.
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

tokenizer(["Hello, this one sentence!"])
# {'input_ids': [[101, 7592, 1010, 2023, 2028, 6251, 999, 102]], 'attention_mask':
# [[1, 1, 1, 1, 1, 1, 1, 1]]}
# input_ids: the tokenizer vocabulary indexes of the tokenized input sentence
# attention_mask: 0 if the corresponding input_id is padding, 1 otherwise



def preprocess_function_batch(examples):
    # truncation=True: truncate to the maximum acceptable input length for
    # the model.
    return tokenizer(examples["text"], truncation=True)

splitted_datasets_encoded = splitted_datasets.map(preprocess_function_batch, batched=True)


from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification

# num_labels: number of labels to use in the last layer added to the model,
# typically for a classification task.

# The AutoModelForSequenceClassification class loads the
# DistilBertForSequenceClassification class as underlying model. Since
# AutoModelForSequenceClassification doesn't accept the parameter 'num_labels',
# it is passed to the underlying class DistilBertForSequenceClassification, which
# accepts it.

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

# This will issue a warning about some of the pretrained weights not being used
# and some weights being randomly initialized. That’s because we are throwing
# away the pretraining head of the BERT model to replace it with a classification
# head which is randomly initialized. We will fine-tune this model on our task,
# transferring the knowledge of the pretrained model to it (which is why doing
# this is called transfer learning).



model_output_dir = f"{model_checkpoint}-finetuned-{task}"
print(model_output_dir) # distilbert-base-uncased-finetuned-imdb



# Start TensorBoard before training to monitor it in progress
# %load_ext tensorboard
# %tensorboard --logdir '{model_output_dir}'/runs


args = TrainingArguments(
    # output_dir: directory where the model checkpoints will be saved.
    output_dir=model_output_dir,
    # evaluation_strategy (default "no"):
    # Possible values are:
    # "no": No evaluation is done during training.
    # "steps": Evaluation is done (and logged) every eval_steps.
    # "epoch": Evaluation is done at the end of each epoch.
    evaluation_strategy="steps",
    # eval_steps: Number of update steps between two evaluations if
    # evaluation_strategy="steps". Will default to the same value as
    # logging_steps if not set.
    eval_steps=50,
    # logging_strategy (default: "steps"): The logging strategy to adopt during
    # training (used to log training loss for example). Possible values are:
    # "no": No logging is done during training.
    # "epoch": Logging is done at the end of each epoch.
    # "steps": Logging is done every logging_steps.
    logging_strategy="steps",
    # logging_steps (default 500): Number of update steps between two logs if
    # logging_strategy="steps".
    logging_steps=50,
    # save_strategy (default "steps"):
    # The checkpoint save strategy to adopt during training. Possible values are:
    # "no": No save is done during training.
    # "epoch": Save is done at the end of each epoch.
    # "steps": Save is done every save_steps (default 500).
    save_strategy="steps",
    # save_steps (default: 500): Number of updates steps before two checkpoint
    # saves if save_strategy="steps".
    save_steps=200,
    # learning_rate (default 5e-5): The initial learning rate for AdamW optimizer.
    # Adam algorithm with weight decay fix as introduced in the paper
    # Decoupled Weight Decay Regularization.
    learning_rate=2e-5,
    # per_device_train_batch_size: The batch size per GPU/TPU core/CPU for training.
    per_device_train_batch_size=16,
    # per_device_eval_batch_size: The batch size per GPU/TPU core/CPU for evaluation.
    per_device_eval_batch_size=16,
    # num_train_epochs (default 3.0): Total number of training epochs to perform
    # (if not an integer, will perform the decimal part percents of the last epoch
    # before stopping training).
    num_train_epochs=1,
    # load_best_model_at_end (default False): Whether or not to load the best model
    # found during training at the end of training.
    load_best_model_at_end=True,
    # metric_for_best_model:
    # Use in conjunction with load_best_model_at_end to specify the metric to use
    # to compare two different models. Must be the name of a metric returned by
    # the evaluation with or without the prefix "eval_".
    metric_for_best_model="accuracy",
    # report_to:
    # The list of integrations to report the results and logs to. Supported
    # platforms are "azure_ml", "comet_ml", "mlflow", "tensorboard" and "wandb".
    # Use "all" to report to all integrations installed, "none" for no integrations.
    report_to="tensorboard"
)


# Function that returns an untrained model to be trained
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(model_checkpoint,
                                                              num_labels=2)

# Function that will be called at the end of each evaluation phase on the whole
# arrays of predictions/labels to produce metrics.
def compute_metrics(eval_pred):
    # Predictions and labels are grouped in a namedtuple called EvalPrediction
    predictions, labels = eval_pred
    # Get the index with the highest prediction score (i.e. the predicted labels)
    predictions = np.argmax(predictions, axis=1)
    # Compare the predicted labels with the reference labels
    results =  metric.compute(predictions=predictions, references=labels)
    # results: a dictionary with string keys (the name of the metric) and float
    # values (i.e. the metric values)
    return results

# Since PyTorch does not provide a training loop, the 🤗 Transformers library
# provides a Trainer API that is optimized for 🤗 Transformers models, with a
# wide range of training options and with built-in features like logging,
# gradient accumulation, and mixed precision.
trainer = Trainer(
    # Function that returns the model to train. It's useful to use a function
    # instead of directly the model to make sure that we are always training
    # an untrained model from scratch.
    model_init=model_init,
    # The training arguments.
    args=args,
    # The training dataset.
    train_dataset=splitted_datasets_encoded["train"],
    # The evaluation dataset. We use a small subset of the validation set
    # composed of 150 samples to speed up computations...
    eval_dataset=splitted_datasets_encoded["test"].shuffle(42).select(range(150)),
    # Even though the training set and evaluation set are already tokenized, the
    # tokenizer is needed to pad the "input_ids" and "attention_mask" tensors
    # to the length managed by the model. It does so one batch at a time, to
    # use less memory as possible.
    tokenizer=tokenizer,
    # Function that will be called at the end of each evaluation phase on the whole
    # arrays of predictions/labels to produce metrics.
    compute_metrics=compute_metrics
)

# ... train the model!
trainer.train()

import numpy as np

# Tokenize test set
dataset_test_encoded = dataset["test"].map(preprocess_function_batch, batched=True)
# Use the model to get predictions
test_predictions = trainer.predict(dataset_test_encoded)
# For each prediction, create the label with argmax
test_predictions_argmax = np.argmax(test_predictions[0], axis=1)
# Retrieve reference labels from test set
test_references = np.array(dataset["test"]["label"])
# Compute accuracy
metric.compute(predictions=test_predictions_argmax, references=test_references)
# {'accuracy': 0.91888}





