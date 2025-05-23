import evaluate
import numpy as np
import pandas as pd
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

MODEL_REPO = "google/mT5-small"
PREFIX = "translate English to Spanish: "

print(f"MODEL_REPO: {MODEL_REPO}\n")

accuracy = evaluate.load("accuracy")
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")
ter = evaluate.load("ter")
METRICS = [
#        ("Accuracy", accuracy),
        ("BLEU", bleu),
        ("ROUGE", rouge),
        ("METEOR", meteor),
        ("TER", ter),
]

data = pd.read_csv("./data/combined.data")
train = data.loc[data["split"] != "test"]
test = data.loc[data["split"] == "test"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)


def preprocess_text(sample):
    input = PREFIX + str(sample[0])
    target = str(sample[1])
    return tokenizer(input, text_target=target, max_length=128, truncation=True)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


tokenized_train_data = [
    preprocess_text((row["en"], row["es"])) for _, row in train.iterrows()
]
tokenized_test_data = [
    preprocess_text((row["en"], row["es"])) for _, row in test.iterrows()
]

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, model=MODEL_REPO, return_tensors="pt"
)


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    
    results = {}
    for name, metric in METRICS:
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        if name == "BLEU":
            results[name] = result["bleu"]
        elif name == "ROUGE":
            results[name] = result["rougeL"]
        elif name == "METEOR":
            results[name] = result["meteor"]
        elif name == "TER":
            results[name] = result["score"]

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    results["gen_len"] = np.mean(prediction_lens)
    results = {k: round(v, 4) for k, v in results.items()}

    print(f"results: {results}\n")

    return results


model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_REPO)

training_args = Seq2SeqTrainingArguments(
    output_dir="mt5",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=False, # internet reported issues with this causing problems when true
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_test_data,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("mt5")
