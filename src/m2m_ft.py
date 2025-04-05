import evaluate
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    M2M100Tokenizer,
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")
ter = evaluate.load("ter")

print("Reading in data")
data = pd.read_csv("./data/combined.data")
test_split = data.loc[data["split"] == "test"]
test_en = test_split["en"].to_list()
test_en = [str(e) for e in test_en]
references = test_split["es"].to_list()
references = [str(e) for e in references]


print("Initializing baseline DataFrame for results")
results = pd.DataFrame(
    columns=["BLEU", "METEOR", "ROUGE", "TER"],
    index=[],
)

m2m_tokenizer = M2M100Tokenizer.from_pretrained(
    "facebook/m2m100_418M", src_lang="en", tgt_lang="es"
)


def translate_batched_m2m(m2m_model, en_sentences, batch_size=8):
    translations = []
    for i in tqdm(
        range(0, len(en_sentences), batch_size), desc="Generating M2M100 translations"
    ):
        encoded = m2m_tokenizer(
            en_sentences[i : i + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        generated_tokens = m2m_model.generate(
            **encoded, forced_bos_token_id=m2m_tokenizer.get_lang_id("es")
        ).to("cpu")
        translations.extend(
            m2m_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        )

    return translations


def evaluate_m2m():
    m2m_model = AutoModelForSeq2SeqLM.from_pretrained(
        "models/m2m100", local_files_only=True
    )
    translations = translate_batched_m2m(m2m_model, test_en)

    bleu_result = bleu.compute(predictions=translations, references=references)["bleu"]
    meteor_result = meteor.compute(predictions=translations, references=references)[
        "meteor"
    ]
    rouge_result = rouge.compute(predictions=translations, references=references)[
        "rougeL"
    ]
    ter_result = ter.compute(predictions=translations, references=references)["score"]
    results.loc["M2M FT"] = {
        "BLEU": bleu_result,
        "METEOR": meteor_result,
        "ROUGE": rouge_result,
        "TER": ter_result,
    }


def run_benchmarks():
    evaluate_m2m()


if __name__ == "__main__":
    run_benchmarks()
    results.to_csv("data/m2m_ft.csv", index=False)
