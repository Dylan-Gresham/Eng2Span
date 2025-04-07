from pprint import pprint

import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

device = "cuda:0" if torch.cuda.is_available() else "cpu"

MODEL_REPO = "models/opus"

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_REPO, local_files_only=True).to(
    device
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)


def translate_with_confidence(src_sentence: str):
    """
    Translate English text to Spanish and return a list of (word, confidence_score) tuples.
    """
    # Tokenize input
    inputs = tokenizer(src_sentence, return_tensors="pt").to(device)

    # Generate translation with scores
    with torch.no_grad():
        output = model.generate(
            **inputs, return_dict_in_generate=True, output_scores=True
        )

    translated_tokens = output.sequences[0]

    special_ids = tokenizer.all_special_ids
    cleaned_ids = [id for id in translated_tokens.tolist() if id not in special_ids]
    decoded_tokens = tokenizer.convert_ids_to_tokens(cleaned_ids)

    # Compute confidence scores
    token_confidences = []
    score_idx = 0

    for i, id in enumerate(translated_tokens[1:]):
        if id.item() in special_ids:
            continue

        logits = output.scores[score_idx][0]
        probs = F.softmax(logits, dim=-1)
        token_confidences.append(probs[id].item())
        score_idx += 1

    # Merge subword tokens into full words with average confidence
    def merge_subword_scores(tokens, scores):
        words = []
        current_word = ""
        current_scores = []

        for token, score in zip(tokens, scores):
            if token.startswith("▁"):
                if current_word:
                    words.append(
                        (current_word, sum(current_scores) / len(current_scores))
                    )
                current_word = token.lstrip("▁")
                current_scores = [score]
            else:
                current_word += token
                current_scores.append(score)

        if current_word:
            words.append((current_word, sum(current_scores) / len(current_scores)))
        return words

    return merge_subword_scores(decoded_tokens, token_confidences)


if __name__ == "__main__":
    while True:
        english = input("Enter English to translate: ")
        if english.lower() == "stop":
            break
        pprint(translate_with_confidence(english))
