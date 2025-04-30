import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, MBartTokenizer

device = "cuda:0" if torch.cuda.is_available() else "cpu"

MODEL_REPO = "dmidge/mbart-large-50-eng2span"

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_REPO).to(device)
model.eval()
tokenizer = MBartTokenizer.from_pretrained(
    "facebook/mbart-large-50", src_lang="en_XX", tgt_lang="es_XX"
)


def translate_with_confidence(src_sentence: str):
    """
    Translate English text to Spanish and return a list of (word, confidence_score) tuples.
    """
    # Tokenize input
    inputs = tokenizer(src_sentence, return_tensors="pt").to(device)
    forced_bos_token_id = tokenizer.lang_code_to_id["es_XX"]

    # Generate translation with scores
    with torch.no_grad():
        output = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    translated_tokens = output.sequences[0]

    special_ids = set(tokenizer.all_special_ids)
    cleaned_ids = [id for id in translated_tokens.tolist() if id not in special_ids]
    decoded_tokens = tokenizer.convert_ids_to_tokens(cleaned_ids)

    # Compute confidence scores
    token_confidences = []
    score_idx = 0

    for _, id in enumerate(translated_tokens[1:]):
        if id.item() in special_ids:
            continue

        logits = output.scores[score_idx][0]
        probs = F.softmax(logits, dim=-1)
        token_confidences.append(1.0 - probs[id].item())
        score_idx += 1

    # Merge subword tokens into full words with average confidence
    def merge_subword_scores(tokens, scores):
        words = []
        confidences = []
        current_word = ""
        current_scores = []

        for token, score in zip(tokens, scores):
            if token.startswith("▁"):
                if current_word:
                    words.append(current_word)
                    confidences.append(sum(current_scores) / len(current_scores))

                current_word = token.lstrip("▁")
                current_scores = [score]
            else:
                current_word += token
                current_scores.append(score)

        if current_word:
            words.append(current_word)
            confidences.append(sum(current_scores) / len(current_scores))

        return words, confidences

    return merge_subword_scores(decoded_tokens, token_confidences)


if __name__ == "__main__":
    while True:
        english = input("Enter English to translate: ")
        if english.lower() == "stop":
            break

        words, scores = translate_with_confidence(english)
        for word, score in zip(words, scores):
            print(f"{word} - {score*100.0:.4f}%")
