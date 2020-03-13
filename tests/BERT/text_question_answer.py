from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

text = r"""
Bob Marley, OM (6 February 1945 – 11 May 1981) was a Jamaican singer, songwriter and musician. 
Considered one of the pioneers of reggae, his musical career was marked by fusing elements of reggae, ska, and rocksteady, as well as his distinctive vocal and songwriting style. 
Marley's contributions to music increased the visibility of Jamaican music worldwide, and made him a global figure in popular culture for over a decade. 
Over the course of his career Marley became known as a Rastafari icon, and he infused his music with a sense of spirituality. 
He is also considered a global symbol of Jamaican culture and identity, and was controversial in his outspoken support for the legalization of marijuana, while he also advocated for Pan-Africanism.
"""

questions = [
    "What did Bob do?",
    "What was his job?"
]
np_answers = 2

for question in questions:
    inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = model(**inputs)

    answer_starts = torch.topk(
        answer_start_scores,
        np_answers
    )[1][0]  # Get the most likely beginning of answer with the argmax of the score

    answer_ends = torch.topk(answer_end_scores, np_answers)[1][0]  # Get the most likely end of answer with the argmax of the score

    print(f"Question: {question}")
    print("Answers:")
    for i in range(np_answers):
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(input_ids[answer_starts[i]:answer_ends[i] + 1]))
        print(answer)
