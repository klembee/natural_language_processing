from transformers import T5Model, T5Tokenizer
import torch


def main():
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5Model.from_pretrained('t5-small')

    input_ids = tokenizer.encode("translate English to German: That is good.", return_tensors="pt")
    outputs = model(input_ids=input_ids)
    scores = outputs[0]

    out_indices = torch.argmax(scores, dim=2)
    predicted_token = tokenizer.convert_ids_to_tokens(out_indices[0])
    print(predicted_token)


if __name__ == "__main__":
    main()
