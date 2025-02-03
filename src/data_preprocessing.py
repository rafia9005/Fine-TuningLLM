from datasets import load_dataset
from .config import max_seq_length

def load_and_preprocess_data():
    dataset = load_dataset('b-mc2/sql-create-context', split="train")

    alpaca_prompt = """Context: {context}
                    Question: {question}
                    Answer: {answer}"""

    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts(src):
        datasets_context = src["context"]
        datasets_question = src["question"]
        datasets_answer = src["answer"]

        texts = [] # accommodate datasets here

        for context, question, answer in zip(datasets_context, datasets_question, datasets_answer):
            text = alpaca_prompt.format(context=context, question=question, answer=answer) + EOS_TOKEN
            texts.append(text)

        return {"text": texts}

    dataset = dataset.map(formatting_prompts, batched=True)
    return dataset
