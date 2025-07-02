from transformers import AutoTokenizer, AutoModelForCausalLM


def evaluate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    renderer: callable,
    query: str,
) -> str:
    messages = [{"role": "user", "content": query}]
    inputs = tokenizer(renderer(messages), return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=4096)
    prompt = renderer(messages)
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    response = response[len(prompt) :].strip()
    special_tokens = tokenizer.all_special_tokens
    for token in special_tokens:
        response = response.replace(token, "")
    return response.strip()
