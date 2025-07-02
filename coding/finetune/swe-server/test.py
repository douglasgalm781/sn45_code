from swebase import LLMClient

llm = LLMClient()

passing = True
response, _ = llm("say the word happy", "gpt-4o")
if "happy" not in response.lower():
    passing = False

response, _ = llm("say the word sad", "claude-3-5-sonnet")
if "sad" not in response.lower():
    passing = False

print("The test passed" if passing else "The test failed")
