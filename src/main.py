import os
import dotenv
from google import genai
from rouge_score import rouge_scorer
from datasets import load_dataset

from src.colors import bcolors as c

dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)
def get_response(prompt: str):
    response = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents=f"Summarize this article: {prompt}. Provide just the summary, no additional text.",
    )
    return response.text

def get_summary():
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")
    article = dataset[0]["article"]
    reference_summary = dataset[0]["highlights"]

    return [article, reference_summary]

reference_article, reference_summary = get_summary()

llm_response = get_response(reference_article)

candidate = llm_response if llm_response is not None else "No response from model"

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, candidate)

print("\r\n")
print(f"{c.OKGREEN}{c.UNDERLINE}{c.BOLD}Reference Article:{c.ENDC} {reference_article}")
print("\r\n")

print(f"{c.OKGREEN}{c.UNDERLINE}{c.BOLD}Reference Summary:{c.ENDC} {reference_summary}")
print("\r\n")

print(f"{c.OKGREEN}{c.UNDERLINE}{c.BOLD}Candidate Summary (LLM Generated):{c.ENDC} {candidate}")
print("\r\n")

print(f"{c.FAIL}{c.BOLD}{c.UNDERLINE}ROUGE-1 Score (Unigram):{c.ENDC} {scores["rouge1"]}")
print("\r\n")

print(f"{c.FAIL}{c.BOLD}{c.UNDERLINE}ROUGE-2 Score (Bigram):{c.ENDC} {scores["rouge2"]}")
print("\r\n")

print(f"{c.FAIL}{c.BOLD}{c.UNDERLINE}ROGUE-L Score (Longest Common Sequence):{c.ENDC} {scores["rougeL"]}")
print("\r\n")

