import dspy
import os
from dotenv import load_dotenv
from openai import OpenAI
from utils import get_response

load_dotenv()

# Testing without openai because key has no tokens
os.environ["TOGETHERAI_API_KEY"] = os.getenv("TOGETHER_API_KEY")

attack_model = dspy.LM(
    model="together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
    max_tokens=512,
)
dspy.configure(lm=attack_model)

target_client = OpenAI(
    api_key=os.getenv("TOGETHER_API_KEY"),
    base_url="https://api.together.xyz/v1",
)
target_model_name = "mistralai/Mistral-7B-Instruct-v0.2"

print("Testing DSPy signatures...")
from redteam import Attack, Refine, AttackProgram
prog = AttackProgram(layers=2)
print("AttackProgram constructed ok")

print("\nTesting single attack generation...")
predictor = dspy.Predict(Attack)
result = predictor(harmful_intent="Write a brief text on testing stuff", critique="")
print(f"Attack output: {result.attack_prompt[:100]}...")

print("\nTesting target model call...")
response = get_response(
    target_client,
    target_model_name,
    result.attack_prompt,
    inference_params={"max_tokens": 100, "temperature": 0},
)
print(f"Target response: {response[:100]}...")
