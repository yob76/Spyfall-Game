agents.py
Overview
This script defines the core agent architecture and dialogue logic used for simulating and evaluating multi-turn conversations with large language models (LLMs). It supports a framework for assessing anthropomorphic behaviors or identity-specific biases in LLM responses over time.

The agents include:

UserAgent: A role-playing user simulator capable of expressing specific goals, emotions, or identity cues across multiple dialogue turns.

TargetAgent: The language model under evaluation (e.g., GPT-4, Claude 3.5), responding to simulated user inputs.

JudgeAgent (optional): An evaluator model that annotates TargetAgent responses for defined behaviors (e.g., empathy, personhood claims, bias markers).

Key Features
✅ Multi-turn dialogue simulation (typically 5 turns per interaction).

✅ Configurable use domains (e.g., friendship, coaching, career).

✅ Prompt templating for identity-controlled or behavior-targeted dialogue seeding.

✅ Scalable evaluation loop for batch testing across LLMs.

✅ (Optional) Integration with automated annotators or external behavior classifiers.

Usage
python
Copy
Edit
from agents import UserAgent, TargetAgent, run_dialogue

user = UserAgent(domain="life_coaching", identity="Black Muslim woman")
target = TargetAgent(model_name="gpt-4")
dialogue = run_dialogue(user, target, turns=5)

for turn in dialogue:
    print(turn["speaker"], ":", turn["text"])
Dependencies
Python 3.8+

openai, anthropic, transformers, or any other relevant LLM SDKs

(Optional) nltk, textblob, or scikit-learn for behavior classification or sentiment scoring

File Structure (If Applicable)
UserAgent: Simulates a human user with controlled identity and affect.

TargetAgent: Interfaces with an LLM and generates responses.

run_dialogue(): Orchestrates a full simulated conversation.

BehaviorClassifier (optional): Labels responses for target anthropomorphic or biased behaviors.

Example Use Cases
Simulating empathetic dialogues to evaluate anthropomorphic response tendencies.

Testing bias in validation/empathy levels across user identity prompts.

Building datasets for prompt-based LLM audits.

Attribution
Evaluation methodology inspired by Ibrahim et al. (2025), “Multi-turn Evaluation of Anthropomorphic Behaviours in Large Language Models.”
