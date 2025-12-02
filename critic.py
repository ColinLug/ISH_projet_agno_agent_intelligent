"""Ce script permettra de créer l'agent de critique de la complexification (cerveau gris)"""

from agno.agent import Agent

# Create each agent for each model
# Models name : Llama 3.1 8B / Qwen 2.5 7B / Mistral 7B / Falcon H1 7B
# Little models name : Gemma 3 / Qwen 4 / Mistral 7B / Llama 3.2

critic_agent = Agent(model="")
