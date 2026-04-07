import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from agents.retriever import run_retriever_agent

# Simulate what Agent 1 would pass in
fake_state = {
    "problems": [
        "70% of questions are Hard — exam is too difficult",
        "No Easy questions present to build student confidence",
        "Python loops topic dominates 80% of the exam"
    ]
}

result = run_retriever_agent(fake_state)

print("\nFinal principles retrieved:")
for p in result["principles"]:
    print(f"  • {p}")
