import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Import answer generation logic
from answer_genration import generate_answer

def run_tests():
    test_cases = [
        {
            "name": "Preset 1 (Structured): Rice yield in Haryana during Kharif",
            "query": "What is the yield of Rice in Haryana during Kharif season?"
        },
        {
            "name": "Preset 2 (General): Suitable crops for Punjab",
            "query": "What crops are suitable for Punjab state?"
        },
        {
            "name": "Preset 3 (General): Organic pest control for vegetables",
            "query": "What are effective organic pest management methods for vegetables?"
        },
        {
            "name": "Custom 1 (Structured - Previously Failed): Wheat yield in Karnal in 2018",
            "query": "What was the wheat production in Karnal in 2018?"
        },
        {
            "name": "Custom 2 (General - Previously Failed): Control aphids on potatoes",
            "query": "How do I control aphids on potato plants?"
        },
        {
            "name": "Out-of-scope (Unrelated): Prime Minister of India",
            "query": "Who is the prime minister of India?"
        }
    ]
    
    print("=" * 70)
    print("🌾 KrishiMitra Verification Suite 🌾")
    print("=" * 70)
    
    for idx, tc in enumerate(test_cases, 1):
        print(f"\n[{idx}] Running test: {tc['name']}")
        print(f"    Query: '{tc['query']}'")
        try:
            res = generate_answer(tc["query"])
            print(f"    Answer: {res['answer'].strip()}")
            print(f"    Used Context: {res['used_context']}")
            print(f"    Sources count: {len(res['sources'])}")
            if res['sources']:
                print(f"    Top Source: {res['sources'][0]['source']} | State: {res['sources'][0]['state']} | Crop: {res['sources'][0]['crop']}")
        except Exception as e:
            print(f"    ❌ Error: {e}")
        print("-" * 70)

if __name__ == "__main__":
    run_tests()
