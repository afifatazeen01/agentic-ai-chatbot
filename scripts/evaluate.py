import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.graph.workflow import rag_app

QUESTIONS = [
    "What is the core definition of Agentic AI as outlined in the eBook?",
    "What are the main architectural components required to build agentic systems?",
    "What real-world industry use cases for Agentic AI are discussed in the eBook?",
    "How does Agentic AI differ from traditional generative AI chatbots according to the text?",
    "What key challenges or limitations of Agentic AI are mentioned?",
    "What is the capital of France?"
]

def evaluate():
    print("========================================")
    print("RAG EVALUATION")
    print("==============\n")
    
    for query in QUESTIONS:
        print(f"Query:\n{query}\n")
        
        try:
            state = {"query": query}
            result = rag_app.invoke(state)
            
            answer = result.get("final_answer", "")
            confidence = result.get("confidence_score", 0.0)
            grounded = result.get("grounded", False)
            chunks = result.get("retrieved_documents", [])
            should_refuse = result.get("should_refuse", False)
            
            pages = sorted(list(set([str(c.get("page")) for c in chunks])))
            
            print("Answer:")
            print(answer + "\n")
            
            print("Confidence:")
            print(f"{confidence:.2f}\n")
            
            print("Pages:")
            print(", ".join(pages) if pages else "None" + "\n")
            
            print("Grounded:")
            print("YES\n" if grounded else "NO\n")
            
            print("Status:")
            if query == "What is the capital of France?":
                if should_refuse and confidence <= 0.5:
                    print("PASS - Correctly refused out-of-scope question")
                else:
                    print("FAIL - Did not correctly refuse out-of-scope question")
            else:
                if grounded and confidence > 0.5:
                    print("PASS")
                else:
                    print("FAIL")
                    
        except Exception as e:
            print(f"ERROR: {str(e)}")
            
        print("\n========================================\n")

if __name__ == "__main__":
    evaluate()
