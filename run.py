import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Run the RAG Chatbot application")
    parser.add_argument("--api", action="store_true", help="Run the FastAPI backend")
    parser.add_argument("--ui", action="store_true", help="Run the Streamlit frontend")
    args = parser.parse_args()
    
    if args.api:
        print("Starting FastAPI server...")
        subprocess.run([sys.executable, "-m", "uvicorn", "app.main:app", "--reload"])
    elif args.ui:
        print("Starting Streamlit UI...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app/ui/streamlit_app.py"])
    else:
        print("Please specify what to run: --api or --ui")

if __name__ == "__main__":
    main()
