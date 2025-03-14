import os
import sys

# Check if all required directories exist
required_dirs = ["data", "index"]
for directory in required_dirs:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# Check for OpenAI API key
from dotenv import load_dotenv

load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    print("⚠️ OpenAI API key not found in .env file")
    print("Please add your API key to the .env file: OPENAI_API_KEY=your_key_here")
    sys.exit(1)

# Run Streamlit app
import streamlit.web.cli as stcli

sys.argv = ["streamlit", "run", os.path.join(os.path.dirname(__file__), "app.py")]
sys.exit(stcli.main())
