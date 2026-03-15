"""Run the Streamlit UI. Usage: python run_streamlit.py"""
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app/app.py", "--server.port=8501"], check=True)
