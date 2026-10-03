"""
Local launcher for the Gradio dashboard with a public share link.
For permanent hosting, deploy to Hugging Face Spaces using app.py.
"""
from app import create_gradio_dashboard

if __name__ == "__main__":
    demo = create_gradio_dashboard()
    # share=True generates a temporary public URL (valid 72h)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)
