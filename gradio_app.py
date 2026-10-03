import sys
import os
import argparse

# Ensure project root is in sys.path for robust relative imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dashboard.app import create_gradio_dashboard, launch_dashboard_app

def main():
    parser = argparse.ArgumentParser(description="Smart Water Quality Prediction System - Gradio Dashboard Launcher")
    parser.add_argument("--port", type=int, default=7860, help="Initial port number for Gradio server (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Generate a public shareable gradio.live URL (default: False)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface IP (default: 127.0.0.1)")
    args = parser.parse_args()

    demo = create_gradio_dashboard()
    launch_dashboard_app(
        demo=demo,
        start_port=args.port,
        max_port=args.port + 15,
        host=args.host,
        share=args.share
    )

if __name__ == "__main__":
    main()
