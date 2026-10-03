from dashboard.app import create_gradio_dashboard

if __name__ == "__main__":
    demo = create_gradio_dashboard()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
