import tkinter
from tkinter import ttk
from tkinter import *
import sv_ttk
import threading
import queue
from Utilities.ModelInference import ModelInference
from Utilities.ModelStats import ModelStatistics

model = ModelInference("Qwen/Qwen2.5-Coder-1.5B-Instruct", threaded_streaming=True)
model_stats = ModelStatistics("Qwen/Qwen2.5-Coder-1.5B-Instruct")

import queue
from threading import Thread
import tkinter as tk

# Assuming root and output_text are initialized in your Tkinter app

# This is the queue used for streaming data from the model to the UI
stream_queue = queue.Queue()

import queue
import tkinter as tk
from threading import Thread

# Create a thread-safe queue for communication between the inference thread and Tkinter UI thread
stream_queue = queue.Queue()

def stream_inference(user_input):
    messages = [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": user_input}
    ]
    
    # Function to update the Tkinter UI with the streamed text
    def update_output():
        while not stream_queue.empty():
            chunk = stream_queue.get_nowait()
            output_text.insert(tk.END, chunk)
            output_text.yview(tk.END)  # Scroll to the bottom
        root.after(100, update_output)  # Schedule the next update
    
    # Start the UI update loop
    root.after(100, update_output)
    
    # Run the streaming inference in a separate thread
    def stream_model_output():
        response = model.infer(
            messages,
            max_new_tokens=512,
            temperature=0.5,
            top_p=0.95,
            top_k=40,
            stream=True,
            remove_prompt=True,
            skip_special_tokens=True
        )
        # Here, the response will already be streamed into stream_queue
        return response
    
    # Start the inference in a separate thread
    threading_thread = Thread(target=stream_model_output)
    threading_thread.start()

# Tkinter UI setup
root = tk.Tk()
output_text = tk.Text(root, wrap=tk.WORD)
output_text.pack(expand=True, fill=tk.BOTH)

# Example user input to test the streaming
user_input = "Please help me with my coding problem."

# Start streaming when the user submits input
stream_inference(user_input)

root.mainloop()

