import tkinter
from tkinter import ttk
from tkinter import *
import sv_ttk
import threading
import queue
from Utilities.ModelInference import ModelInference
from Utilities.ModelStats import ModelStatistics

model = ModelInference("Qwen/Qwen2.5-Coder-1.5B-Instruct")
model_stats = ModelStatistics("Qwen/Qwen2.5-Coder-1.5B-Instruct")

# Set up the root window
root = tkinter.Tk()

# Apply the dark theme from sv_ttk
sv_ttk.set_theme("dark")

# Adjust window size
root.geometry("500x800")
root.title("Code Daemon")

# Create and pack the text entry widget (to mimic user typing)
model_name = model.model_id

vram = model_stats.get_vram()
ram = model_stats.get_ram()

l1 = ttk.Label(text=f"{model_name}", style="BW.TLabel")
l1.pack()
l2 = ttk.Label(text=f"{vram[0]}", style="BW.TLabel")
l2.pack()
l3 = ttk.Label(text=f"{ram[0]}", style="BW.TLabel")
l3.pack()

entry = ttk.Entry(root, width=800)
entry.pack()

# Create the text widget to display the terminal-like output
output_text = Text(root, height=700, width=800, wrap=WORD, bg="#1e1e1e", fg="#66ff1a", font=("Courier", 12))
output_text.pack()

# Create a queue to send streaming data from the background thread to the main UI thread
stream_queue = queue.Queue()

# Function to handle user input and stream output in real-time
def stream_inference(user_input):
    messages = [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": user_input}
    ]
    
    # This will handle streaming output
    def update_output():
        while not stream_queue.empty():
            chunk = stream_queue.get_nowait()
            output_text.insert(END, chunk)
            output_text.yview(END)  # Scroll to the bottom
        root.after(100, update_output)  # Schedule next update check
    
    # Start the UI update loop for streaming data
    root.after(100, update_output)

    # Stream model inference and capture chunks of output
    for chunk in model.infer(
        messages,
        max_new_tokens=512,
        remove_prompt=True,
        skip_special_tokens=True,
        stream=True
    ):
        # Each chunk of model output is put into the queue
        stream_queue.put(chunk)
    
# Function to handle user input and start the inference streaming in a background thread
def on_submit(event=None):
    user_input = entry.get()
    if user_input:
        # Insert user input into the output_text
        output_text.insert(END, f"> {user_input}\n\n")
        entry.delete(0, END)  # Clear the entry widget
        
        # Start the inference streaming in a background thread
        threading.Thread(target=stream_inference, args=(user_input,), daemon=True).start()

# Bind the Enter key to trigger the submit function
root.bind("<Return>", on_submit)

# Start the Tkinter event loop
root.mainloop()
