from tkinter import *
from Utilities.ModelInference import ModelInference

codellama = ModelInference("Qwen/Qwen2.5-Coder-1.5B-Instruct")

# GUI
root = Tk()
root.title("Chatbot")

BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

# Send function
def send():
    # Get user input and display it in the chat window
    user_input = e.get()
    messages = [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": user_input}
    ]   
    send_text = "You -> " + user_input
    txt.insert(END, "\n" + send_text)

    # Convert the user input to lowercase (optional, depending on your logic)
    user = user_input.lower()
    print(messages)

    # # Check if user input is "hello" and handle the case where you don't want to process it
    if user != "hello":
        # Call the model's infer method with streaming enabled
        # Instead of appending one char at a time, we will store and append full output
        full_response = codellama.infer(
            messages,
            max_new_tokens=512,
            remove_prompt=True,
            skip_special_tokens=True,
            stream=True
        )

        print(full_response)

        # Insert the full response at once into the chat window
        txt.insert(END, "\n" + "Assistant -> " + full_response)

    # Clear the input field after sending
    e.delete(0, END)
    txt.yview(END)  # Scroll to the bottom to show the latest message

# Create a Text widget to display the conversation
txt = Text(root, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT, width=60, height=20)
txt.grid(row=1, column=0, columnspan=2)

# Create a Scrollbar for the Text widget
scrollbar = Scrollbar(txt)
scrollbar.place(relheight=1, relx=0.974)

# Create an Entry widget for user input
e = Entry(root, bg="#2C3E50", fg=TEXT_COLOR, font=FONT, width=55)
e.grid(row=2, column=0)

# Create a Send button
send_button = Button(root, text="Send", font=FONT_BOLD, bg=BG_GRAY, command=send)
send_button.grid(row=2, column=1)

# Run the Tkinter main loop
root.mainloop()
