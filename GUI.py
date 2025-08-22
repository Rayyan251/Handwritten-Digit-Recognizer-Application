import tkinter as tk
from tkinter import filedialog, Scrollbar, Canvas
from PIL import Image, ImageTk
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle
import matplotlib.pyplot as plt
# --- Load the Trained Model ---
model_save_path = 'C:\Python\Computer Vision\Handwritten Digit Detection Project\my_ocr_model.h5'  # Update with your model path 
model = keras.models.load_model(model_save_path, compile=False)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# --- Create a LabelEncoder ---
label_encoder = LabelEncoder()

# --- Load Training Labels from Pickle (or your source) ---
with open('C:\Python\Computer Vision\Handwritten Digit Detection Project\training_data.pkl', 'rb') as f:
    X_train, y_train = pickle.load(f)
label_encoder.fit(y_train) 

# --- Preprocessing Function ---
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((32, 32))  # Resize to model input size
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize
    img_array=1-img_array
    plt.imshow(img_array,cmap='gray')
    plt.show()
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    return img_array

# --- Predict Character Function ---
def predict_character(image_data):
    prediction = model.predict(image_data)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_char = label_encoder.inverse_transform([predicted_class])[0]
    result_label.config(text=f"Prediction: {predicted_char}")  # Update the label

# --- Load Image Function ---
def open_image():
    global result_label
    filepath = filedialog.askopenfilename(
        initialdir="/",
        title="Select an image",
        filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("all files", "*.*"))
    )
    if filepath:
        try:
            #img = Image.open(filepath)
            #img = ImageTk.PhotoImage(img)
            #image_label.config(image=img)
            #image_label.image = img  # Keep a reference

            # --- Add Scrollbar ---
            #canvas.delete("all") 
            #image_canvas = ImageTk.PhotoImage(img)
            #canvas.config(scrollregion=(0, 0, image_canvas.width(), image_canvas.height()))
            #canvas.create_image(0, 0, anchor="nw", image=image_canvas)

            image_data = preprocess_image(filepath)
            predict_character(image_data)
        except Exception as e:
            print(f"Error loading image: {e}")

# --- Create the GUI Window ---
root = tk.Tk()
root.title("Handwritten Character Recognizer")

# --- Create Load Image Button ---
load_button = tk.Button(root, text="Load Image", command=open_image)
load_button.pack(pady=10)

# --- Create Label for Image Display ---
#image_label = tk.Label(root) #Remove this line
#image_label.pack() #Remove this line

# --- Create Canvas for Image Display (with Scrollbar) ---
#canvas = Canvas(root, width=400, height=400) #Remove this line
#canvas.pack(side=tk.LEFT) #Remove this line

#scrollbar = Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview) #Remove this line
#scrollbar.pack(side=tk.RIGHT, fill=tk.Y) #Remove this line

#canvas.config(yscrollcommand=scrollbar.set) #Remove this line

# --- Create Label for Prediction Display ---
result_label = tk.Label(root, text="Prediction:", font=("Arial", 14))
result_label.pack()

# --- Run the GUI ---
root.mainloop()