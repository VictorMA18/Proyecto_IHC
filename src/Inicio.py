import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import subprocess

# Función para abrir el programa "Detector de Emociones"
def abrir_detector_emociones(ventana_actual):
    ventana_actual.destroy()  # Cierra la ventana actual
    subprocess.Popen([r"C:\Users\Bryan\Documents\face_classification-master\venv\Scripts\python",
                      r"C:\Users\Bryan\Documents\Proyecto_IHC\Proyecto_IHC\src\Detector_Emociones_Reporte.py"])

# Función para abrir el programa "Detector de Gestos"
def abrir_detector_gestos(ventana_actual):
    ventana_actual.destroy()  # Cierra la ventana actual
    subprocess.Popen([r"C:\Users\Bryan\Documents\face_classification-master\venv\Scripts\python",
                      r"C:\Users\Bryan\Documents\Proyecto_IHC\Proyecto_IHC\src\Detector_Emociones_Reporte.py"])

# Crear la ventana principal
ventana_principal = tk.Tk()
ventana_principal.title("Expresiones")
window_width = 600
window_height = 350
screen_width = ventana_principal.winfo_screenwidth()
screen_height = ventana_principal.winfo_screenheight()
x_cordinate = int((screen_width / 2) - (window_width / 2))
y_cordinate = int((screen_height / 2) - (window_height / 2))

ventana_principal.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")

label = ttk.Label(ventana_principal, text="Menú principal", font=("Arial", 20))
label.pack(pady=20)

# Cargar imágenes para los botones
emotions_image_path = r"C:\Users\Bryan\Documents\Proyecto_IHC\Proyecto_IHC\src\assets\emotions.png"
gestures_image_path = r"C:\Users\Bryan\Documents\Proyecto_IHC\Proyecto_IHC\src\assets\hands.png"

emotions_image = ImageTk.PhotoImage(Image.open(emotions_image_path).resize((60, 60)))
gestures_image = ImageTk.PhotoImage(Image.open(gestures_image_path).resize((60, 60)))

# Crear un frame para organizar los botones en una fila
button_frame = tk.Frame(ventana_principal)
button_frame.pack(pady=20)

# Crear un estilo para los botones
style = ttk.Style()
style.configure("TButton", font=("Arial", 14), padding=10)

# Botón para abrir el detector de emociones
boton_emociones = ttk.Button(button_frame, text="Detector de Emociones", image=emotions_image, compound="top",
                             style="TButton", width=20,
                             command=lambda: abrir_detector_emociones(ventana_principal))
boton_emociones.grid(row=0, column=1, padx=20)

# Botón para abrir el detector de gestos
boton_gestos = ttk.Button(button_frame, text="Detector de Gestos", image=gestures_image, compound="top",
                          style="TButton", width=20,
                          command=lambda: abrir_detector_gestos(ventana_principal))
boton_gestos.grid(row=0, column=0, padx=20)

ventana_principal.mainloop()
