import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

# Configurar tkinter
root = tk.Tk()
root.title("Ventana Permanente")
root.geometry("800x600")
root.attributes('-topmost', 1)  # Mantener la ventana siempre al frente

# Crear frame para OpenCV
frame = tk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True)

# Label para mostrar el contenido de OpenCV
label = tk.Label(frame)
label.pack()

# Función para actualizar el contenido de la ventana OpenCV
def update_frame():
    ret, frame = cap.read()
    if ret:
        # Convertir frame de OpenCV a formato compatible con tkinter
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)

    # Continuar actualizando cada 10ms
    root.after(10, update_frame)

# Iniciar captura de video
cap = cv2.VideoCapture(0)

# Iniciar actualización de la ventana
update_frame()

# Ejecutar el loop principal de tkinter
root.mainloop()

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
