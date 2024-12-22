import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
from PIL import Image
from mainPru6 import gesture_detector

class GestureControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de control con gestos")
        self.root.geometry("700x500")
        
        # Configuración principal
        self.main_frame = ctk.CTkFrame(self.root, fg_color="#dad4e6")
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Título
        self.title_label = ctk.CTkLabel(self.main_frame, text="Sistema de control con gestos", font=("Arial", 18), fg_color="#bca6e6", height=50)
        self.title_label.pack(fill="x")

        # Contenedor de gestos
        self.gestures_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.gestures_frame.pack(fill="both", expand=True)

        # Botón para agregar gestos
        self.add_gesture_button = ctk.CTkButton(self.gestures_frame, text="+", font=("Arial", 50), text_color="black", fg_color="#ffffff", hover_color="#cccccc", border_width=1, border_color="#000000", width=100, height=130, command=self.open_customize_window)
        self.add_gesture_button.pack(side="left", padx=(20, 10), pady=(20, 0), anchor="nw")

        # Botón iniciar
        self.start_button = ctk.CTkButton(self.main_frame, text="Iniciar", fg_color="purple", text_color="white", command=self.start_gestures)
        self.start_button.pack(padx=10, pady=10, anchor="e")

        # Lista de gestos
        self.gesture_right_list = []
        self.gesture_left_list = []

    def open_customize_window(self):
        self.customize_window = ctk.CTkToplevel(self.root)
        self.customize_window.title("Personalizar")
        self.customize_window.geometry("400x400")

        # Mantener la ventana delante
        self.customize_window.transient(self.root)  # Asocia la ventana secundaria con la principal
        self.customize_window.lift()  # Asegura que esté en primer plano
        self.customize_window.grab_set()  # Bloquea la interacción con la ventana principal

        # Opción Izquierda o Derecha
        self.side_label = ctk.CTkLabel(self.customize_window, text="Seleccione el lado:")
        self.side_label.pack(padx=(20, 20), pady=5, anchor="w")

        # Contenedor de opcion de lado
        self.side_container = ctk.CTkFrame(self.customize_window, fg_color="transparent")
        self.side_container.pack()

        self.side_var = tk.StringVar(value="Derecha")
        self.left_radio = ctk.CTkRadioButton(self.side_container, text="Izquierda", variable=self.side_var, value="Izquierda")
        self.left_radio.pack(side="left")
        self.right_radio = ctk.CTkRadioButton(self.side_container, text="Derecha", variable=self.side_var, value="Derecha")
        self.right_radio.pack(padx=(50, 0), side="left")

        # Contenedor de gestos y teclas
        self.gesture_key_container = ctk.CTkFrame(self.customize_window, fg_color="transparent")
        self.gesture_key_container.pack(padx=(20, 0), pady=(20, 0), anchor="w")

        # Contenedor de gestos
        self.gesture_container = ctk.CTkFrame(self.gesture_key_container, fg_color="transparent")
        self.gesture_container.pack(anchor="w", side="left")

        # Selección de gesto con imágenes
        self.gesture_label = ctk.CTkLabel(self.gesture_container, text="Seleccione el gesto:")
        self.gesture_label.pack(anchor="w")

        self.gesture_var = tk.StringVar(value="pulgarArriba")

        self.gesture_images = {
            "pulgarArriba": ctk.CTkImage(light_image=Image.open("img/pulgarArriba.png"), size=(50, 50)),
            "pulgarAbajo": ctk.CTkImage(light_image=Image.open("img/pulgarAbajo.png"), size=(50, 50)),
            "manoAbierta": ctk.CTkImage(light_image=Image.open("img/manoAbierta.png"), size=(50, 50)),
            "manoCerrada": ctk.CTkImage(light_image=Image.open("img/manoCerrada.png"), size=(50, 50)),
            "amorYPaz": ctk.CTkImage(light_image=Image.open("img/amorYPaz.png"), size=(50, 50)),
            "okay": ctk.CTkImage(light_image=Image.open("img/okay.png"), size=(50, 50)),
            "rockOn": ctk.CTkImage(light_image=Image.open("img/rockOn.png"), size=(50, 50)),
            "letraL": ctk.CTkImage(light_image=Image.open("img/letraL.png"), size=(50, 50))
        }

        # Scrollable de gestos
        self.gestures_scrollable = ctk.CTkScrollableFrame(self.gesture_container, width=120, height=200)
        self.gestures_scrollable.pack()

        self.gesture_buttons = {}
        for gesture, image in self.gesture_images.items():
            button = ctk.CTkButton(self.gestures_scrollable, image=image, text="", width=90, command=lambda g=gesture: self.gesture_var.set(g))
            button.pack(pady=5)
            self.gesture_buttons[gesture] = button

        
        # Contenedor de teclas
        self.key_container = ctk.CTkFrame(self.gesture_key_container, fg_color="transparent")
        self.key_container.pack(padx=(75, 0), anchor="w", side="left")

        # Selección de tecla
        self.key_label = ctk.CTkLabel(self.key_container, text="Asigne una tecla:")
        self.key_label.pack(anchor="w")
        
        self.key_var = tk.StringVar(value="q")

        self.keys = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

        # Scrollable de teclas
        self.key_scrollable = ctk.CTkScrollableFrame(self.key_container, width=120, height=200)
        self.key_scrollable.pack()

        self.key_buttons = {}
        for key in self.keys:
            button = ctk.CTkButton(self.key_scrollable, text=key, width=90, command=lambda g=key: self.key_var.set(g))
            button.pack(pady=5)
            self.key_buttons[key] = button
        

        # Contenedor de botones
        self.button_container = ctk.CTkFrame(self.customize_window, fg_color="transparent")
        self.button_container.pack(padx=20, pady=(25, 0), fill="x")
        
        # Botón cancelar
        self.accept_button = ctk.CTkButton(self.button_container, text="Cancelar", width=100, command=lambda: self.customize_window.destroy())
        self.accept_button.pack(side="left")

        # Botón aceptar
        self.accept_button = ctk.CTkButton(self.button_container, text="Aceptar", width=100, command=self.add_gesture)
        self.accept_button.pack(padx=(160, 0), side="left")

    def add_gesture(self):
        # Extraer valores antes de cerrar la ventana
        side = self.side_var.get()
        gesture = self.gesture_var.get()
        key = self.key_var.get()

        if not key:
            messagebox.showerror("Error", "Debe asignar una tecla.")
            return

        # Crear elemento de gesto
        gesture_frame = ctk.CTkFrame(self.gestures_frame, width=100, height=130, fg_color="#ffffff", border_width=1, border_color="#000000")
        gesture_frame.pack(side="left", padx=10, pady=(20, 0), anchor="nw")
        gesture_frame.pack_propagate(False)

        gesture_label = ctk.CTkLabel(gesture_frame, text=f"Tecla {key}")
        gesture_label.pack(pady=(5, 0))

        gesture_image = ctk.CTkLabel(gesture_frame, image=self.gesture_images[gesture], text="")
        gesture_image.pack(pady=(5, 0))
        
        side_label = ctk.CTkLabel(gesture_frame, text=side)
        side_label.pack(pady=(5, 0))

        if side == "Derecha" :
            self.gesture_right_list.append({"side": side, "gesture": gesture, "key": key})
        else:
            self.gesture_left_list.append({"side": side, "gesture": gesture, "key": key})
        
        # Cerrar ventana después de procesar
        self.customize_window.destroy()


    def start_gestures(self):
        if not self.gesture_right_list and not self.gesture_left_list:
            messagebox.showwarning("Advertencia", "Debe agregar al menos un gesto antes de iniciar.")
        else:
            messagebox.showinfo("Iniciar", "Gestos iniciados correctamente.")
            self.root.destroy()  # Destruimos la ventana actual
            gesture_detector(self.gesture_right_list, self.gesture_left_list)  # Abrimos la ventana 2 y le pasamos el texto

if __name__ == "__main__":
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    app = GestureControlApp(root)
    root.mainloop()
