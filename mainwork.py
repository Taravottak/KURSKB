import tkinter
import tkinter as tk
from tkinter import *
from tkinter import messagebox
import customtkinter
from PIL import ImageTk, Image

import webbrowser
import urllib.request

import ast
import hiz


customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("green")

app = customtkinter.CTk()
app.geometry("600x450")
app.title('ЭЙПОЛ')

class Base_window:

    def __init__(self, master, background_img_path, frame_width, frame_height, name_form):

        self.master = master
        self.background_img = ImageTk.PhotoImage(Image.open(background_img_path))
        self.background = customtkinter.CTkLabel(master=self.master, image=self.background_img)
        self.background.pack()

        self.frame = customtkinter.CTkFrame(master=self.background, width=frame_width, height=frame_height, corner_radius=10)
        self.frame.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

        self.name_form = name_form
        self.front_text = customtkinter.CTkLabel(master=self.frame, text=self.name_form, font=('Century Gothic', 20))
        self.front_text.place(x=120, y=45)

        self.wifi()

    def wifi(self):
        imgwifi = customtkinter.CTkImage(Image.open("wifii.png").resize((100, 100), Image.LANCZOS))
        imgnowifi = customtkinter.CTkImage(Image.open("nowifii.png").resize((100, 100), Image.LANCZOS)) 
        try:
            url = "https://www.google.com"
            urllib.request.urlopen(url)
            status = "Connected"

        except Exception as e:
            status = "Not connected"
            print(f"Exception: {e}")
        if status == "Connected":
            customtkinter.CTkLabel(master=self.frame,image=imgwifi,text="").place(x=250,y=45)
        else: customtkinter.CTkLabel(master=self.frame,image=imgnowifi,text="").place(x=250,y=45)
        
        print(status)


class Sign_in(Base_window):

    def __init__(self, master, background_img_path, frame_width, frame_height, name_form):
        super().__init__(master, background_img_path, frame_width, frame_height, name_form)
        self.img2 = customtkinter.CTkImage(Image.open("binabtn.png").resize((100, 100), Image.LANCZOS))
        self.img3 = customtkinter.CTkImage(Image.open("githbtn.png").resize((100, 100), Image.LANCZOS))
        self.data()
        self.btn()

        #self.check() - это сразу запускает функцию

    def data(self):
        self.entry1 = customtkinter.CTkEntry(master=self.frame, width=220, placeholder_text="Username")
        self.entry1.place(x=50, y=110)

        self.entry2 = customtkinter.CTkEntry(master=self.frame, width=220, placeholder_text="Password")
        self.entry2.place(x=50, y=165)

    def check(self):
        username=self.entry1.get()
        password=self.entry2.get()

        file=open('database.txt','r')
        d=file.read()
        r=ast.literal_eval(d)
        file.close()

        if username in r.keys() and password==r[username]:
            app.after(2000, app.destroy)

            if __name__ == "__main__":
                app.mainloop()
                window = hiz.Window(1280, 960, "Жесточайшее приложение")
                window.guessToken()
                window.run()
        else:
            messagebox.showerror('Invalid','invalid username or password')


    def btn(self):
        
        def opengit():
            webbrowser.open("https://github.com/Taravottak/KURSKB.git")


        def openbin():
            webbrowser.open("https://www.binance.com/ru/activity/referral-entry/CPA/together-v3?ref=CPA_00T30A0WTG")

        button_login = customtkinter.CTkButton(master=self.frame, width=220, text="Login", corner_radius=6, text_color='White',command=self.check)
        button_login.place(x=50, y=220)

        plustext = customtkinter.CTkLabel(master=self.frame, text="Don't use our bot early?", font=('Century Gothic', 13))
        plustext.place(x=50, y=310)
        button_small_up = customtkinter.CTkButton(master=self.frame, width=10, height=10, text="Sign up", corner_radius=10, text_color='White',command=self.go_sign_up)
        button_small_up.place(x=210, y=312)

        btnlink1 = customtkinter.CTkButton(master=self.frame, image=self.img2, text="Binance", width=100, height=20, corner_radius=6, compound="left", text_color='White', hover_color="#A4A4A4",command=openbin)
        btnlink1.place(x=50, y=270)

        btnlink2 = customtkinter.CTkButton(master=self.frame, image=self.img3, text="Github", width=100, height=20, corner_radius=6, compound="left", text_color='White', hover_color="#A4A4A4",command=opengit)
        btnlink2.place(x=170, y=270)

    def go_sign_up(self):
        app_new = Toplevel(app)
        app_new.geometry("900x675")# Set the window size to fill the entire screen
        app_new.title('ЭЙПОЛ')
        Sign_up(app_new, "back2w.png", 320, 360, "Sign up")
        

class Sign_up(Base_window):

    def __init__(self, master, background_img_path, frame_width, frame_height, name_form):
        super().__init__(master, background_img_path, frame_width, frame_height, name_form)
        self.data()
        self.btn()

    def data(self):
        self.entry1 = customtkinter.CTkEntry(master=self.frame, width=220, placeholder_text="Username")
        self.entry1.place(x=50, y=110)

        self.entry2 = customtkinter.CTkEntry(master=self.frame, width=220, placeholder_text="Password")
        self.entry2.place(x=50, y=165)

        self.entry3 = customtkinter.CTkEntry(master=self.frame,width=220,placeholder_text="Repeat Password")
        self.entry3.place(x=50,y=220)

    def check(self):
        username=self.entry1.get()
        password=self.entry2.get()
        repassword=self.entry3.get()
        if password==repassword:
            try:
                file=open('database.txt','r+')
                d=file.read()
                r=ast.literal_eval(d)

                dict2={username:password}
                r.update(dict2)
                file.truncate(0)
                file.close

                file=open('database.txt','w')
                w=file.write(str(r))

                messagebox.showinfo('Signup','Succesfully sign up')
                self.master.destroy()
                Sign_in(app, "back2w.png", 320, 360, "Sign in")
            except:
                file=open('database.txt','w')
                pp=str({'Username':'password'})
                file.write(pp)
                file.close
        else:
            messagebox.showerror("Invalid","both password should match")
    def go_sign_in(self):
        self.master.destroy()
        Sign_in(app, "back2w.png", 320, 360, "Sign in")  

    def btn(self):

        button_register = customtkinter.CTkButton(master=self.frame, width=220, text="Register", corner_radius=6, text_color='White',command=self.check)
        button_register.place(x=50,y=275)

        plustext = customtkinter.CTkLabel(master=self.frame, text="I have an account!", font=('Century Gothic', 13))
        plustext.place(x=50, y=310)
        button_small_up = customtkinter.CTkButton(master=self.frame, width=10, height=10, text="Sign in", corner_radius=10, text_color='White',command=self.go_sign_in)
        button_small_up.place(x=210, y=312)

Signin_window = Sign_in(app, "back2w.png", 320, 360, "Sign in")


app.mainloop()
      