from tkinter import *
from tkinter import messagebox

root=Tk()
root.title('LOGIN')
root.geometry('925x500+300+200')
root.configure(bg="#fff")
root.resizable(False,False)

def signin():
    username=user.get()
    password=passwd.get()

    if username=='admin' and password=='1':
        screen= Toplevel(root)
        screen.title("thanks for sign in")
        screen.geometry('925x500+300+200')
        screen.config(bg="white")

        Label(screen,text='Thanks',bg='#fff',font=('Calibri(Body)',50,'bold')).pack(expand=True)

        screen.mainloop()
    elif username!='admin' and password!='1':
        messagebox.showerror("Invalid","invalid username and password")
    elif password!="1":
         messagebox.showerror("Invalid","invalid password")
    '''elif username!="admin":
         messagebox.showerror("Invalid","invalid username")'''#строчка бесполезная проверка юзера с правильным паролем
#------------------------------------------------
#вставка фото
img = PhotoImage(file='bitok(350x350).png')
Label(root,image=img,bg='white').place(x=50,y=65)#label-один из классов Tk используемый для отображения img/text в окне
#------------------------------------------------
frame=Frame(root,width=350,height=350,bg="white")#можно посмотреть рамки и ее границы сменой цвета
frame.place(x=480,y=70)

heading=Label(frame,text='Sign in',fg='#57a1f8',bg='white',font=('Times New Roman',23,'bold'))
heading.place(x=150,y=1)
#------------------------------------------------
def on_enter(e):
    user.delete(0,'end')
def on_leave(e):
    name=user.get()
    if name=='':
        user.insert(0,'Username')

user = Entry(frame,width=25,fg='black',border=0,bg="white",font=('Times New Roman',11))
user.place(x=55,y=50)
user.insert(0,'Username')
user.bind('<FocusIn>', on_enter)
user.bind('<FocusOut>',on_leave)

Frame(frame,width=295,height=2,bg='blue').place(x=55,y=70) #полоса под ником
#--------------------------------------------------
def on_enter(e):
    passwd.delete(0,'end')
def on_leave(e):
    name=passwd.get()
    if name=='':
        passwd.insert(0,'Password')

passwd = Entry(frame,width=25,fg='black',border=0,bg="white",font=('Times New Roman',11))
passwd.place(x=55,y=100)
passwd.insert(0,'Password')
passwd.bind('<FocusIn>', on_enter)
passwd.bind('<FocusOut>', on_leave)

Frame(frame,width=295,height=2,bg='blue').place(x=55,y=120)# полоса под pass
#-----------------------------------------------реализация кнопок
Button(frame,width=39,pady=7,text='Sign in',bg='#57a1f8',fg='white',border=2,command=signin).place(x=65,y=150)

plustext=Label(frame,text="Don't use our bot early?",fg='black',bg='white',font=('Times New Roman',9))
plustext.place(x=65,y=200)

sign_up=Button(frame,width=6,text='Sign up',border=0,bg='white',cursor='hand2',fg='#57a1f8')
sign_up.place(x=200,y=200)
root.mainloop()