#sign up#
from tkinter import *
from tkinter import messagebox
import ast
def upmain(window):
    #window=Tk()
    window.title('SignUp')
    window.geometry('925x500+300+200')
    window.configure(bg="#fff")
    window.resizable(False,False)

    def signup():
        username=user.get()
        password=passwd.get()
        repassword=repass.get()
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
                window.destroy()
            except:
                file=open('database.txt','w')
                pp=str({'Username':'password'})
                file.write(pp)
                file.close
        else:
            messagebox.showerror("Invalid","both password should match")
    def sign():
        window.destroy()#if we click sign in then sign up close
    #------------------------------------------------
    #вставка фото
    img = PhotoImage(file='bitok(350x350).png')
    Label(window,image=img,bg='white').place(x=50,y=90)#label-один из классов Tk используемый для отображения img/text в окне
    #------------------------------------------------
    frame=Frame(window,width=350,height=350,bg="white")#можно посмотреть рамки и ее границы сменой цвета
    frame.place(x=480,y=50)

    heading=Label(frame,text='Sign up',fg='#57a1f8',bg='white',font=('Times New Roman',23,'bold'))
    heading.place(x=100,y=5)
    #------------------------------------------------
    def on_enter(e):
        user.delete(0,'end')
    def on_leave(e):
        name=user.get()
        if name=='':
            user.insert(0,'Username')

    user = Entry(frame,width=25,fg='black',border=0,bg="white",font=('Times New Roman',11))
    user.place(x=30,y=80)
    user.insert(0,'Username')
    user.bind('<FocusIn>', on_enter)
    user.bind('<FocusOut>',on_leave)

    Frame(frame,width=295,height=2,bg='blue').place(x=25,y=107) #полоса под ником
    #--------------------------------------------------
    def on_enter(e):
        passwd.delete(0,'end')
    def on_leave(e):
        name=passwd.get()
        if name=='':
            passwd.insert(0,'Password')

    passwd = Entry(frame,width=25,fg='black',border=0,bg="white",font=('Times New Roman',11))
    passwd.place(x=30,y=150)
    passwd.insert(0,'Password')
    passwd.bind('<FocusIn>', on_enter)
    passwd.bind('<FocusOut>', on_leave)

    Frame(frame,width=295,height=2,bg='blue').place(x=25,y=177)# полоса под pass
    #--------------------------------------------------------повтор пароля
    def on_enter(e):
        repass.delete(0,'end')
    def on_leave(e):
        name=repass.get()
        if name=='':
            repass.insert(0,'Repass please')

    repass = Entry(frame,width=25,fg='black',border=0,bg="white",font=('Times New Roman',11))
    repass.place(x=30,y=220)
    repass.insert(0,'Repass please')
    repass.bind('<FocusIn>', on_enter)
    repass.bind('<FocusOut>',on_leave)

    Frame(frame,width=295,height=2,bg='blue').place(x=25,y=247) #полоса под passwrd

    #-----------------------------------------------реализация кнопок
    Button(frame,width=39,pady=7,text='Sign up',bg='#57a1f8',fg='white',border=0,command=signup).place(x=35,y=280)

    plustext=Label(frame,text="I have an account",fg='black',bg='white',font=('Times New Roman',9))
    plustext.place(x=90,y=320)

    sign_in=Button(frame,width=6,text='Sign in',border=0,bg='white',cursor='hand2',fg='#57a1f8',command=sign)
    sign_in.place(x=200,y=320)

    window.mainloop()
upmain()