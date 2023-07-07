from tkinter import filedialog
from tkinter.ttk import *
from tkinter import *
import time
from PIL import Image
from PIL import ImageDraw
from PIL import ImageOps
from PIL import ImageFont
from PIL import ImageTk
from IPython import display
from pdfminer.layout import LAParams, LTFigure, LTImage, LTTextContainer, LTChar
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from reportlab.pdfgen.canvas import Canvas as cs
import glob
import subprocess
import os,sys,io
import matplotlib.pyplot as plt
import numpy as np
import PIL
import os
import time
import tensorflow as tf
import tensorflow_probability as tfp
import time
from io import StringIO
from tensorflow.keras import layers
Model1=r"C:\Users\dakap\Documents\Python Projects\Models\Model1"
Model2=r"C:\Users\dakap\Documents\Python Projects\Models\Model2"
Model3=r"C:\Users\dakap\Documents\Python Projects\Models\Model3"
Model4=r"C:\Users\dakap\Documents\Python Projects\Models\Model4"
Model5=r"C:\Users\dakap\Documents\Python Projects\Models\Model5"
Model6=r"C:\Users\dakap\Documents\Python Projects\Models\Model6"
Model7=r"C:\Users\dakap\Documents\Python Projects\Models\Model7"
Model8=r"C:\Users\dakap\Documents\Python Projects\Models\Model8"
Model9=r"C:\Users\dakap\Documents\Python Projects\Models\Model9"
Model0=r"C:\Users\dakap\Documents\Python Projects\Models\Model0"
ModelA=r"C:\Users\dakap\Documents\Python Projects\Models\ModelA"
ModelB=r"C:\Users\dakap\Documents\Python Projects\Models\ModelB"
ModelC=r"C:\Users\dakap\Documents\Python Projects\Models\ModelC"
ModelD=r"C:\Users\dakap\Documents\Python Projects\Models\ModelD"
ModelE=r"C:\Users\dakap\Documents\Python Projects\Models\ModelE"
ModelF=r"C:\Users\dakap\Documents\Python Projects\Models\ModelF"
ModelG=r"C:\Users\dakap\Documents\Python Projects\Models\ModelG"
ModelH=r"C:\Users\dakap\Documents\Python Projects\Models\ModelH"
ModelI=r"C:\Users\dakap\Documents\Python Projects\Models\ModelI"
ModelJ=r"C:\Users\dakap\Documents\Python Projects\Models\ModelJ"
ModelK=r"C:\Users\dakap\Documents\Python Projects\Models\ModelK"
ModelL=r"C:\Users\dakap\Documents\Python Projects\Models\ModelL"
ModelM=r"C:\Users\dakap\Documents\Python Projects\Models\ModelM"
ModelN=r"C:\Users\dakap\Documents\Python Projects\Models\ModelN"
ModelO=r"C:\Users\dakap\Documents\Python Projects\Models\ModelO"
ModelP=r"C:\Users\dakap\Documents\Python Projects\Models\ModelP"
ModelQ=r"C:\Users\dakap\Documents\Python Projects\Models\ModelQ"
ModelR=r"C:\Users\dakap\Documents\Python Projects\Models\ModelR"
ModelS=r"C:\Users\dakap\Documents\Python Projects\Models\ModelS"
ModelT=r"C:\Users\dakap\Documents\Python Projects\Models\ModelT"
ModelU=r"C:\Users\dakap\Documents\Python Projects\Models\ModelU"
ModelV=r"C:\Users\dakap\Documents\Python Projects\Models\ModelV"
ModelW=r"C:\Users\dakap\Documents\Python Projects\Models\ModelW"
ModelX=r"C:\Users\dakap\Documents\Python Projects\Models\ModelX"
ModelY=r"C:\Users\dakap\Documents\Python Projects\Models\ModelY"
ModelZ=r"C:\Users\dakap\Documents\Python Projects\Models\ModelZ"
ModelLA=r"C:\Users\dakap\Documents\Python Projects\Models\ModelLA"
ModelLB=r"C:\Users\dakap\Documents\Python Projects\Models\ModelLB"
ModelLC=r"C:\Users\dakap\Documents\Python Projects\Models\ModelLC"
ModelLD=r"C:\Users\dakap\Documents\Python Projects\Models\ModelLD"
ModelLE=r"C:\Users\dakap\Documents\Python Projects\Models\ModelLE"
ModelLF=r"C:\Users\dakap\Documents\Python Projects\Models\ModelLF"
ModelLG=r"C:\Users\dakap\Documents\Python Projects\Models\ModelLG"
ModelLH=r"C:\Users\dakap\Documents\Python Projects\Models\ModelLH"
ModelLI=r"C:\Users\dakap\Documents\Python Projects\Models\ModelLI"
ModelLJ=r"C:\Users\dakap\Documents\Python Projects\Models\ModelLJ"
ModelLK=r"C:\Users\dakap\Documents\Python Projects\Models\ModelLK"
ModelLL=r"C:\Users\dakap\Documents\Python Projects\Models\ModelLL"
ModelLM=r"C:\Users\dakap\Documents\Python Projects\Models\ModelLM"
ModelLN=r"C:\Users\dakap\Documents\Python Projects\Models\ModelLN"
ModelLO=r"C:\Users\dakap\Documents\Python Projects\Models\ModelLO"
ModelLP=r"C:\Users\dakap\Documents\Python Projects\Models\ModelLP"
ModelLQ=r"C:\Users\dakap\Documents\Python Projects\Models\ModelLQ"
ModelLR=r"C:\Users\dakap\Documents\Python Projects\Models\ModelLR"
ModelLS=r"C:\Users\dakap\Documents\Python Projects\Models\ModelLS"
ModelLT=r"C:\Users\dakap\Documents\Python Projects\Models\ModelLT"
ModelLU=r"C:\Users\dakap\Documents\Python Projects\Models\ModelLU"
ModelLV=r"C:\Users\dakap\Documents\Python Projects\Models\ModelLV"
ModelLW=r"C:\Users\dakap\Documents\Python Projects\Models\ModelLW"
ModelLX=r"C:\Users\dakap\Documents\Python Projects\Models\ModelLX"
ModelLY=r"C:\Users\dakap\Documents\Python Projects\Models\ModelLY"
ModelLZ=r"C:\Users\dakap\Documents\Python Projects\Models\ModelLZ"
ModelList=[Model1,Model2,Model3,Model4,Model5,Model6,Model7,Model8,Model9,Model0,ModelA,ModelB,ModelC,ModelD,ModelE,ModelF,ModelG,ModelH,ModelI,ModelJ,ModelK,ModelL,ModelM,ModelN,ModelO,ModelP,ModelQ,ModelR,ModelS,ModelT,ModelU,ModelV,ModelW,ModelX,ModelY,ModelZ,ModelLA,ModelLB,ModelLC,ModelLD,ModelLE,ModelLF,ModelLG,ModelLH,ModelLI,ModelLJ,ModelLK,ModelLL,ModelLM,ModelLN,ModelLO,ModelLP,ModelLQ,ModelLR,ModelLS,ModelLT,ModelLU,ModelLV,ModelLW,ModelLX,ModelLY,ModelLZ]
LetterList=['1','2','3','4','5','6','7','8','9','0','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
noise_dim = 100
num_examples_to_generate = 1
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((8, 8, 256)))
    assert model.output_shape == (None, 8, 8, 256)  
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 32, 32, 1)
    return model
charlist=[]
ImageList=[]
ImagecoordL=[]
PrimList=[]
def PDFReader(input1):
    label_file_explorer.configure(text="File Opened: "+input1)
    label_file_explorer.grid(column = 1, row = 2) 
    PrimString=""
    fp = open(input1, 'rb')
    rmanager = PDFResourceManager()
    laparams = LAParams()
    device = PDFPageAggregator(rmanager, laparams=laparams)
    interpreter = PDFPageInterpreter(rmanager, device)
    pages = PDFPage.get_pages(fp)
    pagecounter=0
    for page in pages:
        pagecounter+=1
        interpreter.process_page(page)
        layout = device.get_result()
        for lobj in layout:
            if isinstance(lobj, LTFigure):
                for n in lobj:
                    print(n)
                    if isinstance(lobj, LTFigure):
                        x1, y1, x2, y2= n.bbox[0], n.bbox[1],n.bbox[2], n.bbox[3]
                        ImagecoordL.append([x1, y1, x2, y2,pagecounter])
                        image = Image.frombytes('RGB', n.srcsize, n.stream.get_data(), 'raw')
                        with open('temp' + '.jpg', 'wb') as fp:
                            image.save(fp)
                        image.close()
                        #im =Image.frombytes(mode="1", data = n.stream.data, size = n.srcsize,decoder_name='raw') 
                        #im.save("temp1.jpg")
                        print('At ('+str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+') Image ')
                        image2=Image.open('temp.jpg')
                        ImageList.append(image2)
            if isinstance(lobj, LTTextContainer):
                for text_line in lobj:
                    for character in text_line:
                        if isinstance(character, LTChar):
                            x1, y1, x2, y2, text= character.bbox[0], character.bbox[1],character.bbox[2], character.bbox[3],character.get_text()
                            if text.islower():
                                if text=="a" or text=="c" or text=="e" or text=="m" or text=="n" or text=="o" or text=="r" or text=="s" or text=="u" or text=="v"or text=="w" or text=="x" or text=="z":
                                    y1=y1+(y2-y1)*0.3   
                                if text=="g" or text=="j" or text=="p" or text=="q":
                                    y1=y1-(y2-y1)*0.2 
                                    y2=y2-(y2-y1)*0.2 
                            PrimList.append([x1+0.5, y1, x2-0.5, y2, text,pagecounter])
                            #print(character.set_bbox)
                            charlist.append(text)
                            PrimString=PrimString+text
                            print('At ('+str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+') is text: '+text)
    print(PrimString)
def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img
def generatePDF():
    if not PrimList:
        pass
        #label_file_explorer = Label(window,text = "No File Selected", width = 100, height = 4, fg = "black")
        #label_file_explorer.grid(column = 1, row = 2) 
    else:
        canvas = cs("Final.pdf")
        seed = tf.random.normal([len(charlist)+20, noise_dim])
        currentpage=1
        for i in range (0, len(charlist)):
            label_file_explorer = Label(window,text = "Processing", width = 100, height = 4, fg = "black")
            label_file_explorer.grid(column = 1, row = 2) 
            #if i%5==0:
            progress['value'] = i/len(charlist)*100
            window.update()
            progress.grid(column = 1, row = 6) 
            print(PrimList[i][5])
            if "\\"==(charlist[i]):
                canvas.drawImage('BackSlash.png',PrimList[i][0],PrimList[i][3],width=PrimList[i][2]-PrimList[i][0],height=PrimList[i][3]-PrimList[i][1])
            elif charlist[i]==",":
                canvas.drawImage('comma.png',PrimList[i][0],PrimList[i][3]-(PrimList[i][3]-PrimList[i][1])/2,width=(PrimList[i][2]-PrimList[i][0]),height=(PrimList[i][3]-PrimList[i][1]), mask='auto')
            elif charlist[i].isspace():
                canvas.drawImage('Space.png',PrimList[i][0],PrimList[i][3],width=(PrimList[i][2]-PrimList[i][0])*2,height=PrimList[i][3]-PrimList[i][1], mask='auto')
            elif charlist[i]==".":
                canvas.drawImage('Period.png',PrimList[i][0],PrimList[i][3]-(PrimList[i][3]-PrimList[i][1])/2,width=(PrimList[i][2]-PrimList[i][0]),height=(PrimList[i][3]-PrimList[i][1]), mask='auto')
            elif charlist[i]=="!":
                canvas.drawImage('Exclamation.png',PrimList[i][0],PrimList[i][3],width=PrimList[i][2]-PrimList[i][0],height=PrimList[i][3]-PrimList[i][1], mask='auto')
            elif charlist[i]=="?":
                canvas.drawImage('Question.png',PrimList[i][0],PrimList[i][3],width=PrimList[i][2]-PrimList[i][0],height=PrimList[i][3]-PrimList[i][1], mask='auto')
            elif charlist[i]=="@":
                canvas.drawImage('@.png',PrimList[i][0],PrimList[i][3],width=PrimList[i][2]-PrimList[i][0],height=PrimList[i][3]-PrimList[i][1], mask='auto')
            elif charlist[i]=="#":
                canvas.drawImage('#.png',PrimList[i][0],PrimList[i][3],width=PrimList[i][2]-PrimList[i][0],height=PrimList[i][3]-PrimList[i][1], mask='auto')
            elif charlist[i]=="$":
                canvas.ddrawImage('$.png',PrimList[i][0],PrimList[i][3],width=PrimList[i][2]-PrimList[i][0],height=PrimList[i][3]-PrimList[i][1], mask='auto')
            elif charlist[i]=="%":
                canvas.drawImage('%.png',PrimList[i][0],PrimList[i][3],width=PrimList[i][2]-PrimList[i][0],height=PrimList[i][3]-PrimList[i][1], mask='auto')
            elif charlist[i]=="^":
                canvas.drawImage('^.png',PrimList[i][0],PrimList[i][3],width=PrimList[i][2]-PrimList[i][0],height=PrimList[i][3]-PrimList[i][1], mask='auto')
            elif charlist[i]=="%":
                canvas.drawImage('%.png',PrimList[i][0],PrimList[i][3],width=PrimList[i][2]-PrimList[i][0],height=PrimList[i][3]-PrimList[i][1], mask='auto')
            elif charlist[i]=="*":
                canvas.drawImage('multi.png',PrimList[i][0],PrimList[i][3],width=PrimList[i][2]-PrimList[i][0],height=PrimList[i][3]-PrimList[i][1], mask='auto')
            elif charlist[i]=="(":
                canvas.drawImage('(.png',PrimList[i][0],PrimList[i][3],width=PrimList[i][2]-PrimList[i][0],height=PrimList[i][3]-PrimList[i][1], mask='auto')
            elif charlist[i]==")":
                canvas.drawImage(').png',PrimList[i][0],PrimList[i][3],width=PrimList[i][2]-PrimList[i][0],height=PrimList[i][3]-PrimList[i][1], mask='auto')
            elif charlist[i]=="-":
                canvas.drawImage('- and _.png',PrimList[i][0],PrimList[i][3],width=PrimList[i][2]-PrimList[i][0],height=PrimList[i][3]-PrimList[i][1], mask='auto')
            elif charlist[i]=="_":
                canvas.drawImage('- and _.png.png',PrimList[i][0],PrimList[i][3],width=PrimList[i][2]-PrimList[i][0],height=PrimList[i][3]-PrimList[i][1], mask='auto')
            elif charlist[i]=="+":
                canvas.drawImage('+.png',PrimList[i][0],PrimList[i][3],width=PrimList[i][2]-PrimList[i][0],height=PrimList[i][3]-PrimList[i][1], mask='auto')
            elif charlist[i]=="=":
                canvas.drawImage('=.png',PrimList[i][0],PrimList[i][3],width=PrimList[i][2]-PrimList[i][0],height=PrimList[i][3]-PrimList[i][1], mask='auto')
            elif charlist[i]=="{":
                canvas.drawImage('{.png',PrimList[i][0],PrimList[i][3],width=PrimList[i][2]-PrimList[i][0],height=PrimList[i][3]-PrimList[i][1], mask='auto')
            elif charlist[i]=="}":
                canvas.drawImage('}.png',PrimList[i][0],PrimList[i][3],width=PrimList[i][2]-PrimList[i][0],height=PrimList[i][3]-PrimList[i][1], mask='auto')
            elif charlist[i]=="[":
                canvas.drawImage('[.png',PrimList[i][0],PrimList[i][3],width=PrimList[i][2]-PrimList[i][0],height=PrimList[i][3]-PrimList[i][1], mask='auto')
            elif charlist[i]=="]":
                canvas.drawImage('].png',PrimList[i][0],PrimList[i][3],width=PrimList[i][2]-PrimList[i][0],height=PrimList[i][3]-PrimList[i][1], mask='auto')
            elif charlist[i]==":":
                canvas.drawImage('colon.png',PrimList[i][0],PrimList[i][3],width=PrimList[i][2]-PrimList[i][0],height=PrimList[i][3]-PrimList[i][1], mask='auto')
            elif charlist[i]==";":
                canvas.drawImage(';.png',PrimList[i][0],PrimList[i][3],width=PrimList[i][2]-PrimList[i][0],height=PrimList[i][3]-PrimList[i][1], mask='auto')
            elif "“"==(charlist[i]) or "”"==(charlist[i]) or "\""==(charlist[i]):
                canvas.drawImage('Quote.png',PrimList[i][0],PrimList[i][3]+(PrimList[i][3]-PrimList[i][1])/6,width=(PrimList[i][2]-PrimList[i][0])*2,height=(PrimList[i][3]-PrimList[i][1])*2, mask='auto')
            elif "\'"==(charlist[i]) or "‘"==(charlist[i]) or "’"==(charlist[i]):
                canvas.drawImage('HQ.png',PrimList[i][0],PrimList[i][3]+(PrimList[i][3]-PrimList[i][1])/5,width=(PrimList[i][2]-PrimList[i][0])*2,height=(PrimList[i][3]-PrimList[i][1])*2, mask='auto')
            elif charlist[i]=="<":
                canvas.drawImage('Left.png',PrimList[i][0],PrimList[i][3],width=PrimList[i][2]-PrimList[i][0],height=PrimList[i][3]-PrimList[i][1], mask='auto')
            elif charlist[i]==">":
                canvas.drawImage('Right.png',PrimList[i][0],PrimList[i][3],width=PrimList[i][2]-PrimList[i][0],height=PrimList[i][3]-PrimList[i][1], mask='auto')
            elif "/"==(charlist[i]):
                canvas.drawImage('Slash.png',PrimList[i][0],PrimList[i][3],width=PrimList[i][2]-PrimList[i][0],height=PrimList[i][3]-PrimList[i][1], mask='auto')
            elif "|"==(charlist[i]):    
                canvas.drawImage('Vertical.png',PrimList[i][0],PrimList[i][3],width=PrimList[i][2]-PrimList[i][0],height=PrimList[i][3]-PrimList[i][1], mask='auto')
            else:
                searcher=0
                try:
                    searcher=LetterList.index(charlist[i].upper()) 
                except:
                    searcher=LetterList.index(charlist[1].upper()) 
                if charlist[i].islower():
                    searcher=searcher+26
                #imgL = Image.new('RGBA', (100, 100), (255, 255, 255,255))
                #fnt = ImageFont.truetype("arial.ttf", 30)
                #d = ImageDraw.Draw(imgL)
                #text_width, text_height = d.textsize(str(charlist[i]),font=fnt)
                #d.text(((100-text_width)/2,(100-text_height)/2), str(charlist[i]), fill=(255, 0, 0), font=fnt)
                #((100-text_width)/2,(100-text_height)/2)
                #imgL.save('temp1.png')
                generator = make_generator_model()
                generator_optimizer = tf.keras.optimizers.Adam(1e-4)    
                checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,generator=generator)   
                checkpoint.restore(tf.train.latest_checkpoint(ModelList[searcher])).expect_partial()
                #checkpoint.restore(tf.train.latest_checkpoint(Model0)).expect_partial()
                predictions = generator(seed, training=False)
                fig = plt.figure(figsize=(1, 1))
                #for j in range(predictions.shape[0]):
                plt.subplot(1, 1, 0+1)
                plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
                plt.axis('off')
                image_data = np.asarray(fig2img(fig))
                image_data_bw = image_data.take(2, axis=2)
                non_empty_columns = np.where(image_data_bw.mean(axis=0)<253)[0]
                non_empty_rows = np.where(image_data_bw.mean(axis=1)<253)[0]
                cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
                image_data_new = image_data[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1 , :]
                new_image = Image.fromarray(image_data_new)
                print(searcher) 
                if PrimList[i][5]>currentpage:
                    print("--->"+str(len(ImagecoordL)))
                    for i in range (0,len(ImagecoordL)):
                        if ImagecoordL[i][4]==currentpage:
                            #image_data2 = np.asarray(ImageList[i].matrix)
                            #print(ImageList[i].matrix)
                            print('xxx'+str(ImagecoordL[i][3]))
                            canvas.drawInlineImage(ImageList[i],ImagecoordL[i][0],ImagecoordL[i][1],width=(ImagecoordL[i][2]-ImagecoordL[i][0]),height=(ImagecoordL[i][3]-ImagecoordL[i][1]))
                    currentpage+=1
                    canvas.showPage()
                canvas.drawInlineImage(new_image,PrimList[i][0],PrimList[i][3],width=(PrimList[i][2]-PrimList[i][0]),height=(PrimList[i][3]-PrimList[i][1]), anchor='c')
            #canvas.drawImage('temp.png',PrimList[i][0],PrimList[i][3])
            plt.close('all')
            #os.remove(temp.png)
        label_file_explorer = Label(window,text = "Done!", width = 100, height = 4, fg = "black")
        label_file_explorer.grid(column = 1, row = 2) 
        canvas.save()
        subprocess.Popen(['Final.pdf'], shell=True)
def browseFiles():
    input1 = filedialog.askopenfilename(initialdir = "/",title = "Select a File", filetypes = (("PDF Files", "*.pdf*"),("all files", "*.*")))
    label_file_explorer.configure(text="File Opened: "+input1)
    label_file_explorer.grid(column = 1, row = 2) 
    if input1=="":
        pass
    else:
        PDFReader(input1)
def center(win):
    win.update_idletasks()
    width = win.winfo_width()
    frm_width = win.winfo_rootx() - win.winfo_x()
    win_width = width + 2 * frm_width
    height = win.winfo_height()
    titlebar_height = win.winfo_rooty() - win.winfo_y()
    win_height = height + titlebar_height + frm_width
    x = win.winfo_screenwidth() // 2 - win_width // 2
    y = win.winfo_screenheight() // 2 - win_height // 2
    win.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    win.deiconify()
window = None
splash_root = Tk()
splash_root.geometry("960x560")
splash_root.title('GANCEG')
label_file_explorer=None
progress=None
splash_root.overrideredirect(True)
center(splash_root)
#splash_root.eval('tk::PlaceWindow . center')
image4=Image.open('Splash.jpg')
image4 = image4.resize((960, 560), Image. ANTIALIAS)
center_image=ImageTk.PhotoImage(image4)
image_label = Label(splash_root,image =center_image,bg = "black")
image_label.grid(column = 1, row = 1) 
def main():  
    splash_root.destroy()
    global window
    global label_file_explorer
    global progress
    window = Tk()
    window.title('GANCEG®')
    window.iconbitmap("GANCEG.ico")
    window.geometry("650x500")
    window.config(background = "black")
    progress = Progressbar(window, orient = HORIZONTAL,length = 100, mode = 'determinate')
    label_file_explorer = Label(window,
                                text = "SelectedFile",
                                width = 100, height = 4,
                                fg = "black")
    button_explore = Button(window,text = "Browse Files",command = browseFiles)
    image3=Image.open('GANCEG.jpg')
    image3 = image3.resize((300, 300), Image. ANTIALIAS)
    center_image=ImageTk.PhotoImage(image3)
    image_label = Label(window,image =center_image,bg = "black")
    button_exit = Button(window,text = "Exit",command = exit)
    button_generator = Button(window,text = "Generate PDF",command = generatePDF)
    image_label.grid(column = 1, row = 1) 
    label_file_explorer.grid(column = 1, row = 2) 
    button_explore.grid(column = 1, row = 3)
    button_generator.grid(column = 1, row = 4)
    button_exit.grid(column = 1,row = 5)
    mainloop()
splash_root.after(6000,main)
mainloop()

