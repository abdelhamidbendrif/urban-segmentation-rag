"""Local desktop interface: inspect each segmentation stage."""
import queue
import sys
import threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'moteur'))
from experience import Experience, window_graph
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import LineCollection


class Application:
    def __init__(self, root):
        self.root=root
        self.engine=None
        self.busy=False
        self.events=queue.Queue()
        root.title('Comprendre la segmentation urbaine')
        root.geometry('1180x820')
        root.minsize(960,720)
        outer=ttk.Frame(root,padding=16);outer.pack(fill='both',expand=True)
        ttk.Label(outer,text='Image → couleurs → regions → graphe → fusion',font=('Segoe UI',19,'bold')).pack(anchor='w')
        ttk.Label(outer,text='But : reduire les fragments visuels. Le programme ne nomme pas les routes ou les batiments.').pack(anchor='w',pady=(3,12))
        controls=ttk.Frame(outer);controls.pack(fill='x')
        self.choose=ttk.Button(controls,text='Choisir une image',command=self.choose_image);self.choose.pack(side='left')
        ttk.Label(controls,text='   K :').pack(side='left')
        self.k=tk.StringVar(value='8')
        self.kbox=ttk.Combobox(controls,textvariable=self.k,values=[4,6,8,12,16],width=4,state='readonly');self.kbox.pack(side='left')
        self.kbox.bind('<<ComboboxSelected>>',self.reset)
        ttk.Label(controls,text='   Taille max :').pack(side='left')
        self.size=tk.StringVar(value='480')
        self.sizebox=ttk.Combobox(controls,textvariable=self.size,values=[480,960],width=5,state='readonly');self.sizebox.pack(side='left')
        self.sizebox.bind('<<ComboboxSelected>>',self.resize_image)
        ttk.Label(controls,text='   Seuil T :').pack(side='left')
        self.t=tk.StringVar(value='0.2')
        self.tbox=ttk.Combobox(controls,textvariable=self.t,values=['0.05','0.1','0.15','0.2','0.3'],width=5,state='readonly');self.tbox.pack(side='left')
        self.file_label=ttk.Label(outer,text='');self.file_label.pack(anchor='w',pady=8)
        bar=ttk.Frame(outer);bar.pack(fill='x',pady=(0,10))
        self.buttons=[]
        for stage,title in [(1,'1. K-means'),(2,'2. Regions + RAG'),(3,'3. Fusion Beveridge')]:
            button=ttk.Button(bar,text=title,command=lambda n=stage:self.run(n));button.pack(side='left',padx=(0,8));self.buttons.append(button)
        self.export=ttk.Button(bar,text='Sauvegarder le resultat',command=self.save);self.export.pack(side='right')
        self.progress=ttk.Progressbar(outer,mode='indeterminate');self.progress.pack(fill='x')
        self.explanation=tk.StringVar()
        ttk.Label(outer,textvariable=self.explanation,wraplength=1080,font=('Segoe UI',11),justify='left').pack(fill='x',pady=10)
        self.notebook=ttk.Notebook(outer);self.notebook.pack(fill='both',expand=True)
        images=ttk.Frame(self.notebook,padding=8);self.notebook.add(images,text='Images')
        self.images=[];self.image_titles=[]
        for title in ['Image originale','Resultat de l etape']:
            frame=ttk.Frame(images);frame.pack(side='left',fill='both',expand=True)
            heading=ttk.Label(frame,text=title,font=('Segoe UI',12,'bold'));heading.pack(pady=8);self.image_titles.append(heading)
            panel=ttk.Label(frame,anchor='center');panel.pack(fill='both',expand=True);self.images.append(panel)
        graph_tab=ttk.Frame(self.notebook);self.notebook.add(graph_tab,text='RAG : zoom explicatif')
        self.figure=Figure(figsize=(10,4.4),dpi=100,layout='constrained')
        self.canvas=FigureCanvasTkAgg(self.figure,master=graph_tab);self.canvas.get_tk_widget().pack(fill='both',expand=True)
        ttk.Label(graph_tab,text='Points jaunes = regions ; traits = voisinage. Ce zoom n est pas le graphe complet.').pack(pady=8)
        measures=ttk.Frame(self.notebook,padding=12);self.notebook.add(measures,text='Mesures')
        self.table=ttk.Treeview(measures,columns=['stage','groups','regions','edges'],show='headings',height=6)
        for key,title in [('stage','Etape'),('groups','Groupes de couleurs'),('regions','Regions (image complete)'),('edges','Liens du RAG complet')]:
            self.table.heading(key,text=title);self.table.column(key,width=210)
        self.table.pack(fill='x')
        ttk.Label(measures,text='Un nombre de regions plus faible ne prouve pas une meilleure precision.\n480 px = essai rapide ; 960 px = taille du rapport. Les nombres changent avec la taille.',wraplength=1000).pack(anchor='w',pady=20)
        self.status=tk.StringVar(value='Pret')
        ttk.Label(outer,textvariable=self.status).pack(anchor='w',pady=(8,0))
        first=next(iter(sorted(f for f in (ROOT/'images').glob('*') if f.suffix.lower() in ('.png','.jpg','.jpeg','.bmp'))),None)
        if first:self.load(first)
        else:self.file_label.configure(text='Aucune image dans images/ : cliquer sur Choisir une image.')
        root.after(100,self.poll)

    def show_image(self,index,array,title):
        im=Image.fromarray(array);im.thumbnail((530,410))
        picture=ImageTk.PhotoImage(im)
        self.images[index].configure(image=picture)
        self.images[index].image=picture
        self.image_titles[index].configure(text=title)

    def load(self,path):
        try:
            self.engine=Experience(path,int(self.size.get()))
            h,w=self.engine.rgb.shape[:2]
            self.file_label.configure(text=f'{Path(path).name} | image de calcul : {w} x {h} pixels')
            self.reset()
        except Exception as error:
            messagebox.showerror('Image',str(error))

    def choose_image(self):
        path=filedialog.askopenfilename(filetypes=[('Images','*.png *.jpg *.jpeg *.bmp'),('Tous','*.*')])
        if path:self.load(path)

    def resize_image(self,event=None):
        if self.engine:self.load(self.engine.path)

    def reset(self,event=None):
        if not self.engine:return
        self.engine.clear()
        for i in range(2):self.show_image(i,self.engine.rgb,'Image originale' if i==0 else 'En attente de K-means')
        self.explanation.set('Commencer par 1. K-means : on regroupe les pixels de couleurs proches. K est le nombre de groupes de couleurs, pas le nombre de regions.')
        self.status.set('Pret : choisir K puis lancer l etape 1.')
        self.figure.clear();self.canvas.draw_idle();self.refresh()

    def refresh(self):
        for row in self.table.get_children():self.table.delete(row)
        e=self.engine
        if e and e.clusters is not None:self.table.insert('','end',values=['K-means',e.k,'pas encore calcule','-'])
        if e and e.base is not None:self.table.insert('','end',values=['Regions + RAG',e.k,len(e.base),e.base.number_of_edges()])
        if e and e.final is not None:self.table.insert('','end',values=[f'Beveridge T={e.threshold}', '-',len(e.final),e.final.number_of_edges()])
        states=[not self.busy,not self.busy and e is not None and e.clusters is not None,not self.busy and e is not None and e.base is not None]
        for button,enabled in zip(self.buttons,states):button.configure(state='normal' if enabled else 'disabled')
        self.export.configure(state='normal' if not self.busy and e is not None and e.final is not None else 'disabled')
        self.choose.configure(state='disabled' if self.busy else 'normal')
        for box in [self.kbox,self.sizebox,self.tbox]:box.configure(state='disabled' if self.busy else 'readonly')

    def run(self,stage):
        if self.busy or self.engine is None:return
        k=int(self.k.get());threshold=float(self.t.get())
        self.busy=True;self.refresh();self.progress.start(12)
        self.status.set('Calcul en cours... La fenetre reste utilisable ; attendre la fin de cette etape.')
        def worker():
            try:
                result=self.engine.kmeans(k) if stage==1 else self.engine.rag() if stage==2 else self.engine.fusion(threshold)
                self.events.put((stage,result,None))
            except Exception as error:self.events.put((stage,None,str(error)))
        threading.Thread(target=worker,daemon=True).start()

    def poll(self):
        try:
            stage,result,error=self.events.get_nowait()
            self.busy=False;self.progress.stop()
            if error:
                self.status.set('Erreur : '+error);messagebox.showerror('Calcul',error)
            else:
                e=self.engine
                if stage==1:
                    title=f'K-means : {e.k} groupes de couleurs'
                    text=f'Chaque pixel prend la couleur moyenne de son groupe. Avec K={e.k}, on ne connait pas encore le nombre de regions. Cliquer sur 2 pour separer les zones connectees.'
                    self.figure.clear();self.canvas.draw_idle()
                elif stage==2:
                    title=f'{len(e.base):,} regions connectees'
                    text=f'Les zones separees deviennent des regions distinctes : {len(e.base):,} regions et {e.base.number_of_edges():,} liens. Un lien signifie une frontiere commune. Voir les onglets RAG et Mesures. Puis lancer 3.'
                    self.draw_rag()
                else:
                    title=f'Beveridge : {len(e.final):,} regions'
                    text=f'Fusion verifiee : {len(e.base):,} → {len(e.final):,} regions avec T={e.threshold}. Petit score = fusion autorisee. Refaire 3 avec un autre T repart du meme RAG initial. Les regions ne sont pas des objets automatiquement reconnus.'
                    self.draw_rag()
                self.show_image(1,result,title);self.explanation.set(text);self.status.set('Etape terminee. '+title)
            self.refresh()
        except queue.Empty:pass
        self.root.after(100,self.poll)

    def draw_rag(self):
        self.figure.clear()
        for index,(labels,title) in enumerate([(self.engine.regions,'Avant fusion'),(self.engine.merged,'Apres fusion')],1):
            ax=self.figure.add_subplot(1,2,index)
            if labels is None:
                ax.text(.5,.5,'Lancer l etape 3',ha='center',va='center');ax.axis('off');continue
            rgb,points,edges=window_graph(labels,self.engine.rgb)
            ax.imshow(rgb,alpha=.75)
            ax.add_collection(LineCollection([(points[a],points[b]) for a,b in edges],colors='#285b84',linewidths=.6))
            ax.scatter(points[:,0],points[:,1],s=16,c='#ffd400',edgecolors='black',linewidths=.3)
            ax.set_title(f'{title} : {len(points)} noeuds dans le zoom',fontsize=10)
            ax.axis('off')
        self.canvas.draw_idle()

    def save(self):
        folder=ROOT/'resultats'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        try:
            self.engine.save(folder)
            self.figure.savefig(folder/'rag_zoom.png',dpi=150)
            self.status.set('Sauvegarde : '+str(folder))
            messagebox.showinfo('Sauvegarde','Resultats dans :\n'+str(folder))
        except Exception as error:messagebox.showerror('Sauvegarde',str(error))


if __name__=='__main__':
    root=tk.Tk()
    Application(root)
    root.mainloop()
