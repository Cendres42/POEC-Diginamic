import pprint
from bson.objectid import ObjectId
from bson.errors import InvalidId
from classes.livres import *
import sys
import curses
from curses import wrapper

class MenuSuppression():
    def __init__(self):
         self.box=False
    #
    # @brief méthode proposant un menu pour supprimer une publication
    # @return les infos pour la suppression:  id,titre,year,auteur
    #
    def precisionPubli(self):
        win = curses.initscr()
        # Par defaut, le curseur est masque
        curses.curs_set(0)
        curses.noecho()
        # Permet a ncurses d'utiliser des couleurs (optionnel)
        curses.start_color()
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLUE)
        # Efface tout l'ecran
        win.clear()
        # Rafraichit tout l'écran
        win.refresh()
        # -------------------------------------------------
        # - Etape 2: Construire une fenetre               -
        # -------------------------------------------------

        # Construction d'une fenetre centree
        width  = 50  # Largeur de la fenetre
        height = 15   # Hauteur de la fenetre
        posX = int( (curses.COLS - width) / 2)
        posY = int( (curses.LINES - height) / 2)
        self.box = curses.newwin(height, width, posY, posX)
        # utilisation partie droite clavier
        self.box.keypad(True)
        # Fixe la couleur de fond (texte noir sur fond bleu)
        self.box.bkgd(' ', curses.color_pair(2))
        # Dessine une "boite" sur les bords de la fenetre
        self.box.box()


        # Ajoute un titre a la fenetre (avec bordures)
        self.box.addch(0,2, curses.ACS_RTEE)
        self.box.addstr(0, 3, " Suppression d'une publication ")
        self.box.addstr(2, 3, "Complétez le ou les champs ci-dessous. ")
        self.box.addstr(4, 3, "Appuyez sur entrée pour valider votre saisie. ")

        self.box.addch(0, 34, curses.ACS_LTEE)

        # Rafraichit la fenetre
        self.box.refresh()

        # -------------------------------------------------
        # - Etape 3: Lecture d'une saisie                 -
        # -------------------------------------------------

        # Intitule de la zone de saisie
        self.box.addstr(6, 3, "Identifiant : ")
        self.box.addstr(8, 3, "Titre : ")
        self.box.addstr(10, 3, "Année de parution :")
        self.box.addstr(12, 3, "Nom de l'auteur :")
        # Affiche une zone noire pour montrer la zone de saisie
        self.box.addstr(7, 8, "                ", curses.color_pair(1))
        self.box.addstr(9, 8, "                ", curses.color_pair(1))
        self.box.addstr(11, 8, "                ", curses.color_pair(1))
        self.box.addstr(13, 8, "                ", curses.color_pair(1))

        self.box.move(7,8)
        curses.curs_set(1)
        self.box.refresh()

        k   = 0
        titre=""
        year=""
        auteur = ""
        select=0
        pos = 0
        id=""
        i=0
        while True:
            k = self.box.getch()
            if k in [10,13,curses.KEY_ENTER]:
                if select<2:
                    select+=1
                    pos=0
                    self.box.move(7+select*2,8)
                    win.refresh()
                    self.box.refresh()
                else:
                    break
            elif k==27:
                break
            elif k==curses.KEY_UP:
                select-=1
                pos=0
                self.box.move(7+select*2,8)
                win.refresh()
                self.box.refresh()
            elif k==curses.KEY_DOWN:
                select+=1
                pos=0
                self.box.move(7+select*2,8)
                win.refresh()
                self.box.refresh()
            elif k == curses.KEY_BACKSPACE or k == 127 or k == 8:
                if select==0:
                    if pos > 0:
                        pos -= 1
                        self.box.addstr(7, 8+pos, " ", curses.color_pair(1))
                        id= id[:-1]
                elif select==1: 
                    if pos > 0:
                        pos -= 1
                        self.box.addstr(9, 8+pos, " ", curses.color_pair(1))
                        titre = titre[:-1]
                elif select==2:
                    if pos > 0:
                        pos -= 1
                        self.box.addstr(11, 8+pos, " ", curses.color_pair(1))
                        year = year[:-1]
                elif select==3:
                    if pos > 0:
                        pos -= 1
                        self.box.addstr(13, 8+pos, " ", curses.color_pair(1))
                        auteur = auteur[:-1]
            elif pos < 30:
                if select==0:
                    self.box.addch(7, 8+pos, k, curses.color_pair(1))
                    id += chr(k)
                    pos += 1
                elif select==1:
                    self.box.addch(9, 8+pos, k, curses.color_pair(1))
                    titre += chr(k)
                    pos += 1
                elif select==2:
                    self.box.addch(11, 8+pos, k, curses.color_pair(1))
                    year += chr(k)
                    pos += 1	
                elif select==3:
                    self.box.addch(13, 8+pos, k, curses.color_pair(1))
                    auteur += chr(k)    
                    pos += 1	
        if year!="":
            year=int(year)	
        return id,titre,year,auteur
#
# @brief fonction qui permet la suppression d'une publication à partir des infos entrées
# @param la base de donnée "bibli"
#  
def supprimerPubli(bibli):
    nbSup=0
    menu=MenuSuppression()
    id,titre,year,auteur=menu.precisionPubli()
    if titre=="" and year=="" and auteur=="":
        nbSup=bibli.removeMediabyID(id)
    elif titre=="" and year=="" and id=="":
        nbSup=bibli.removeMediabyAuteur(auteur)
    elif id=="" and year=="" and auteur=="":
        nbSup=bibli.removeMediabyTitre(titre)
    elif titre=="" and id=="":
        nbSup=bibli.removeMediabyMulti(auteur,year)
    if nbSup==0:
        print("Aucune publication n'a été supprimée")


   