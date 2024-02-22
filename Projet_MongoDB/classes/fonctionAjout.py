import pprint
from bson.objectid import ObjectId
from bson.errors import InvalidId
from classes.livres import *
import sys
import curses
from curses import wrapper


class MenuAjout():
    def __init__(self):
         self.box=False
    #
    # @brief méthode proposant un menu pour l'ajout d'un publication
    # @param la base de donnée et le type de publi insérée (livre ou article)
    #
    def precisionPubli(self,bibli,choix):
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
        self.box.addstr(0, 3, " Ajout d'une publication : ")
        self.box.addstr(2, 3, "Complétez les champs ci-dessous. ")
        self.box.addstr(3, 3, "Utilisez les flèches pour naviguer. ")
        self.box.addstr(4, 3, "Appuyez sur entrée pour valider votre saisie. ")

        self.box.addch(0, 29, curses.ACS_LTEE)

        # Rafraichit la fenetre
        self.box.refresh()

        # -------------------------------------------------
        # - Etape 3: Lecture d'une saisie                 -
        # -------------------------------------------------

        # Intitule de la zone de saisie
        self.box.addstr(6, 3, "Titre : ")
        self.box.addstr(8, 3, "Année de parution :")
        self.box.addstr(10, 3, "Nom de l'auteur :")
        # Affiche une zone noire pour montrer la zone de saisie
        self.box.addstr(7, 8, "                ", curses.color_pair(1))
        self.box.addstr(9, 8, "                ", curses.color_pair(1))
        self.box.addstr(11, 8, "                ", curses.color_pair(1))

        self.box.move(7,8)
        curses.curs_set(1)
        self.box.refresh()

        # Boucle pour lire la saisie clavier du champ "nom"
        k   = 0
        titre=""
        year=""
        auteur = ""
        select=0
        pos = 0
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
                        titre = titre[:-1]
                elif select==1: 
                    if pos > 0:
                        pos -= 1
                        self.box.addstr(9, 8+pos, " ", curses.color_pair(1))
                        year = year[:-1]
                elif select==2:
                    if pos > 0:
                        pos -= 1
                        self.box.addstr(11, 8+pos, " ", curses.color_pair(1))
                        auteur = auteur[:-1]
            elif pos < 30:
                if select==0:
                    self.box.addch(7, 8+pos, k, curses.color_pair(1))
                    titre += chr(k)
                    pos += 1
                elif select==1:
                    self.box.addch(9, 8+pos, k, curses.color_pair(1))
                    year += chr(k)
                    pos += 1
                elif select==2:
                    self.box.addch(11, 8+pos, k, curses.color_pair(1))
                    auteur += chr(k)    
                    pos += 1			

            
        # Si la touche "Entree" a ete pressee
        if k in [10,13, curses.KEY_ENTER]:
            # Re-ecrit la zone noire ... pour la remettre en bleu
            for i in range(2,13):
                self.box.addstr(i, 3, "                                              ", curses.color_pair(2))

            # Dis bonjour a l'utilisateur
            self.box.addstr(2,3, "Vous avez ajouté cette publication ")
            self.box.addstr(4, 3, "Titre : ")
            self.box.addstr(6, 3, "Année de parution :")
            self.box.addstr(8, 3, "Auteur :")
            self.box.addstr(5,3,titre)
            self.box.addstr(7,3,year)
            self.box.addstr(9,3,auteur)

            self.box.refresh()
            # Attend l'appui sur une touche avant de sortir
            key = win.getch()

            if choix==1:
                bibli.createBook(titre,auteur,int(year))
            elif choix==2:
                bibli.createArticle(titre,auteur,int(year))
        # Wait key-press before end of program
        curses.endwin()
    #
    # @brief méthode proposant un menu pour choisir le type de publication à ajouter
    # @param le type de publi sélectionné à un instant t (livre ou article)
    #
    def afficheMenuAjout(self,choix):
        if choix==1:
            col=1
        else:
            col=2
        self.box.addstr(4, 3, " Un livre  ",curses.color_pair(col))
        if choix==2:
            col=1
        else:
            col=2
        self.box.addstr(5,3," Un article",curses.color_pair(col))
        if choix==3:
            col=1
        else:
            col=2
        self.box.addstr(6,3," Revenir au menu principal ",curses.color_pair(col))
        self.box.refresh()
    
    def open(self,bibli):
        win = curses.initscr()
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
        width  = 40  # Largeur de la fenetre
        height = 12   # Hauteur de la fenetre
        posX = int( (curses.COLS - width) / 2)
        posY = int( (curses.LINES - height) / 2)
        self.box = curses.newwin(height, width, posY, posX)
        self.box.keypad(True)
        # Fixe la couleur de fond (texte noir sur fond bleu)
        self.box.bkgd(' ', curses.color_pair(2))
        # Dessine une "boite" sur les bords de la fenetre
        self.box.box()

        # Ajoute un titre a la fenetre (avec bordures)
        self.box.addch(0,2, curses.ACS_RTEE)
        self.box.addstr(0, 3, " Type de publication ")
        self.box.addch(0, 19, curses.ACS_LTEE)

        # Rafraichit la fenetre
        self.box.refresh()

        # -------------------------------------------------
        # - Etape 3: Lecture d'une saisie                 -
        # -------------------------------------------------

        # Intitule de la zone de saisie
        self.box.addstr(2, 3, " Voulez-vous ajouter :")
        self.afficheMenuAjout(1)
        self.box.refresh()

        choix = 1
        while True:
            k = self.box.getch()
            if k in [27, 10,13,curses.KEY_ENTER]:
                break
            elif k==curses.KEY_UP and choix>=2:
                choix-=1
            elif k==curses.KEY_DOWN and choix<4:
                choix+=1
            self.afficheMenuAjout(choix)
        curses.endwin()
        # Si la touche "Entree" a ete pressee
        if k in [10,13, curses.KEY_ENTER]:
            self.precisionPubli(bibli,choix)
        
    