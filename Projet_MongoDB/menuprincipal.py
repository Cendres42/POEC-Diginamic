import sys
import curses
from curses import wrapper

class Menuprincipal():
	def __init__(self):
		self.box=False
 	#
    # @brief méthode proposant un menu pour choisir une action
    # @param le choix à un instant t
    #	
	def afficheMenuprincipal(self,choix):
		if choix==1:
			col=1
		else:
			col=2
		self.box.addstr(4, 3, " Rechercher un livre ",curses.color_pair(col))
		if choix==2:
			col=1
		else:
			col=2
		self.box.addstr(5,3," Ajouter une publication ",curses.color_pair(col))
		if choix==3:
			col=1
		else:
			col=2
		self.box.addstr(6,3," Supprimer un livre ",curses.color_pair(col))
		if choix==4:
			col=1
		else:
			col=2
		self.box.addstr(7,3," Quitter l'application ",curses.color_pair(col))
		self.box.refresh()
	#
    # @brief méthode proposant un menu pour choisir une action
    # @return le choix final de l'action
    #
	def open(self):
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
		self.box.addstr(0, 3, " Menu principal ")
		self.box.addch(0, 19, curses.ACS_LTEE)

		# Rafraichit la fenetre
		self.box.refresh()

		# -------------------------------------------------
		# - Etape 3: Lecture d'une saisie                 -
		# -------------------------------------------------

		# Intitule de la zone de saisie
		self.box.addstr(2, 3, " Faites votre choix dans ce menu :")
		self.afficheMenuprincipal(1)
		self.box.refresh()

		# Boucle pour lire la saisie clavier du champ "nom"
		choix = 1
		while True:
			k = self.box.getch()
			if k in [27, 10,13,curses.KEY_ENTER]:
				break
			elif k==curses.KEY_UP and choix>=2:
				choix-=1
			elif k==curses.KEY_DOWN and choix<4:
				choix+=1
			self.afficheMenuprincipal(choix)
		curses.endwin()
		# Si la touche "Entree" a ete pressee
		if k in [10,13, curses.KEY_ENTER]:
			return choix
		# Wait key-press before end of program
		#key = win.getch()

		


