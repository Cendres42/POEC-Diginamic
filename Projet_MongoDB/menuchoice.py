import sys
import curses
from curses import wrapper

class Menuchoice():
	def __init__(self):
		self.box=False
	#
    # @brief méthode proposant un menu pour choisir une nouvelle action
	# @param le choix à un instant t
    #	
	def afficheMenuSec(self,choix):
		if choix==1:
			col=1
		else:
			col=2
		self.box.addstr(4, 3, " Une nouvelle sélection ",curses.color_pair(col))
		if choix==2:
			col=1
		else:
			col=2
		self.box.addstr(5,3," Fitrer vos résultats ",curses.color_pair(col))
		if choix==3:
			col=1
		else:
			col=2
		self.box.addstr(6,3," Afficher les statistiques de la bibliothèque ",curses.color_pair(col))
		if choix==4:
			col=1
		else:
			col=2
		self.box.addstr(7,3," Revenir au menu principal ",curses.color_pair(col))

	#
    # @brief méthode proposant un menu pour choisir une nouvelle action
	 # @return le choix final de l'action
    #	
	def open2(self):
		width  = 50  # Largeur de la fenetre
		height = 14   # Hauteur de la fenetre
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
		self.box.addstr(0, 3, " Menu secondaire ")
		self.box.addch(0, 20, curses.ACS_LTEE)

		# Rafraichit la fenetre
		self.box.refresh()

		# -------------------------------------------------
		# - Etape 3: Lecture d'une saisie                 -
		# -------------------------------------------------

		# Intitule de la zone de saisie
		self.box.addstr(2, 3, " Que voulez-vous faire ? ")
		self.afficheMenuSec(1)
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
			self.afficheMenuSec(choix)

		# Wait key-press before end of program
		#key = win.getch()
		curses.endwin()
        
		if choix==1:
			return 1
		elif choix==2:
			return 2
		elif choix==3:
			return 3
		elif choix==4:
			return 0

