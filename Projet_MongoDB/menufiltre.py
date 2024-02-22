import sys
import curses
from curses import wrapper

class MenuFiltre():
	def __init__(self):
		self.box=False
	#
    # @brief méthode proposant un menu pour choisir un filtre
    # @param le choix à un instant t
    #	
	def afficheMenuFiltre(self,typefiltre):
		if typefiltre==1:
			col=1
		else:
			col=2
		self.box.addstr(4, 3, " Par auteur ",curses.color_pair(col))
		if typefiltre==2:
			col=1
		else:
			col=2
		self.box.addstr(5,3," Par titre ",curses.color_pair(col))
		if typefiltre==3:
			col=1
		else:
			col=2
		self.box.addstr(6,3," Par année de publication ",curses.color_pair(col))
		if typefiltre==4:
			col=1
		else:
			col=2
		self.box.addstr(7,3," Revenir au menu principal ",curses.color_pair(col))
		self.box.addstr(9, 3, " Valeur de filtre : ")
        # Affiche une zone noire pour montrer la zone de saisie
		self.box.addstr(9, 23, "                ", curses.color_pair(1))
		self.box.refresh()
	#
    # @brief méthode proposant un menu pour choisir uu filtre
    # @return le choix final du filtre
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
		self.box.addstr(0, 3, " Menu filtre ")
		self.box.addch(0, 19, curses.ACS_LTEE)

		# Rafraichit la fenetre
		self.box.refresh()

		# -------------------------------------------------
		# - Etape 3: Lecture d'une saisie                 -
		# -------------------------------------------------

		# Intitule de la zone de saisie
		self.box.addstr(2, 3, " Par quoi voulez-vous filtrer vos résultats ? ")
		self.afficheMenuFiltre(1)
		self.box.refresh()

		typefiltre = 1
		while True:
			k = self.box.getch()
			if k in [27, 10,13,curses.KEY_ENTER] or typefiltre==4:
				break
			elif k==curses.KEY_UP and typefiltre>=2:
				typefiltre-=1
			elif k==curses.KEY_DOWN and typefiltre<4:
				typefiltre+=1
			self.afficheMenuFiltre(typefiltre)
		
		# Si la touche "Entree" a ete pressee
		self.box.move(9,23)
		curses.curs_set(1)
		self.box.refresh()
		win.refresh()
		k   = 0
		filtre=""
		pos = 0
		while True:
			k = self.box.getch()
			if k==27:
				break
			elif k == curses.KEY_BACKSPACE or k == 127 or k == 8:
				if pos > 0:
					pos -= 1
					self.box.addstr(9, 23+pos, " ", curses.color_pair(1))
					filtre = filtre[:-1]
			# Si la touche "Entree" a ete pressee
			elif k in [10,13, curses.KEY_ENTER]:
				break
			elif pos < 50:
				self.box.addch(9, 23+pos, k, curses.color_pair(1))
				filtre+= chr(k)
				pos += 1
		curses.endwin()
		if typefiltre==1:
			tofiltre="authors"
		elif typefiltre==2:
			tofiltre="title"
		elif typefiltre==3:
			tofiltre="year"
			filtre=int(filtre)

		return tofiltre,filtre
	
	