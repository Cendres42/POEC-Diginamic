import sys
import curses
from curses import wrapper

class Menurecherche():
	def __init__(self):
		self.box=False
	#
    # @brief méthode proposant un menu pour choisir un type de recherche
    # @param le choix à un instant t
    #
	def afficheMenuMixte(self,choix):
		if choix==1:
			col=1
		else:
			col=2
		self.box.addstr(4, 3, " Rechercher par auteur ",curses.color_pair(col))
		if choix==2:
			col=1
		else:
			col=2
		self.box.addstr(5,3," Rechercher par titre ",curses.color_pair(col))
		if choix==3:
			col=1
		else:
			col=2
		self.box.addstr(6,3," Rechercher par année de publication ",curses.color_pair(col))
		
		self.box.addstr(9, 3, " Valeur cherchée : ")
        # Affiche une zone noire pour montrer la zone de saisie
		self.box.addstr(9, 23, "                ", curses.color_pair(1))
		self.box.refresh()
	#
    # @brief méthode proposant un menu pour choisir un type de recherche
    # @return le choix final du type de recherche
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
		self.box.addstr(0, 3, " Menu recherche ")
		self.box.addch(0, 19, curses.ACS_LTEE)

		# Rafraichit la fenetre
		self.box.refresh()

		# -------------------------------------------------
		# - Etape 3: Lecture d'une saisie                 -
		# -------------------------------------------------

		# Intitule de la zone de saisie
		self.box.addstr(2, 3, " Par quel critère voulez-vous chercher ? ")
		self.afficheMenuMixte(1)
		self.box.refresh()

		choix = 1
		while True:
			k = self.box.getch()
			if k in [27, 10,13,curses.KEY_ENTER] :
				break
			elif k==curses.KEY_UP and choix>=2:
				choix-=1
			elif k==curses.KEY_DOWN and choix<4:
				choix+=1
			self.afficheMenuMixte(choix)
		
		# Si la touche "Entree" a ete pressee
		self.box.move(9,23)
		curses.curs_set(1)
		self.box.refresh()
		win.refresh()
		k   = 0
		sous_selection=""
		pos = 0
		while True:
			k = self.box.getch()
			if k==27:
				break
			elif k == curses.KEY_BACKSPACE or k == 127 or k == 8:
				if pos > 0:
					pos -= 1
					self.box.addstr(9, 23+pos, " ", curses.color_pair(1))
					sous_selection = sous_selection[:-1]
			# Si la touche "Entree" a ete pressee
			elif k in [10,13, curses.KEY_ENTER]:
				tri=self.open2()
				break
			elif pos < 50:
				self.box.addch(9, 23+pos, k, curses.color_pair(1))
				sous_selection+= chr(k)
				pos += 1
			
		win.refresh
		
		return choix,sous_selection,tri
	
	#
    # @brief méthode proposant un menu pour choisir un tri
    # @return le choix du tri à un instant t
    #
	def afficheMenuTri(self,choix):
		if choix==1:
			col=1
		else:
			col=2
		self.box.addstr(4, 3, " Trier par auteur ",curses.color_pair(col))
		if choix==2:
			col=1
		else:
			col=2
		self.box.addstr(5,3," Trier par titre ",curses.color_pair(col))
		if choix==3:
			col=1
		else:
			col=2
		self.box.addstr(6,3," Trier par année de publication ",curses.color_pair(col))
		if choix==4:
			col=1
		else:
			col=2
		self.box.addstr(7,3," Revenir au menu principal ",curses.color_pair(col))
	#
    # @brief méthode proposant un menu pour choisir un tri
	# @return le choix final du type de tri
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
		self.box.addstr(0, 3, " Menu tri ")
		self.box.addch(0, 19, curses.ACS_LTEE)

		# Rafraichit la fenetre
		self.box.refresh()

		# -------------------------------------------------
		# - Etape 3: Lecture d'une saisie                 -
		# -------------------------------------------------

		# Intitule de la zone de saisie
		self.box.addstr(2, 3, " Critère de tri de vos résultats : ")
		self.afficheMenuTri(1)
		self.box.refresh()

		choix = 1
		while True:
			k = self.box.getch()
			if k in [27, 10,13,curses.KEY_ENTER] :
				break
			elif k==curses.KEY_UP and choix>=2:
				choix-=1
			elif k==curses.KEY_DOWN and choix<4:
				choix+=1
			self.afficheMenuTri(choix)

		# Wait key-press before end of program
		#key = win.getch()
		curses.endwin()
		tri=choix
		return tri

		

