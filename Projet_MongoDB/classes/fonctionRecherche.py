import pprint
from bson.objectid import ObjectId
from bson.errors import InvalidId
from classes.livres import *
import sys
import curses
from curses import wrapper
from menurecherche import *
from menufiltre import *
from menuchoice import *
import colorama
from stat_books import *
#
# @brief fonction qui affiche les données recherchée par auteur, titre ou année de parution
# @param la base de données (bibli)
#
def rechercherMedia(bibli):
    selection = 0
    tri = 0
    sous_selection = ""
    toSkip=0
    filtre=""
    typefiltre=""
    filtre=""
    while True:
        try:
            if selection == 0:
                menu=Menurecherche()
                selection,sous_selection,tri=menu.open()
            if selection<=0 or selection>3:
                    raise ValueError
        except ValueError:
            print("Votre saisie doit être un chiffre entre 1 et 3")
            break   
        else:
            colorama.init()
            if tri==4:
                return
            if selection== 2:
                if typefiltre!="":
                    publi=bibli.findByTitle(sous_selection,tri,toSkip,typefiltre,filtre)
                else:
                    publi=bibli.findByTitle(sous_selection,tri,toSkip)
            elif selection==1:
                if typefiltre!="":
                    publi=bibli.findByAuthors(sous_selection,tri,toSkip,typefiltre,filtre)
                else:
                    publi=bibli.findByAuthors(sous_selection,tri,toSkip)
            elif selection==3:
                if typefiltre!="":
                    publi=bibli.findByYear(int(sous_selection),tri,toSkip,typefiltre,filtre)
                else:
                    publi=bibli.findByYear(int(sous_selection),tri,toSkip)
            print(publi)
           
            try:
                next=int(input("Pour afficher les 5 résultats suivants tapez 1 \n sinon tapez 2\n"))
                if next<=0 or next>2:
                    raise ValueError
            except ValueError:
                print("Vous devez saisir 1 ou 2")
            else:
                if next==1:
                    toSkip+=5
                elif next==2:
                    menu=Menuchoice()
                    x= menu.open2()
                    if x==0:
                        break
                    elif x==1:
                        selection = 0
                        tri = 0
                        sous_selection = ""
                        toSkip=0
                        typefiltre=""
                        filtre=""
                    elif x==2:
                        toSkip=0
                        menu=MenuFiltre()
                        typefiltre,filtre=menu.open()
                        print(f"-----------------------------------\n \033[46m Voici le résulat de votre filtre :\033[0m\n")
                    elif x==3:
                        graph()
                      