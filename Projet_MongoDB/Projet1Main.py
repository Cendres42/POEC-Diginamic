from pymongo.mongo_client import MongoClient
from classes.fonctionRecherche import *
from classes.fonctionAjout import *
from classes.livres import *

from classes.fonctionSuppression import *
from menuprincipal import *
from menurecherche import *

# création des variables liées à la base de donnnées
client = MongoClient("localhost",27017)
db=client["my-first-db"]     
coll = db["books"]
bibli=Bibliotheque(coll)
result=""
#
#@ brief fonction de menu proposant à l'utilisateur différents choix 
#
def Menu():
    while True:
        try:
            menu=Menuprincipal()
            choix=menu.open()
            if choix<=0 or choix>4:
                raise ValueError
        except ValueError:
            print("Votre choix doit être un chiffre entre 1 et 4")
        else:    
            if choix==1:
                rechercherMedia(bibli)
            elif choix==2:
                menu=MenuAjout()
                menu.open(bibli)
            elif choix==3:
                supprimerPubli(bibli)
            elif choix==4:
                break

        
Menu()
