import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import collections
#data = pd.read_json('C:/Users/Gwen/Desktop/Data/MongoDB/books.json')

# importation de la collection de la base de donnnées 
client = MongoClient("localhost",27017)
db=client["my-first-db"]     
coll = db["books2"]
data = pd.DataFrame(list(coll.find()))
x=data.iloc[0:200]

#
# @ brief fonction de création des datawiz books
#
def graph():
    result = data.groupby("year")['_id'].nunique()
    result2=data.groupby("type")['_id'].nunique()
    figure = plt.figure(figsize = (10, 10))
    plt.gcf().subplots_adjust(left = 0.1, bottom = 0.1,right = 0.9, top = 0.9, wspace = 0.4, hspace = 0.4)
    
    axes = figure.add_subplot(2, 2, 1)
    axes.set_xlabel('Années',size=6,color='darkblue')
    axes.set_title('Evolution du nombre de publications',size=10, color='darkblue')
    axes.plot(result,color="indigo",label="Nombre de publications")
    
    axes = figure.add_subplot(2, 2, 2)
    axes.set_title('Répartition des publications par type',size=10,color='darkblue')
    axes.pie(x=result2,labels=('Articles','Livres','PhD'),autopct='%.0f%%',textprops={'fontsize': 8},colors=[ "mediumaquamarine","teal",'aqua'])

    
    auteurs=[]
    for auteur in data['authors']:
        for elt in auteur:
            auteurs.append(elt)
    c = collections.Counter(auteurs)
    result3=c.most_common(5)
    h=[]
    for i in range (len(result3)):
        h.append(result3[i][0])
    v=[]
    for i in range (len(result3)):
        v.append(result3[i][1])

    axes = figure.add_subplot(2, 2, 3)
    axes.set_title('5 auteurs avec le plus de publications',size=10,color='darkblue')
    axes.bar(height= v, x=h ,color = 'darkslategrey')
    plt.xticks(rotation=90,size=5)

    tab_len_aut=[]
    for aut in data["authors"]:
        tab_len_aut.append(len(aut))
    tab_len_aut=[]
    for aut in data["authors"]:
        tab_len_aut.append(len(aut))
    dico={}
    for i in range(1,11):
        dico[i]=tab_len_aut.count(i)
    z=[]
    for elt in dico.keys():
        z.append(elt)
    w=[]
    for elt in dico.values():
        w.append(elt)
    axes = figure.add_subplot(2, 2, 4)
    axes.set_title('Nombre de publication ayant entre 1 et 10 auteurs',size=10,color='darkblue')
    axes.barh(z, w ,color = 'darkred')
 
    plt.show()
   
    
#plt.savefig('C:/Users/Gwen/Desktop/Data/Maison/'+col+piece+".png", dpi=300, format="png")
#graph()

#résalisation de stats avec mongo:
    
plop=coll.aggregate([{"$group":{"_id":"$type","nb":{"$sum":1}}}])
plop=list(plop)
for item in plop:
    if item['_id']=='Phd':
        nbPhd=item['nb']
    if item['_id']=='Book':
        nbBook=item['nb']
    if item['_id']=='Article':
        nbArticle=item['nb']
#print(nbPhd,nbBook,nbArticle)
nbMynPubliAnnuel=coll.aggregate([{"$group":{"_id":"$year","myn":{"$sum":1}}},{"$group":{"_id":"clef","moyenne":{"$avg":"$myn"}}}])
nb=0
somme=0
for item in nbMynPubliAnnuel:
    nb+=1
    #print(round(item))
    #somme+=item["myn"]
#print(round(somme/nb))
nbmynouvrageparauteur=coll.aggregate([{"$unwind":"$authors"},{"$group":{"_id":"$authors","myn":{"$sum":1}}},{"$sort":{"myn":-1}},{"$limit":5}])