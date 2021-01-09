import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("ex1data1.csv")
##print(df) pour visualiser les données mais optionnel

x=df.iloc[:, 0] #recuperer la premiere colonne dans x
y=df.iloc[:, 1] # recuperer la deuxième colonne dans y

x=x.values.reshape(96,1)
y=y.values.reshape(96, 1)
X=np.c_[x, np.ones(96)] # on ajoute un vecteur de bias de un dans la var x


'''
Définissons notre fonction de cout et le model 
'''
def model(X, theta):
    return X.dot(theta)

np.random.seed(1)#avant de commencer fixons nos valeurs aléatoire avec cette ligne de code

'''Ici on test en premier comment notre model ce comport vis-à-vis des données
avec des paramètres donnés aléatoirement
En suite on le commente et on passe a l'apprentissage de notre model
'''
##theta=np.random.randn(2, 1)
##h=model(X, theta)
##plt.scatter(x, y)
##plt.plot(x, h, c='r')
##plt.show()




" definissons la fonction de cout"

def cost_function(X, y, theta):
    m=len(y)
    return 1/(2*m)*np.sum((model(X, theta)-y)**2)

#print(cost_function(X, y, theta)) # afficher premiere ce qu'est notre erreur

''' Definition de l'algo de descente gradient
en calculant les element de theta
'''


def gradient(X, y, theta):
    m=len(y)
    return (1/m)*X.T.dot(model(X, theta)-y)


" on definit la descente gradient"

def gradient_desct(X, y, theta, alpha, iters):
    cost_hist=np.zeros(iters)
    for i in range(iters):
        theta = theta - alpha*gradient(X, y, theta)
        cost_hist[i]=cost_function(X, y, theta)
    return theta, cost_hist

"Maintenant on peut bien tester le model avec les nouveaux paramtres"

##theta_final, cost=gradient_desct(X, y, theta, alpha=0.01, iters=1500)
##
##h=model(X, theta_final) # Voila le nouveau model

#print(cost_function(X, y, theta_final)) #On peut afficher le cout pour voir

"On va ploter à nouveux nos données et la n regression linéaire"


#plt.plot(range(1000), cost, c='r') # ça on le fait juste pour le processus d'apprentissage du model


theta_f, cost = gradient_desct(X, y, np.random.randn(2, 1), alpha=0.01, iters=1000)
h=model(X, theta_f)
plt.scatter(x, y)
plt.plot(x, h, c='g')
plt.show()

"Le coefficient de determination du model afin d'évaluer la performance de notre model"
def coeff(y, h):
    u=((y-h)**2).sum()
    v=((y-y.mean())**2).sum()
    return 1-(u/v)
print(coeff(y, h))
    






