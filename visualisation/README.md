# Comment voir la solution

## Installer nodejs
```
sudo apt-get update
sudo apt-get install nodejs npm
```

## Installer les dépendances
```
cd visualisation
npm install
```

## Lancer le serveur
```
node server.js
```

## Ouvrir le navigateur
http://localhost:8081/?solution={nom_du_fichier}
example: http://localhost:8081/?solution=sol1_merged.csv
Définir une zone avc 2 click pour afficher tous les trips avec au moins un cadeau dans cette zone.


## Extra
- Voir toutes les solutions disponibles:
http://localhost:8081/solutions
- Voir un ensemble de trips par leur id:
http://localhost:8081/?solution=best_sol_897_12675059481.6.csv&trips=[1,2,3,4,5,6]