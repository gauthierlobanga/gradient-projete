# ExoPy - Solveur avance du gradient projete

ExoPy est un programme Python pedagogique et robuste pour resoudre des problemes
d'optimisation quadratique convexe en dimension 2 avec contraintes lineaires.

Il applique une methode de gradient projete avec ensemble actif, recherche
lineaire exacte et verification des conditions KKT. L'objectif est de donner un
affichage clair, mathematiquement ordonne et suffisamment detaille pour suivre
toute la resolution pas a pas.

## Probleme resolu

Le programme resout un probleme de la forme:

```text
min f(x1, x2) = a*x1^2 + b*x2^2 + c*x1 + d*x2

sous contraintes:
h_i(x) = alpha_i1*x1 + alpha_i2*x2 + beta_i >= 0
```

Avec la forme matricielle:

```text
f(x) = 1/2*x^T*Q*x + p^T*x

Q = [ 2a   0  ]
    [  0  2b  ]

p = [ c ]
    [ d ]
```

Pour garantir un minimum global dans cette version, le programme impose:

```text
a >= 0
b >= 0
```

## Fonctionnalites principales

- Saisie interactive securisee des coefficients de la fonction objectif.
- Saisie interactive des contraintes lineaires `h_i(x) >= 0`.
- Verification automatique de l'admissibilite du point initial.
- Detection des contraintes actives et inactives a chaque iteration.
- Construction de la matrice de projection:

```text
P = I - A^T*(A*A^T)^+*A
```

ou `+` designe la pseudo-inverse de Moore-Penrose.

- Gestion des contraintes actives dependantes grace a la pseudo-inverse.
- Calcul de la direction de descente projetee.
- Relachement automatique d'une contrainte active si son multiplicateur KKT est negatif.
- Recherche lineaire exacte sur:

```text
phi(lambda) = f(x + lambda*Z)
             = A*lambda^2 + B*lambda + C
```

- Calcul du pas maximal admissible avant violation d'une contrainte.
- Detection des cas non bornes dans une direction admissible.
- Affichage complet des gradients, projections, pas, blocages et variations de l'objectif.
- Diagnostic final avec point obtenu, valeur optimale, contraintes actives et multiplicateurs KKT.

## Prerequis

- Python 3.10 ou plus recent recommande.
- `numpy`.

Installation de la dependance:

```bash
pip install numpy
```

Verifier Python:

```bash
python --version
```

## Lancement

Depuis le dossier du projet:

```bash
python app.py
```

Le programme demande ensuite:

1. Les coefficients `a`, `b`, `c`, `d` de la fonction objectif.
2. Le nombre de contraintes.
3. Les coefficients `alpha1`, `alpha2`, `beta` de chaque contrainte.
4. Un point initial admissible.
5. Le nombre maximal d'iterations.

## Mode demo

Un exemple complet est integre pour tester rapidement le solveur:

```bash
python app.py --demo
```

Le probleme demo est:

```text
min f(x1, x2) = x1^2 + x2^2 - 4*x1 - 6*x2

sous:
x1 >= 0
x2 >= 0
x1 + x2 <= 5
```

Sous le format du programme, la derniere contrainte est saisie comme:

```text
-x1 - x2 + 5 >= 0
```

Mode demo silencieux:

```bash
python app.py --demo --quiet
```

Ce mode affiche uniquement le point final et la valeur de l'objectif.

## Options disponibles

```bash
python app.py --help
```

Options:

```text
--demo       Lance l'exemple integre sans saisie interactive.
--quiet      Masque les details et affiche seulement le resultat final.
--max-iter   Definit le nombre maximal d'iterations.
--tol        Definit la tolerance numerique.
```

Exemple:

```bash
python app.py --demo --max-iter 100 --tol 1e-10
```

## Exemple de saisie interactive

Pour resoudre:

```text
min f(x1, x2) = x1^2 + x2^2 - 4*x1 - 6*x2

sous:
x1 >= 0
x2 >= 0
x1 + x2 <= 5
```

Saisie:

```text
Coefficient a de x1^2: 1
Coefficient b de x2^2: 1
Coefficient c de x1: -4
Coefficient d de x2: -6

Nombre de contraintes [0]: 3

Contrainte h1
  alpha1: 1
  alpha2: 0
  beta: 0

Contrainte h2
  alpha1: 0
  alpha2: 1
  beta: 0

Contrainte h3
  alpha1: -1
  alpha2: -1
  beta: 5

x1 initial: 0
x2 initial: 0
Nombre maximal d'iterations [50]: 50
```

## Details affiches pendant la resolution

A chaque iteration, le programme affiche:

- Le point courant `Xk`.
- La valeur `f(Xk)`.
- Le gradient `grad f(Xk)`.
- La valeur de chaque contrainte `h_i(Xk)`.
- L'etat de chaque contrainte: active ou inactive.
- L'ensemble actif.
- L'ensemble de travail utilise pour la projection.
- La matrice de projection `P`.
- Le gradient projete `P*grad f`.
- Les multiplicateurs KKT quand ils sont disponibles.
- La direction de descente `Z`.
- Le produit `grad f(Xk)^T*Z`, qui confirme la descente.
- La fonction lineaire/quadratique `phi(lambda)`.
- Les contraintes pouvant bloquer le pas.
- Le pas maximal admissible `lambda_max`.
- Le pas retenu `lambda*`.
- Le nouveau point `Xk+1`.
- La variation de l'objectif.

## Methode mathematique

### 1. Ensemble actif

Une contrainte est active au point `x` si:

```text
|h_i(x)| <= tol
```

Les contraintes actives forment la matrice:

```text
A = [ alpha_1^T ]
    [ alpha_2^T ]
    [   ...     ]
```

### 2. Projection du gradient

La direction admissible est cherchee dans le noyau des contraintes actives:

```text
A*Z = 0
```

On projette le gradient avec:

```text
P = I - A^T*(A*A^T)^+*A
```

Puis:

```text
Z = -P*grad f(x)
```

Si `P*grad f(x)` est non nul, le programme avance dans cette direction.

### 3. Conditions KKT

Quand le gradient projete devient nul, le programme teste les multiplicateurs:

```text
A^T*mu = grad f(x)
```

Pour un probleme ecrit avec contraintes `h_i(x) >= 0`, les conditions KKT sont:

```text
grad f(x*) = A_active^T*mu
mu_i >= 0
h_i(x*) >= 0
mu_i*h_i(x*) = 0
```

Si tous les multiplicateurs sont positifs ou nuls, le point est optimal.

Si un multiplicateur est negatif, la contrainte correspondante est relachee et
la descente reprend avec un ensemble actif reduit.

### 4. Pas maximal admissible

Pour une contrainte inactive:

```text
h_i(x + lambda*Z) = h_i(x) + lambda*alpha_i^T*Z
```

Elle limite le pas uniquement si:

```text
alpha_i^T*Z < 0
```

Le pas de blocage est alors:

```text
lambda_i = h_i(x) / (-alpha_i^T*Z)
```

Le pas maximal admissible est:

```text
lambda_max = min(lambda_i)
```

### 5. Recherche lineaire exacte

Le programme minimise exactement:

```text
phi(lambda) = f(x + lambda*Z)
```

avec:

```text
phi(lambda) = A*lambda^2 + B*lambda + C
```

Si `A > 0`, le minimum libre est:

```text
lambda_libre = -B / (2*A)
```

Le pas final est:

```text
lambda* = min(lambda_libre, lambda_max)
```

Si la fonction decroit sans limite dans une direction admissible non bornee, le
programme signale un probleme non borne.

## Gestion d'erreurs

Le programme gere explicitement:

- Les coefficients non numeriques.
- Les valeurs infinies ou `NaN`.
- Les tolerances invalides.
- Les nombres d'iterations hors limites.
- Les contraintes avec vecteur `alpha` nul.
- Les points initiaux non admissibles.
- Les contraintes violees apres un pas.
- Les matrices de contraintes actives singulieres ou dependantes.
- Les problemes convexes non bornes sur l'ensemble admissible.
- L'interruption clavier avec `Ctrl+C`.
- Les erreurs inattendues, avec trace technique pour faciliter le debogage.

Codes de sortie:

```text
0    Resolution terminee avec succes.
1    Erreur inattendue.
2    Erreur de validation ou de resolution.
130  Operation interrompue par l'utilisateur.
```

## Structure du code

```text
app.py
README.md
```

Classes principales:

```text
ProblemeQuadratique
```

Represente la fonction objectif, calcule `f(x)`, `grad f(x)`, `Q`, `p` et les
coefficients de `phi(lambda)`.

```text
Contrainte
```

Represente une contrainte lineaire `h_i(x) >= 0`, calcule sa valeur et son pas
de blocage.

```text
GradientProjete
```

Contient le solveur: ensemble actif, projection, direction, pas, historique,
affichage detaille et diagnostic final.

```text
DirectionInfo
PasInfo
```

Objets internes qui structurent les informations affichees pendant la
resolution.

## Utilisation comme module Python

Il est possible d'importer les classes depuis un autre script:

```python
import numpy as np

from app import Contrainte, GradientProjete, ProblemeQuadratique

probleme = ProblemeQuadratique(a=1, b=1, c=-4, d=-6)
contraintes = [
    Contrainte(1, np.array([1, 0]), 0),
    Contrainte(2, np.array([0, 1]), 0),
    Contrainte(3, np.array([-1, -1]), 5),
]
x0 = np.array([0, 0], dtype=float)

solver = GradientProjete(probleme, contraintes, x0, max_iter=50, tol=1e-8)
x_opt, f_opt, historique = solver.resoudre(verbose=True)
```

## Limites actuelles

- La fonction objectif est quadratique diagonale en dimension 2.
- Les contraintes sont lineaires et de type `h_i(x) >= 0`.
- La version actuelle force la convexite simple avec `a >= 0` et `b >= 0`.
- Le solveur est concu pour la comprehension mathematique et la robustesse
  pedagogique, pas pour remplacer une bibliotheque d'optimisation industrielle.

## Conseils de modelisation

- Pour une contrainte `x1 >= 0`, utiliser:

```text
alpha1 = 1, alpha2 = 0, beta = 0
```

- Pour une contrainte `x2 >= 0`, utiliser:

```text
alpha1 = 0, alpha2 = 1, beta = 0
```

- Pour une contrainte `x1 + x2 <= 5`, reecrire:

```text
-x1 - x2 + 5 >= 0
```

donc:

```text
alpha1 = -1, alpha2 = -1, beta = 5
```

- Pour une contrainte `2*x1 - 3*x2 >= 4`, reecrire:

```text
2*x1 - 3*x2 - 4 >= 0
```

donc:

```text
alpha1 = 2, alpha2 = -3, beta = -4
```

## Depannage

### `ModuleNotFoundError: No module named 'numpy'`

Installer NumPy:

```bash
pip install numpy
```

### Point initial non admissible

Le programme affiche les contraintes violees. Choisir un point qui satisfait
toutes les contraintes `h_i(x) >= 0`.

### Le resultat s'arrete avec `ITERATIONS_MAX_ATTEINTES`

Augmenter le nombre maximal d'iterations:

```bash
python app.py --max-iter 200
```

ou reduire la tolerance:

```bash
python app.py --tol 1e-10
```

### Probleme non borne

Cela signifie que le solveur a trouve une direction admissible dans laquelle
l'objectif continue de diminuer sans limite. Il faut ajouter des contraintes ou
revoir la fonction objectif.

## Verification rapide

Commande recommandee apres modification:

```bash
python app.py --demo --quiet
```

Sortie attendue pour l'exemple demo:

```text
X* = (2, 3)^T
f(X*) = -13
Resolution terminee avec succes.
```

