# §6 — Le résultat P1 sur A1 (traduction française)

*Traduction française du §6 de [`docs/reports/plan_exp_lot2.md`](../plan_exp_lot2.md#6-the-a1-p1-result),
rédigée à la demande de l'utilisateur. La version anglaise reste la version de référence du projet
(`CLAUDE.md` : documentation en anglais partout, sauf demande explicite) ; en cas de divergence future
entre les deux, c'est elle qui prévaut.*

---

## 6. Le résultat P1 sur A1

Deux jobs ont tourné, couvrant à eux deux toutes les couches de `mlp_ln_mnist`, les deux façons de
construire la courbure exacte (à partir des probabilités prédites par le modèle, et à partir des
étiquettes d'entraînement), et les cinq points de contrôle (0 %, 1 %, 10 %, 50 %, 100 % de
l'entraînement) : le job `21128864` (le modèle entier) et le job `21141581` (un complément limité à
la première couche, `features.0`, nécessaire car le garde-fou de coût du job principal avait aussi
sauté la métrique la moins chère sur cette couche — voir `p1_features0_a1.sh`). Les chiffres bruts
sont sous `fisher_ref/outputs/mlp_ln_mnist/diag/seed0/<fraction>/` et
`fisher_ref/outputs/features0/mlp_ln_mnist/diag/seed0/<fraction>/`.

### 6.1 Ce qui, dans ces chiffres, est fiable

Le bruit d'échantillonnage n'a été mesuré qu'une fois, sur le réseau entier, et seulement au
dernier point de contrôle : deux estimations indépendantes construites à partir des mêmes 55 000
images d'entraînement, coupées en deux moitiés, diffèrent de 0,343 par pur hasard (intervalle à
95 % : 0,332–0,352) ; deux estimations totalement indépendantes de cette taille différeraient
d'environ 0,24. C'est l'échelle de référence pour les comparaisons faites à l'échelle du réseau
entier.

Ce bruit **n'a pas** été remesuré couche par couche — un bloc plus petit a son propre bruit
d'échantillonnage, différent, et aucun des chiffres par couche ci-dessous n'a été comparé à ce bruit
qui lui correspondrait (c'est un travail à faire séparément, pas quelque chose que ce run répond de
lui-même ; le script SLURM du job sur la première couche le dit explicitement). Ce qui rend les
constats ci-dessous crédibles malgré tout, c'est qu'ils se reproduisent, dans le même sens, aux cinq
points de contrôle et avec les deux façons de construire la référence — un bruit qui se répéterait
cinq fois de suite, dans le même sens, sur deux constructions indépendantes, est une explication
bien moins probable que le fait que l'effet soit réel.

### 6.2 HF3 — le raccourci d'AdaFisher pour sa propre diagonale coûte-t-il quelque chose ?

AdaFisher estime la courbure propre à chaque paramètre en combinant deux moyennes calculées
séparément — l'une sur les entrées d'une couche, l'autre sur ses erreurs de sortie — plutôt qu'en
suivant comment les deux évoluent ensemble. Les deux façons de la calculer ne coïncident exactement
que si rien, dans les entrées et les erreurs, n'est corrélé ; sinon, le raccourci est biaisé. Sur
A1, cette corrélation peut être mesurée directement, couche par couche :

- Sur la toute première couche (`features.0`), le raccourci ne coûte presque rien — les deux
  lectures concordent à 0,001 près en erreur relative, et cela à chacun des cinq points de contrôle
  (par exemple 0,994 contre 0,995 en fin d'entraînement). Ce n'est pas un hasard : environ 130 des
  784 pixels de MNIST sont exactement nuls sur l'ensemble du jeu de données (déjà mesuré plus tôt
  dans ce projet), et une quantité identiquement nulle ne peut être corrélée avec rien — le
  raccourci n'a donc rien à biaiser sur cette partie de la couche.
- Sur toutes les couches situées après elle, le raccourci coûte quelque chose de réel, et ce coût
  augmente au fil de l'entraînement. Sur la couche de sortie (`head`) : à l'initialisation, les deux
  lectures sont indiscernables (0,922 d'erreur relative dans les deux cas) ; en fin d'entraînement,
  la vraie diagonale affiche 0,906 contre 0,949 pour la diagonale propre à AdaFisher — un écart réel,
  et à ce stade non négligeable. Le même ordre, et la même croissance au fil de l'entraînement,
  apparaît sur `features.3`, et se maintient que la courbure soit estimée à partir des prédictions
  du modèle ou des étiquettes d'entraînement.

**HF3 est tranchée** : le raccourci n'est pas gratuit. Son coût suit la quantité de variabilité
réelle des entrées d'une couche — proche de zéro là où les entrées sont presque toujours nulles (la
première couche), réel et croissant partout ailleurs.

### 6.3 HF4 — les deux couches de normalisation ont-elles besoin du terme de couplage échelle/décalage ?

> **Erratum (lot 3, `plan_exp_lot3.md` §0.7).** Les chiffres ci-dessous restent valables ; deux de
> leurs lectures, non. (1) La lecture (d) **n'est pas** la formule du mode `diag` livré : le code
> calculait `diag(H)·diag(S)`, c'est-à-dire exactement la diagonale de la lecture (b), alors que
> `diag.py` somme l'entrée brute sur le lot et les positions avant d'élever au carré. Le lot 3 la
> renomme `hadamard_diag` et ajoute la vraie formule sous le nom `diag_py`. (2) Trois des quatre
> classements sont des théorèmes, pas des constats : la diagonale exacte est la meilleure diagonale
> au sens de Frobenius, donc (c) bat toujours (d) ; et (a) est la meilleure matrice sans termes
> croisés échelle/décalage, donc elle bat toujours (b) et (c). Seul (b) contre (c) était une
> comparaison empirique. De plus, (b) tel que codé ne garde **aucun** terme croisé : la phrase qui
> attribue l'avantage de (b) sur (c) au « terme de couplage » n'est pas étayée ; c'est la part des
> termes croisés (`cross_term_share_total`) qui mesure ce couplage.

A1 comporte deux couches de normalisation (`features.1`, `features.4`), chacune avec 64 paramètres :
32 pour une échelle par canal, 32 pour un décalage par canal. Quatre lectures simplifiées de leur
courbure exacte ont été comparées : (a) garder chacune des deux moitiés de 32 paramètres en entier,
mais sans aucun couplage entre elles ; (b) la forme réduite que la théorie sous-jacente prescrit
réellement, qui conserve un petit terme de couplage plutôt que de le supprimer ; (c) tout réduire à
un seul nombre par paramètre (une diagonale pure) ; et (d) la formule que le mode d'optimisation
`diag` utilise déjà, en production, pour ces couches.

Classées de la plus proche à la plus éloignée de la courbure réelle, l'ordre est le même sur les
deux couches, à chacun des cinq points de contrôle : **(a) > (b) > (c) > (d)**. Sur `features.1` en
fin d'entraînement, l'erreur relative est de 0,667 pour (a), 0,754 pour (b), 0,795 pour (c) et 0,821
pour (d) ; le même ordre, avec les mêmes écarts, se retrouve sur `features.4` et à chaque point de
contrôle antérieur.

Deux conséquences découlent de cet ordre. D'abord, le terme de couplage que la théorie conserve fait
un travail réel : environ deux tiers de l'amplitude totale de la courbure d'une couche de
normalisation se trouve dans le couplage entre son échelle et son décalage, mesuré directement plutôt
que supposé — le supprimer (c) est systématiquement moins bon que le conserver (b). Cela confirme la
décision prise plus tôt (lot 5) de ne pas forcer ce terme à zéro. Ensuite, et c'est nouveau : la
formule que l'optimiseur utilise réellement aujourd'hui pour ces couches (d) est, sur ce réseau
réellement entraîné, systématiquement la lecture **la moins fidèle** des quatre — moins bonne même
que la diagonale pure (c), à chaque point de contrôle sur les deux couches. Ce n'est pas un bug — on
savait déjà que les deux formules estiment des quantités différentes, pas la même de deux façons —
mais c'est désormais un fait mesuré sur laquelle des deux est la plus proche de la vérité, là où il
n'y avait auparavant qu'un argument structurel.

**HF4 est tranchée** : le terme de couplage est réel et mérite d'être conservé ; la diagonale des
couches de normalisation actuellement livrée en production est, mesurablement, la plus faible des
lectures testées.

### 6.4 Le résultat marquant, qui n'était pourtant pas l'une des deux questions posées

Le même run a aussi mesuré, à chaque point de contrôle et sur chaque couche, quelle part de la mise
à jour *idéale* des paramètres chaque approximation permettrait réellement d'obtenir si elle servait
à préconditionner un vrai gradient (la métrique « M5 »), sur toute une gamme d'intensités
d'amortissement (damping).

Le motif est le même sur chaque couche, et il est important. À l'initialisation, les bonnes
approximations structurées (K-FAC et ses raffinements) récupèrent 61 à 97 % du pas idéal, selon la
couche et l'intensité d'amortissement choisie ; les approximations diagonales — y compris celle
d'AdaFisher — n'en récupèrent que 12 à 37 %. En fin d'entraînement, sur ces mêmes couches, la qualité
de chaque approximation s'est nettement dégradée : les structurées ne récupèrent plus que 13 à 46 %
du pas idéal, et les diagonales ne récupèrent presque plus rien — une fraction de pour cent sur les
deux premières couches linéaires, 6 à 12 % sur la couche de sortie.

En d'autres termes : la courbure ne fait pas que diminuer en taille au fil de l'entraînement (son
amplitude globale chute d'environ deux ordres de grandeur entre le premier et le dernier point de
contrôle, ce qui correspond à l'aplatissement de la perte) — sa *forme* devient elle aussi bien plus
difficile à capturer pour n'importe laquelle des structures de ce projet. Une approximation presque
exacte au début de l'entraînement devient un piètre substitut à la vérité à la fin. C'est exactement
le type de changement que toute cette campagne vise à mesurer, et A1 est le premier modèle sur
lequel il a effectivement été mesuré, plutôt que simplement argumenté.

### 6.5 Deux constats secondaires à retenir

- À quel point une simplification de type Kronecker est « propre » varie fortement selon la couche.
  Sur la première couche linéaire, la meilleure simplification de ce type explique déjà l'essentiel
  de la forme du bloc (un second motif concurrent ne fait qu'environ un quart de sa taille) ; sur la
  couche de sortie, les deux motifs sont de taille bien plus proche (presque moitié-moitié), ce qui
  explique précisément pourquoi les approximations de type Kronecker s'y comportent nettement moins
  bien que sur la première couche, à chaque point de contrôle.
- La courbure ne reste pas confinée à une couche à la fois : le couplage mesuré entre la courbure de
  deux couches quelconques est important partout où il a été vérifié (0,72–0,92, sur une échelle où 1
  signifierait que les deux courbures évoluent de façon parfaitement solidaire, et 0 qu'il n'existe
  aucune relation). Aucune paire de couches de ce réseau ne peut être traitée en toute sécurité comme
  indépendante des autres.

La seule métrique nécessitant une décomposition matricielle complète, trop coûteuse pour être menée
sur chaque structure et chaque niveau d'amortissement (une divergence de type Stein, « M3 »), a été,
comme prévu, sautée sur la première couche — et **enregistrée comme sautée** plutôt que simplement
absente en silence ; elle a été menée à terme partout ailleurs.

### 6.6 Ce qui reste ouvert

- Une version par couche de la mesure de bruit du §6.1, pour pouvoir comparer les chiffres par
  couche ci-dessus à leur propre bruit d'échantillonnage plutôt qu'à celui du réseau entier.
- Le balayage complet de l'intensité d'amortissement derrière les §6.3/§6.4 existe dans les données
  brutes (la colonne `lambda_alpha` de `metrics.csv`) ; seule l'extrémité « amortissement léger » en
  est résumée ci-dessus, car c'est le régime que l'investigation antérieure de ce projet sur le blocage
  de l'auto-encodeur (`mnist_autoencoder`) avait identifié comme pratiquement pertinent.
- Les deux jobs utilisés ici ne couvrent que la graine (seed) 0. A2/A3 — les modèles du lot 3, qui ont
  effectivement du partage de paramètres — ainsi que des graines supplémentaires restent à venir.
