# Écrêtage plutôt qu'amortissement : étude de faisabilité

*Étude exploratoire demandée après la lecture de
[`docs/reports/plan_lambda_dominance.md`](../plan_lambda_dominance.md). Aucune modification n'a été
apportée au code du projet. Une sonde temporaire a été exécutée en local pour trancher la question
centrale ; elle vit dans le répertoire de travail temporaire de la session, pas dans le dépôt.*

*Rédigé en français à la demande explicite de l'utilisateur ; la langue de référence du projet reste
l'anglais (`CLAUDE.md`).*

---

## 1. La réponse courte

L'idée est bonne, mais elle ne répare pas ce qu'on croit qu'elle répare.

Aujourd'hui, la constante `λ` fait deux métiers à la fois :

- **métier n°1** : empêcher une division par un nombre quasi nul ;
- **métier n°2** : fixer, de fait, la taille maximale d'un pas de mise à jour.

L'écrêtage (« clipping ») fait le métier n°2 **beaucoup mieux** — avec une garantie exacte au lieu
d'une conséquence involontaire. Il ne fait pas le métier n°1, et surtout **il ne corrige pas
l'erreur d'échelle de 2 × 10⁸** identifiée dans le plan.

Conséquence, mesurée aujourd'hui et pas seulement raisonnée : si l'on branche un écrêtage réglé pour
reproduire la taille de pas actuelle, **95 % à 100 % des coordonnées sont écrêtées**, sur toutes les
couches, pour les trois modes testés. L'optimiseur cesse d'être « du momentum déguisé » pour devenir
« du momentum de signe déguisé ». On échange une dégénérescence contre une autre.

**Ce qui vaut malgré tout la peine.** Trois gains réels, qui n'ont rien à voir avec la raison
initiale :

1. **L'écrêtage échoue en sécurité.** Baisser `λ` fait exploser le pas — c'est mesuré, et
   massivement : **81 divergences** dans le balayage de `λ`, avec une perte qui grossit
   géométriquement jusqu'à 10³² avant de déborder (§5.4). Déplacer un seuil d'écrêtage ne peut pas
   faire exploser le pas, parce que le pas est borné par construction. Tout le domaine que
   l'expérience E2 n'atteint qu'au prix d'un protocole délicat (baisser le taux d'apprentissage en
   proportion de `λ`) devient explorable sans précaution. Réserve importante : l'autre moitié des
   échecs du balayage (**100 plantages** dans l'inversion matricielle, *avant* le premier pas) n'est
   pas du ressort de l'écrêtage — voir §5.4.
2. **Il fournit gratuitement le diagnostic qui manquait.** La « proportion écrêtée » est, à chaque
   pas, exactement le percentile que l'expérience E1 a coûté 32 minutes de calcul sur grappe à
   mesurer une seule fois. Chez nous elle vaut 1,00 ; la recommandation publiée pour l'optimiseur qui
   a popularisé la technique est 0,1 à 0,5.
3. **Il existe déjà dans le code ancêtre du projet, et AdaFisher l'a retiré.** Le K-FAC de référence
   embarqué dans le dépôt officiel AdaFisher applique un écrêtage global
   ([`reference_repos/AdaFisher/optimizers/kfac.py:131-146`](../../../reference_repos/AdaFisher/optimizers/kfac.py)),
   avec un paramètre `kl_clip = 0.001`. AdaFisher a gardé l'amortissement et jeté l'écrêtage.

**Ce qu'il ne faut pas en attendre.** Nulle part dans la littérature l'écrêtage ne *remplace*
l'amortissement : il vient toujours **en plus**. Le correctif de fond reste celui que le plan appelle
S1 — un amortissement proportionnel à la courbure de chaque couche — qui est d'ailleurs prescrit
noir sur blanc par deux des cinq articles sources du projet et non implémenté ici.

---

## 2. Rappel du problème, en une page

Ces cinq optimiseurs promettent de diviser chaque mise à jour par la courbure locale de la fonction
de perte : petits pas là où la perte se courbe fortement, grands pas là où elle est plate. Pour ne
pas diviser par zéro, on ajoute une petite constante `λ` avant de diviser.

La campagne de mesure a établi que `λ` est **plus grand que la courbure estimée dans toutes les
directions, de toutes les couches, de tous les réseaux, dans tous les modes** : `λ` se situe au
100ᵉ percentile dans les 105 cas mesurés. Le diviseur vaut donc `λ` partout, la division ne dépend
plus de la direction, et les cinq méthodes se ramènent à du momentum simple avec un facteur fixe.

La cause est arithmétique et bien identifiée : deux conventions héritées rétrécissent la courbure
stockée, d'un facteur ≈ 115 par facteur mémorisé (une moyenne mobile dont les coefficients ne
somment pas à 1) et d'un facteur 16 384 (le signal rétropropagé est celui d'une perte déjà moyennée
sur un lot de 128 exemples). Ensemble : environ 2 × 10⁸.

C'est ce facteur-là que la question « et si on écrêtait au lieu d'amortir ? » cherche à contourner.

---

## 3. « Écrêter » veut dire quatre choses différentes

C'est le point le plus important de ce rapport, parce que trois de ces quatre familles donnent des
réponses opposées dans notre cas.

### Famille A — Plancher sur la courbure elle-même

Remplacer `courbure + ε` par `max(courbure, ε)`. C'est la lecture la plus littérale de « écrêter au
lieu d'amortir », et c'est une vraie technique : elle laisse intactes les directions dont la courbure
est correctement estimée, là où l'ajout de `ε` déplace *toutes* les valeurs, y compris les bonnes.

**Chez nous elle ne change strictement rien, et c'est démontrable sans rien lancer.** Si `ε` est
supérieur à toute la courbure — ce qui est précisément le résultat de E1, `λ` au 100ᵉ percentile —
alors `max(courbure, ε) = ε` partout, exactement comme `courbure + ε ≈ ε` partout. Les deux formules
coïncident dans le régime dégénéré. La famille A n'a d'intérêt qu'**après** correction de l'échelle,
jamais avant.

### Famille B — Écrêtage coordonnée par coordonnée du pas (type « Sophia »)

On calcule le pas naturel `momentum / courbure`, puis on plafonne chaque coordonnée à une valeur `ρ`.

Il faut voir qu'**il s'agit toujours d'un amortissement**, sous une autre forme. L'identité est
exacte :

```
écrêter( m / h , ρ )  =  m / max( h , |m| / ρ )
```

Autrement dit, écrêter revient à mettre un plancher sous la courbure — mais un plancher qui vaut
`|m| / ρ`, donc **proportionnel à l'amplitude du gradient**, au lieu d'une constante fixe. C'est de
là que vient sa robustesse : si la perte est multipliée par mille, le plancher suit, alors que `λ`
ne suit pas.

### Famille C — Écrêtage global du pas (région de confiance)

On calcule la mise à jour complète, on mesure « de combien elle déplace le modèle » au sens de la
courbure, et si ce déplacement dépasse un budget fixé, on **réduit la mise à jour entière** par un
seul facteur scalaire. Les directions relatives sont préservées ; seule l'amplitude d'ensemble est
plafonnée.

C'est la forme retenue par K-FAC et ses descendants, et c'est celle qui est déjà dans le dépôt
AdaFisher. Coût : un produit scalaire par pas, rien d'autre.

### Famille D — Écrêtage spectral

On écrase *toutes* les valeurs singulières de la mise à jour à 1 (c'est ce que fait l'optimiseur
Muon). C'est le cas limite de la famille A poussé jusqu'au bout : on ne garde plus que la direction,
plus aucune information d'amplitude. Mentionné pour la complétude ; hors sujet pour un projet dont
l'objet est justement d'étudier ce que l'amplitude de la courbure apporte.

---

## 4. Ce que dit la littérature

### 4.1 Le problème que nous avons est nommé dans l'article fondateur de K-FAC

Martens & Grosse (2015), §6.2 — c'est-à-dire l'article que l'article AdaFisher cite pour justifier
son `λ` :

> « nous avons constaté qu'il ne semble jamais y avoir de "bon" choix de `λ` donnant des mises à jour
> d'une qualité comparable à celles produites par les méthodes utilisant la Fisher exacte. […] `λ`
> doit rester grand pour compenser l'inexactitude intrinsèque du modèle, ce qui a pour effet
> secondaire d'effacer les petites valeurs propres. »

Leur solution n'est pas un `λ` mieux réglé : c'est un schéma en **deux étages** — un amortissement
réparti sur les deux facteurs (§6.3), puis un **rééchelonnement de la mise à jour** calculé sur la
Fisher exacte (§6.4). Le second étage est un contrôle d'amplitude, c'est-à-dire de la famille C.
AdaFisher a repris le principe de Tikhonov et aucun des deux étages.

### 4.2 TKFAC a mesuré exactement notre pathologie, et l'a corrigée sans écrêtage

TKFAC (§5.16) rapporte, sur ResNet-20 / CIFAR-10, que « l'amortissement des couches convolutives
devient très vite bien plus grand que la moyenne des éléments diagonaux, ce qui signifie que
l'amortissement peut jouer le rôle principal à la place de la matrice de Fisher peu après le début
de l'entraînement » — la phrase décrit notre situation mot pour mot. Leur correctif est un
amortissement **relatif**, proportionnel à la trace de la couche divisée par sa dimension. TEKFAC
(Éq. 3.5) reprend le même mécanisme.

C'est exactement le correctif S1 du plan, et c'est l'argument le plus fort pour lui : deux articles
sources sur cinq ont rencontré le problème et l'ont réglé ainsi, pas par de l'écrêtage.

### 4.3 L'écrêtage par coordonnée : Sophia

Sophia (Liu et al., 2023) est la référence moderne de la famille B. La mise à jour est
`écrêter(momentum / courbure, ρ)`. Deux éléments transposables directement :

- Les auteurs recommandent de régler `ρ` **par la proportion de coordonnées écrêtées**, qu'ils
  appellent `win_rate`, et de la maintenir **entre 0,1 et 0,5**. C'est une cible observable en
  continu, alors que le percentile de `λ` a demandé chez nous une campagne dédiée.
- Ils décrivent explicitement le régime dégénéré : quand l'information de courbure est trompeuse,
  l'écrêtage se déclenche et « l'optimiseur retombe sur une descente de signe ». C'est précisément
  ce que notre mesure prédit pour nous, avec un taux de déclenchement de 100 %.

### 4.4 L'écrêtage global : ACKTR, et le propre dépôt AdaFisher

ACKTR (Wu et al., 2017, §3.2) combine K-FAC avec une région de confiance : le pas effectif vaut
`min(η_max, √(2δ / Δᵀ F̂ Δ))`. Le point à retenir : ils utilisent **aussi** l'amortissement factorisé
de K-FAC. La région de confiance s'ajoute, elle ne remplace pas.

Le K-FAC embarqué dans le dépôt AdaFisher applique la même idée sous le nom `kl_clip`, avec
`ν = min(1, √(κ / Δᵀ g))`. La valeur par défaut est `κ = 0.001`.

**Observation utile pour nous.** Dans notre régime où `λ` domine, `Δ ≈ g / λ`, donc `Δᵀ g ≈ ‖g‖² / λ`
et le facteur devient `min(1, √(κλ) / ‖g‖)` : la région de confiance **dégénère en un écrêtage
classique de la norme du gradient**. C'est une dégénérescence bien plus bénigne que celle
d'aujourd'hui : l'écrêtage de gradient est un algorithme éprouvé, avec une justification théorique
(Zhang et al., 2019 : la courbure locale croît avec la norme du gradient, un pas fixe qui convient
dans les zones plates diverge dans les zones raides), alors que « du momentum à un taux effectif de
1,0 » n'en a aucune.

### 4.5 Découpler direction et amplitude : le « greffage »

Agarwal, Anil et al. ont introduit le *learning-rate grafting* : prendre la **direction** d'un
optimiseur et l'**amplitude** d'un autre. C'est décrit comme « l'ingrédient clé pour faire
fonctionner Shampoo en pratique ». C'est le cousin direct du correctif S2 du plan (renormaliser la
direction préconditionnée à la longueur du momentum).

Un travail récent (« Purifying Shampoo », 2025) va plus loin : le greffage sert à « compenser
l'obsolescence et le mauvais calibrage des valeurs propres du préconditionneur », et **corriger
directement les valeurs propres supprime le besoin de greffage**. Corriger les valeurs propres est
exactement ce que font nos modes `ekfac` et `tekfac`. Nuance importante pour nous : cette correction
suppose que les valeurs propres sont estimées **à la bonne échelle** ; chez nous elles sont estimées
fidèlement à une échelle fausse de 2 × 10⁸, donc la correction ne peut pas jouer ce rôle.

### 4.6 Une différence entre écrêter et normaliser

La normalisation (S2, ou le greffage) impose *toujours* l'amplitude. L'écrêtage ne la touche *que
lorsqu'elle dépasse le seuil*. La littérature préfère majoritairement le plafond, pour une raison
simple : près d'un minimum, les gradients diminuent naturellement, et une normalisation empêche
cette décroissance. À noter toutefois : écrêter introduit un biais sur le pas moyen, c'est un sujet
d'étude à part entière.

---

## 5. Est-ce applicable chez nous ? La mesure

### 5.1 Ce qui a été fait

Sonde temporaire, exécutée en local, sur le réseau `mlp_ln_mnist`, à partir du point de contrôle à
50 % de l'entraînement de chaque mode, avec le protocole de ré-échauffement déjà employé par les
expériences E1 et E3 du plan (1 000 pas). Pour chaque couche et chaque mode, on calcule, dans la
base propre du mode :

- le **pas maximal appliqué aujourd'hui** ;
- la **proportion de coordonnées** qu'un écrêtage réglé sur ce pas maximal écrêterait ;
- la distribution du **pas naturel non amorti** `|momentum| / courbure`.

Cohérence vérifiée : les maxima de courbure rapportés par la sonde reproduisent ceux de E1 (par
exemple `ekfac / features.3` : 3,4 × 10⁻⁴ fois `λ`, chiffre identique).

### 5.2 Le résultat

| mode | couche | pas max actuel | proportion écrêtée à ce seuil | seuil pour écrêter 50 % | taux d'apprentissage qu'il faudrait |
|---|---|---|---|---|---|
| `ekfac` | `features.0` | 0,0173 | **0,993** | 3,0 × 10⁵ | 5,7 × 10⁻⁸ |
| `ekfac` | `features.3` | 0,0123 | **1,000** | 2,2 × 10⁵ | 5,5 × 10⁻⁸ |
| `ekfac` | `head` | 0,0051 | **0,982** | 2,9 × 10⁵ | 1,8 × 10⁻⁸ |
| `diag` | `features.0` | 0,0037 | **0,998** | 4,8 × 10³ | 7,6 × 10⁻⁷ |
| `diag` | `features.3` | 0,0068 | **0,965** | 2,5 × 10² | 2,7 × 10⁻⁵ |
| `diag` | `head` | 0,0063 | **0,955** | 2,2 × 10² | 2,9 × 10⁻⁵ |
| `kfac` | toutes | 0,002 – 0,017 | **1,000** | 10⁷ – 10¹² | 10⁻¹⁴ – 10⁻¹⁰ |

Trois lectures.

**Première : l'écrêtage seul ne rend pas la méthode sensible à la courbure.** Au seuil qui reproduit
le comportement actuel, 95 % à 100 % des coordonnées sont plafonnées. La mise à jour devient
« `ρ` fois le signe du momentum » — informative sur la direction, muette sur l'amplitude. C'est le
symétrique exact du problème actuel.

**Deuxième : le problème d'échelle est déplacé, pas résolu.** Pour qu'une moitié des coordonnées
échappe au plafond, il faut placer le seuil autour de 10⁵ (pour `ekfac`) ou 10¹² (pour `kfac`), et
donc diviser le taux d'apprentissage d'autant pour que le pas reste raisonnable. C'est mot pour mot
la reparamétrisation que l'expérience E2 réalise déjà en couplant `λ` et le taux d'apprentissage.
`ρ` est le même problème que `λ` sous un autre nom.

**Troisième, et c'est là que se trouve le gain :** *cette reparamétrisation-là ne peut pas
diverger*. Avec l'amortissement, le pas maximal vaut `taux × max|momentum| / λ` — il dépend des
données et explose quand `λ` descend (la campagne l'a mesuré : `NaN` à `λ = 10⁻⁶` à taux constant).
Avec l'écrêtage, le pas maximal vaut `taux × ρ`, exactement, quelles que soient les données et
quelle que soit l'échelle de la courbure. **On peut balayer tout le domaine sans risque.**

### 5.3 Une limite propre à notre cas, mesurée elle aussi

Le seuil de l'écrêtage par coordonnée est `|momentum| / ρ`. Il ne rend service que s'il ne suit pas
déjà la courbure. Or nos modes utilisent la Fisher **empirique** : la courbure y est construite à
partir des mêmes gradients que le momentum. Mesuré sur la sonde, la corrélation entre le logarithme
du momentum et le logarithme de la courbure vaut **+0,42 à +0,64 sur les couches de poids**
(essentiellement nulle sur les couches de normalisation).

Autrement dit, sur les couches qui portent l'essentiel des paramètres, un tiers environ de la
variation de la courbure est déjà reproduit par le momentum lui-même. L'écrêtage différenciera donc
moins que ne le ferait un amortissement bien calibré. Ce n'est pas rédhibitoire, mais il ne faut pas
espérer de l'écrêtage la finesse d'un préconditionneur correct.

**Réserve de lecture.** Un seul réseau, un seul point de la trajectoire, une seule graine aléatoire.
Les colonnes `kfac` où l'étalement du pas naturel atteint 10²⁸⁶ sont l'artefact de déficience de rang
déjà connu du projet (136 valeurs propres exactement nulles sur la première couche, dues aux pixels
MNIST toujours noirs) : à ne pas lire comme un résultat.

### 5.4 D'où viennent réellement les `NaN` du dépôt

Question posée en marge de cette étude, et la réponse est nette : **il y a deux populations de
`NaN` et elles n'ont rien à voir l'une avec l'autre.**

**Population 1 — 945 cellules dans `benchmarks/outputs/`, et aucune n'est un échec.** Toutes sont
dans les colonnes de *précision* (`val_acc`, `test_acc`), et toutes appartiennent à des runs
`mnist_autoencoder`. Un auto-encodeur n'a pas d'étiquettes, donc pas de précision à rapporter :
`bench.py:35` passe `metric_fn=None` (« reconstruction : pas de précision ») et `records.py:70`
renvoie `float("nan")` comme valeur de remplissage — comportement documenté à `records.py:8`.
Vérifié sur l'ensemble des fichiers CSV du répertoire : **la seule colonne contenant des valeurs non
finies est `val_acc`**, jamais une perte. Cohérent avec l'audit de la campagne 1 (« 0 perte non
finie sur ~700 000 pas enregistrés »).

**Population 2 — le balayage de `λ` (étape 15), et là ce sont de vrais échecs.** 181 au total, dans
`fisher_ref/outputs/warmup_sgd_lambda_sweep_lam*.json`, en deux mécanismes bien séparés :

| mécanisme | nombre | modes touchés | à quel moment |
|---|---|---|---|
| plantage dans l'algèbre linéaire | **100** | `kfac` 30, `tkfac` 29, `ekfac` 21, `tekfac` 20 | avant le pas 1 |
| divergence numérique | **81** | `diag` 22, les témoins sans courbure 55, `ekfac`/`tekfac` 4 | en cours de route |

*Le premier mécanisme* est une erreur `_LinAlgError` : `linalg.inv: the diagonal element is zero`
pour `kfac`/`tkfac`, `linalg.eigh: the algorithm failed to converge` pour `ekfac`/`tekfac`. Quand
`λ` descend, `A + √(λπ)I` cesse d'être numériquement inversible sur un facteur exactement déficient
en rang. Tout est marqué `NaN` dans ces runs, y compris le temps d'exécution, parce qu'ils n'ont
jamais démarré. **Aucun écrêtage ne prévient cela** : la panne est dans la construction du
préconditionneur, pas dans la taille du pas. C'est le correctif S5 du plan.

*Le second mécanisme* est un débordement pur, pas une division 0/0 : juste avant le premier `NaN`,
la perte vaut déjà 10⁸ à 10³² et double à chaque pas. Deux signatures le confirment :

- **Le moment.** Sur les 81 divergences, **75 surviennent dans les 40 pas qui suivent un multiple de
  `TCov = 100`** (premiers `NaN` : 109-135, puis 204-262, puis 411-440, 512, 578-605, 720-735, 911).
  C'est exactement le calendrier de mise à jour des facteurs : la matrice identité qui sert de germe
  décroît d'un facteur 12 à chaque mise à jour, et tant qu'elle est là, elle protège le diviseur.
  Dès qu'elle s'efface, le diviseur tombe à `λ`, le pas est multiplié par `taux / λ`, et la
  divergence part.
- **Les témoins.** Les arcs `sgd_*` — un optimiseur à momentum **sans aucune estimation de
  courbure**, avec seulement le même plafond de pas — divergent eux aussi, et ils comptent pour 55
  des 81 cas. Leur `NaN` ne peut venir que de la taille du pas. C'est la confirmation indépendante
  que ces divergences sont un problème de pas, pas de courbure.

Et le nombre d'échecs croît de façon monotone quand `λ` descend : quasi rien à `λ = 10⁻⁴`,
presque tous les réseaux à `λ = 10⁻⁸`. C'est le comportement d'un pas qui explose en `1/λ`.

**Conclusion pour cette étude.** Sur les deux mécanismes réels, l'écrêtage en supprime un (la
divergence) par construction, et **ne touche pas** l'autre (le plantage matriciel). Cela renforce le
§1 point 1 et le nuance en même temps : un plafond rend le balayage de `λ` sûr du côté de
l'entraînement, mais il faut quand même S5 pour que les modes `kfac`/`tkfac` survivent à un `λ`
petit.

---

## 6. Avantages et limites, mis côte à côte

### Ce que l'écrêtage apporte réellement

1. **Une borne dure sur le pas.** `taux × ρ`, garantie, indépendante des données et de l'échelle de
   la courbure. Aujourd'hui rien ne borne le pas : il vaut `taux × max|momentum| / λ`, quantité qui
   n'est ni contrôlée ni surveillée. C'est la garde-fou que le plan réclame en S5 après l'effondrement
   silencieux de l'étape 15.
2. **Un balayage sûr.** Le plan impose une règle de protocole (« tout balayage de `λ` doit maintenir
   le facteur de pas constant ») précisément parce que la formulation actuelle mélange deux effets.
   Avec un plafond, la contrainte disparaît d'elle-même.
3. **Un réglage guidé par une cible observable.** `ρ` se règle sur la proportion écrêtée, dont la
   littérature donne une cible universelle (0,1 à 0,5), alors que `λ` se règle sur un percentile qui
   doit être mesuré hors ligne. C'est un gain d'ergonomie, pas un gain d'information — voir la
   réserve du §7.1.
4. **Une comparaison honnête avec Adam.** Adam borne naturellement son pas à environ le taux
   d'apprentissage, par construction. Nos modes ne bornent rien. Comparer les deux « au même taux
   d'apprentissage » compare deux pas d'amplitudes arbitrairement différentes — le plan le note déjà
   en E5. Un plafond explicite met les deux familles sur la même échelle.
5. **Un précédent direct.** La technique est dans le K-FAC du dépôt AdaFisher, avec sa valeur par
   défaut. Ce n'est pas une invention.

### Ce qu'il n'apporte pas

1. **Il ne corrige pas l'échelle.** Mesuré : 95–100 % d'écrêtage. Les deux conventions héritées
   restent à corriger, et le correctif relatif (S1) reste nécessaire.
2. **Il a sa propre dégénérescence.** Tout écrêter revient à une descente de signe. C'est
   symétriquement aussi peu informatif que tout amortir — avec une circonstance atténuante : la
   descente de signe est un algorithme connu qui fonctionne souvent bien, contrairement au régime
   actuel.
3. **Il n'ajoute pas de degré de liberté.** `(taux, ρ)` en offre deux, comme `(taux, λ)`. Le gain est
   que le mauvais réglage sature au lieu de diverger, pas qu'il y ait un bouton de plus.
4. **Coût réel sur `kfac` et `tkfac`.** L'écrêtage par coordonnée n'a un sens géométrique que dans
   une base où l'opérateur est diagonal. `ekfac` et `tekfac` en ont déjà une (c'est leur principe) ;
   `kfac` et `tkfac` inversent leurs facteurs sans jamais les diagonaliser. Deux options : ajouter
   une décomposition propre (coût non négligeable), ou écrêter dans la base des paramètres — ce qui
   reste légitime (c'est une région de confiance en norme infinie) mais n'est plus équivalent à un
   plancher sur la courbure.
5. **La corrélation momentum/courbure émousse l'effet** (§5.3).
6. **Il ne touche pas au transitoire de démarrage.** Pendant les premières centaines de pas, le
   diviseur n'est ni la courbure ni `λ` mais le résidu de la matrice identité qui sert de germe. Le
   correctif S3 reste nécessaire.

---

## 7. Comment on le réaliserait

Par ordre croissant de coût. **Rien de tout cela n'est proposé à l'implémentation aujourd'hui** ;
c'est l'inventaire des points de branchement, pour que la décision se prenne en connaissance de
cause.

### 7.1 Étape 0 — Ce qu'il ne sert à rien de journaliser, et ce qui le remplace

*Correction d'une première version de ce rapport, qui recommandait de journaliser en continu la
proportion de coordonnées qui seraient écrêtées. C'était une mauvaise recommandation.*

Trois raisons :

1. **La réponse est déjà connue, et c'est une constante.** E1 a établi que `λ` est au 100ᵉ percentile
   dans les 105 cas mesurés. La proportion écrêtée vaudrait donc 1,00 à chaque pas de chaque
   entraînement. Journaliser une constante connue n'apprend rien.
2. **La quantité n'est pas définie sans seuil.** « La proportion qui *serait* écrêtée » suppose un
   `ρ` de référence, qu'il faudrait choisir arbitrairement tant qu'on n'écrête pas. Le nombre obtenu
   dépendrait entièrement de ce choix : c'est un chiffre creux.
3. **Le plan a déjà la bonne métrique, et elle est meilleure.** E5 propose de journaliser le
   *facteur d'amplification du pas*, `‖F̃⁻¹m̂‖ / ‖m̂‖`, par couche et par pas. Cette quantité-là n'a
   besoin d'aucun seuil arbitraire, vaut pour les cinq modes de la même façon, est directement
   comparable à Adam, et aurait détecté l'effondrement silencieux de l'étape 15. C'est elle qu'il
   faut, pas la proportion écrêtée.

**Où la proportion écrêtée redevient utile :** à l'intérieur d'une expérience où l'échelle ou `λ`
varient réellement (E2, E4), et une fois que `ρ` a un sens parce qu'on écrête pour de bon. Comme
instrument de réglage, pas comme journal permanent.

### 7.2 Option C — Région de confiance globale (petit changement de structure)

Après avoir calculé les directions de tous les modules, calculer le scalaire
`ν = min(1, √(κ / Σ direction · gradient))` et multiplier toutes les directions par `ν`.

Obstacle structurel : `step()` traite aujourd'hui les modules **un par un** (calcul puis application
immédiate). Il faudrait deux passes — calculer toutes les directions, puis appliquer. Changement
petit mais réel.

Deux décisions à prendre explicitement : (a) le budget `κ` est-il global ou par couche ; (b) les
paramètres non pris en charge par l'optimiseur (les couches GroupNorm, le jeton de classe et la
table de positions d'un Transformer) entrent-ils dans la somme ? Le §2.3 du plan montre qu'ils vivent
aujourd'hui à une échelle 1 000 fois différente ; les inclure sans y penser mélangerait deux régimes.

### 7.3 Option B — Écrêtage par coordonnée (une ligne pour trois modes sur cinq)

Pour `ekfac` et `tekfac`, la base propre existe déjà et la division y est déjà faite terme à terme :
il suffit de plafonner le résultat de cette division. Pour `diag`, même chose. Pour `kfac` et
`tkfac`, voir la limite 4 du §6.

Le réglage se fait par la proportion écrêtée, cible 0,1 à 0,5, en s'appuyant sur le journal de
l'étape 0.

### 7.4 Ce qu'il ne faut pas faire

Remplacer `courbure + λ` par `max(courbure, λ)` (famille A). C'est la lecture la plus naturelle de la
question posée, et c'est un changement qui, dans notre régime, est **prouvablement sans effet** : les
deux expressions valent `λ` partout tant que `λ` domine. Une demi-journée de travail pour un résultat
nul, et un résultat nul difficile à interpréter si on ne l'a pas prévu.

---

## 8. Recommandation

1. **Journaliser le facteur d'amplification du pas** (E5 du plan), pas la proportion qui serait
   écrêtée — voir §7.1 pour la correction de ce point. Gratuit, sans seuil arbitraire, valable pour
   les cinq modes, utile quelle que soit la suite.
2. **Ne pas considérer l'écrêtage comme le correctif du problème `λ`.** Le correctif est
   l'amortissement relatif (S1), prescrit par TKFAC §5.16 et TEKFAC Éq. 3.5, éventuellement avec le
   rétablissement de l'échelle (S1 le rend largement inutile). L'écrêtage est **complémentaire** :
   S1 place le plancher au bon endroit, l'écrêtage garantit que le pas reste borné pendant qu'on
   cherche ce bon endroit.
3. **Si l'on veut tester une forme d'écrêtage, commencer par la famille C** (un scalaire par pas,
   précédent direct dans le code ancêtre, aucun besoin de décomposition propre), et la traiter
   d'abord comme un **instrument de mesure** : la quantité `Σ direction · gradient` par pas est en
   elle-même un diagnostic de l'amplitude réelle des mises à jour.
4. **La famille B (Sophia) est le meilleur candidat sur `ekfac` / `tekfac`**, et seulement une fois
   l'échelle corrigée — sinon on mesure une descente de signe.
5. **Garder en tête le cadrage général** : l'expérience E0 a montré que le bruit des courbes de
   validation, qui est à l'origine de toute cette enquête, est reproduit par un optimiseur sans
   aucune estimation de courbure. L'écrêtage doit donc être jugé sur « est-ce que cela rend la
   méthode meilleure que du momentum simple ? », pas sur la question du bruit.

---

## 9. Ce qui reste incertain

- La sonde ne couvre **qu'un réseau** (`mlp_ln_mnist`), **un point** de la trajectoire et **une
  graine**. La proportion écrêtée de 95–100 % est si loin de 50 % que l'ordre de grandeur ne fait
  guère de doute, mais les chiffres exacts ne doivent pas être cités comme un résultat de campagne.
- Aucune publication, à ma connaissance, ne combine correction des valeurs propres (EKFAC) et
  écrêtage par coordonnée. Ce serait donc une combinaison nouvelle — peu risquée sur le plan
  technique, mais sans référence à laquelle se comparer.
- Le biais introduit par l'écrêtage sur le pas moyen est documenté dans la littérature sur la
  descente de gradient écrêtée ; il n'a pas été quantifié ici.
- La question « l'écrêtage rendrait-il ces méthodes meilleures qu'Adam ? » n'est pas abordée : rien
  dans ce document ne la traite, et rien dans la campagne actuelle ne permet d'y répondre.

---

## Sources

**Articles du projet** (dans `docs/papers/`) :
- Martens & Grosse, *Optimizing Neural Networks with Kronecker-factored Approximate Curvature*,
  §6.2 (insuffisance de `λ`), §6.3 (amortissement factorisé), §6.4 (rééchelonnement) —
  [arXiv:1503.05671](https://arxiv.org/abs/1503.05671)
- TKFAC, §5.16 (amortissement adaptatif, et la mesure de la même pathologie sur ResNet-20) —
  [arXiv:2011.10741](https://arxiv.org/abs/2011.10741)
- TEKFAC, Éq. 3.5 (amortissement relatif à la trace) — [arXiv:2011.13609](https://arxiv.org/abs/2011.13609)
- AdaFisher, §3 et Alg. 1 (le `λ = 0.001` et sa justification) —
  [arXiv:2405.16397](https://arxiv.org/abs/2405.16397)

**Littérature externe** :
- Liu et al., *Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training*
  — [arXiv:2305.14342](https://arxiv.org/abs/2305.14342), et la consigne de réglage du `win_rate`
  dans [l'implémentation officielle](https://github.com/Liuhong99/Sophia)
- Wu et al., *Scalable trust-region method for deep reinforcement learning using Kronecker-factored
  approximation* (ACKTR), §3.2 — [arXiv:1708.05144](https://arxiv.org/abs/1708.05144)
- Zhang et al., *Why Gradient Clipping Accelerates Training: A Theoretical Justification for
  Adaptivity* — [arXiv:1905.11881](https://arxiv.org/abs/1905.11881)
- Agarwal, Anil et al., *Disentangling Adaptive Gradient Methods from Learning Rates* (greffage) —
  [arXiv:2002.11803](https://arxiv.org/abs/2002.11803)
- *Purifying Shampoo: Investigating Shampoo's Heuristics by Decomposing its Preconditioner* —
  [arXiv:2506.03595](https://arxiv.org/abs/2506.03595)
- Jordan et al., optimiseur Muon (écrêtage spectral), vu à travers
  [*Understanding Gradient Orthogonalization for Deep Learning via Non-Euclidean Trust-Region
  Optimization*](https://arxiv.org/abs/2503.12645)

**Code du dépôt** :
- [`reference_repos/AdaFisher/optimizers/kfac.py:131-146`](../../../reference_repos/AdaFisher/optimizers/kfac.py)
  — l'écrêtage global `kl_clip`, présent dans le K-FAC de référence et absent d'AdaFisher
- [`src/adafisher_modes/approximations/ekfac.py:132-148`](../../../src/adafisher_modes/approximations/ekfac.py)
  — le point où un écrêtage par coordonnée s'insérerait en une ligne
- [`src/adafisher_modes/optimizer.py`](../../../src/adafisher_modes/optimizer.py) `step()` /
  `_step_module` — le point où une région de confiance globale demanderait deux passes
