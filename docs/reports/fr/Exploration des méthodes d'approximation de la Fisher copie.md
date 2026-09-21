# Approximer et calculer la matrice de Fisher sur GPU moderne

**Techniques cross-domaines pour l'approximation de la Fisher empirique, méthodes de calcul de la Fisher exacte, et classement comparatif**

_Rapport de recherche — septembre 2026. Destiné au dépôt `fisher_drift_analysis` (critique d'AdaFisher / FisherAdapTune et conception de méthodes correctrices)._

---

## Comment lire ce rapport

Trois conventions valent pour tout le document.

**Statut de chaque affirmation.** `[ÉTABLI]` = résultat publié dans une source vérifiée sur la source primaire ; `[DÉRIVÉ]` = arithmétique effectuée à partir de chiffres publiés ; `[EXTRAPOLATION]` = raisonnement argumenté mais non mesuré dans la littérature — c'est-à-dire une piste de recherche, pas un fait.

**Vérification.** Toute référence portant un identifiant arXiv a été confrontée à la source primaire (titre + auteurs + date de soumission v1). Les références qui n'ont pas pu être vérifiées, ou qui se sont révélées non substantiables, sont listées en §7.2 et ne sont utilisées nulle part comme appui d'un argument.

**Terminologie.** Tout acronyme est défini à sa première occurrence, et redéfini en §1.1 dans un glossaire compact. Le lecteur peut lire les sections 2, 3 et 4 indépendamment après la section 1.

---

## 0. Résumé exécutif

Ce rapport répond à trois questions : (1) quelles techniques, y compris hors du champ du machine learning, permettraient d'approximer la matrice d'information de Fisher plus efficacement que K-FAC/EKFAC ; (2) comment calculer et exploiter la Fisher **exacte** à coût raisonnable ; (3) comment classer ces techniques. Six conclusions structurent l'ensemble.

`[ÉTABLI]` **(a) Le coût wall-time des optimiseurs du second ordre n'est pas là où on le dit.** Il n'est ni dans la passe arrière supplémentaire, ni dans la construction des facteurs de covariance, ni dans les FLOPs : il est **entièrement concentré dans les routines spectrales denses** (décomposition en valeurs propres, SVD). Ces routines atteignent 1 à 2 % du pic d'un GPU moderne là où un produit matriciel dense en atteint 60 à 80 % — un écart de près de deux ordres de grandeur en débit effectif, et jusqu'à quatre sur les petites matrices par couche (§1.3.1). Cet écart domine toute considération de complexité asymptotique aux tailles rencontrées (§1.3). La frontière pertinente n'est donc pas « premier ordre vs second ordre » mais **« exprimable en produits matriciels denses vs non exprimable »**.

`[ÉTABLI]` **(b) Conséquence immédiate et déjà démontrée : supprimer la décomposition spectrale.** DASH (Modoranu et al., arXiv:2602.02016) remplace la décomposition propre de Shampoo par des itérations de Newton couplées en demi-précision, empilées en tenseurs 3D traités par produit matriciel par lots, et mesure **119 ms contre 3080 ms** sur le même pas d'optimiseur (facteur 25,9×) — avec, pour la variante Newton–Denman–Beavers, une **meilleure perplexité de validation que la décomposition exacte** (11,68 vs 11,80). L'inexactitude de l'itération polynomiale joue le rôle d'une régularisation du spectre. C'est une réfutation empirique publiée de l'idée que la décomposition spectrale est nécessaire à un préconditionneur de courbure — et AdaFisher, qui repose sur une décomposition de ses facteurs de Kronecker, y est directement exposé.

`[ÉTABLI en physique/PINNs — EXTRAPOLATION sur le transfert au PEFT]` **(c) Pour la Fisher exacte, la voie n'est pas l'échantillonnage mais la dualité.** Deux faits se combinent. D'abord, pour toute perte de log-vraisemblance d'une famille exponentielle à lien canonique (softmax-entropie croisée sur les logits, erreur quadratique gaussienne), la Fisher **est** la matrice de Gauss-Newton généralisée, dont le facteur de sortie a une forme analytique : **aucun échantillonnage de label n'est nécessaire**. Ensuite, cette matrice s'écrit $J^\top\Lambda J$, de rang au plus $B(C-1)$ ($B$ = taille de batch, $C$ = nombre de classes) : le système $(F+\lambda I)x = g$ se résout **exactement** par la formule de Woodbury dans un espace de dimension $B(C-1)$. La communauté du Monte-Carlo variationnel quantique a poussé cette formulation duale jusqu'à **10⁶ paramètres** (Chen & Heyl, _Nature Physics_ 2024) et celle des réseaux informés par la physique jusqu'à **12,8 M paramètres sur un seul GPU** (Jnini & Vella, arXiv:2505.21404), pendant que la communauté ML continuait d'empiler des variantes de Kronecker. Le transfert vers le PEFT n'a pas été fait.

`[DÉRIVÉ]` **(d) En PEFT/LoRA sur des tâches à peu de classes, la Fisher vraie et exacte est calculable dès aujourd'hui**, et pour un coût qui n'est pas une approximation de l'ordre de grandeur : 37 Mo pour RoBERTa-base + LoRA rang 8 sur une tâche binaire GLUE, 0,87 Go pour LLaMA-3-8B + LoRA rang 16. En classification binaire, **elle coûte exactement le même prix que la Fisher empirique** (une rétropropagation). Il n'y a donc, dans ce régime, aucune justification technique à utiliser une Fisher empirique, une factorisation de Kronecker ou une moyenne mobile exponentielle — et cela fournit une **vérité terrain mesurable** pour auditer AdaFisher et FisherAdapTune sur pièces plutôt que par principe (§3.4). La ligne de fracture est nette : en modélisation de langue causale, le facteur longueur × vocabulaire fait exploser la dimension duale et l'échantillonnage redevient obligatoire.

`[ÉTABLI en algèbre — EXTRAPOLATION sur la Fisher]` **(e) K-FAC n'est pas « une approximation avec une hypothèse » : c'est une troncature au rang 1 dans un espace où le rang exact est $N$, et ce n'est même pas la meilleure troncature au rang 1.** Le réarrangement de Van Loan–Pitsianis (1992) transforme $|F - B\otimes C|_F$ en un problème d'approximation de rang 1 d'une matrice réarrangée $\mathcal{R}(F)$ ; le bloc de Fisher exact d'une couche vérifie $\mathcal{R}(F_\ell)=\frac1N\sum_n \mathrm{vec}(a_na_n^\top)\mathrm{vec}(g_ng_n^\top)^\top$, de rang $\le N$. K-FAC prend le produit extérieur des **moyennes** ; l'optimum de rang 1 est le produit extérieur des **vecteurs singuliers dominants**. Les deux coïncident sous l'hypothèse d'indépendance, qui est exactement la condition « $\mathcal{R}(F_\ell)$ de rang 1 ». Le rapport $\sigma_2/\sigma_1$ de $\mathcal{R}(F_\ell)$ est donc **une mesure scalaire, calculable et non heuristique du biais d'indépendance**, couche par couche et au fil du temps. Personne ne l'a publiée sur une Fisher de réseau moderne, et c'est le diagnostic le moins cher et le plus décisif de tout ce rapport (§2.B.1).

`[ÉTABLI hors ML — EXTRAPOLATION sur le transfert]` **(f) Le damping et l'EMA — les deux hyperparamètres que toute la littérature K-FAC règle à la main — ont chacun une théorie mature dans un autre domaine.** Le damping $\varepsilon I$ est le _diagonal loading_ du beamforming robuste, où il est dérivé depuis 2003 comme le multiplicateur de Lagrange d'un problème min-max sur une boule d'incertitude ; c'est aussi le shrinkage de Ledoit–Wolf, dont la valeur optimale a une forme close ; c'est encore l'inflation de covariance de l'assimilation de données, estimée en ligne à partir des innovations. La moyenne mobile exponentielle scalaire souffre d'une pathologie connue depuis quarante ans en identification de systèmes (l'_estimator windup_ : divergence de la covariance dans les directions non excitées par le régresseur), dont le remède — l'oubli directionnel — n'a jamais été transposé. Aucun de ces quatre corpus n'a été appliqué à la courbure de réseaux de neurones (§2.E).

**Le classement final (§4)** place en tête, par ordre décroissant de rapport gain/risque : (1) racine inverse matmul-only avec empilement par lots, (2) formulation duale exacte par Woodbury en régime PEFT, (3) correction iEF de la Fisher empirique, (4) diagnostic $\sigma_2/\sigma_1$ du réarrangement, (5) critère de fraîcheur adaptatif par facteur, (6) Kronecker de rang $r>1$ par SVD randomisée du réarrangement, (7) diagonale + rang faible estimée conjointement.

---

## 1. Cadre

### 1.1 Taxonomie rigoureuse et glossaire

Soit un réseau $f(x,\theta)\in\mathbb{R}^C$ de paramètres $\theta\in\mathbb{R}^P$, un modèle prédictif $r(y\mid z)$ avec $z=f(x,\theta)$, une perte $\mathcal L(y,z)=-\log r(y\mid z)$, un jeu de données ${(x_n,y_n)}_{n=1}^N$, une taille de batch $B$. On note $J_n=\partial f(x_n,\theta)/\partial\theta\in\mathbb{R}^{C\times P}$ la jacobienne par échantillon.

**Fisher vraie** $F$ — l'espérance porte sur des labels **tirés du modèle** : $$F=\frac1N\sum_n \mathbb{E}_{y\sim r(\cdot\mid z_n)}\big[\nabla_\theta\log r(y\mid z_n),\nabla_\theta\log r(y\mid z_n)^\top\big].$$ C'est la seule définition pour laquelle vaut l'identité de l'information $F=\mathbb{E}[-\nabla^2_\theta\log r]$, donc la seule qui hérite des garanties du gradient naturel d'Amari (invariance de reparamétrisation au premier ordre).

**Gauss-Newton généralisée (GGN)** $G=\frac1N\sum_n J_n^\top\Lambda_n J_n$, avec $\Lambda_n=\nabla^2_z\mathcal L(y,z)|_{z_n}$ la hessienne de la perte par rapport aux **sorties**. On différencie deux fois en ignorant la courbure du réseau. $G\succeq0$ dès que $\mathcal L$ est convexe en $z$.

**Fisher empirique** $\tilde F=\frac1N\sum_n\nabla_\theta\mathcal L(y_n,z_n)\nabla_\theta\mathcal L(y_n,z_n)^\top$ — labels **réels du dataset**, un gradient par échantillon, rang $\le B$, coût d'une seule rétropropagation. C'est ce qu'utilisent Adam, EWC, AdaFisher et FisherAdapTune.

**Fisher échantillonnée** $\hat F_K$ — estimateur Monte-Carlo non biaisé de $F$ obtenu en tirant $K$ labels $\tilde y_{nk}\sim r(\cdot\mid z_n)$ ; coût $K$ rétropropagations, rang $\le BK$. C'est ce que prescrit K-FAC dans sa formulation originale.

**Hessien** $H=G+\frac1N\sum_n\sum_c[\nabla_z\mathcal L]_c\nabla^2_\theta f_c$ ; le second terme est indéfini et responsable des valeurs propres négatives. $H$ n'est pas invariant par reparamétrisation, contrairement à $F$ et $G$.

**Autres termes.** _K-FAC_ (Kronecker-Factored Approximate Curvature) : $F_\ell\approx A_\ell\otimes G_\ell$ par couche, où $A_\ell=\mathbb{E}[aa^\top]$ est la covariance des activations d'entrée et $G_\ell=\mathbb{E}[gg^\top]$ celle des gradients rétropropagés de pré-activation. _EKFAC_ : K-FAC dont les valeurs propres sont ré-estimées exactement dans la base propre de Kronecker (KFE). _GGNVP_ (produit GGN-vecteur) : calcul de $Gv$ sans former $G$ ; **unité de coût de ce rapport, ≈ 1 forward + 1 backward ≈ un pas SGD complet**. _Sketching_ (esquisse) : multiplication par une matrice aléatoire $S$ réduisant une dimension tout en préservant les propriétés spectrales. _Low-rank_ (rang faible) : approximation $UU^\top$ avec $U\in\mathbb{R}^{P\times m}$, $m\ll P$. _SIMT_ (_single instruction, multiple threads_) : le modèle d'exécution des GPU, où 32 fils s'exécutent en verrou ; il pénalise la récursion arborescente, la divergence de branches et les tailles de blocs hétérogènes.

**Sigles employés dans la suite.** PEFT (_parameter-efficient fine-tuning_, ajustement d'un petit sous-ensemble de paramètres) ; LoRA (_low-rank adaptation_, adaptateurs $BA$ de rang $r$ ajoutés aux matrices de poids gelées) ; GLUE, suite de tâches de classification de texte ; PINN (_physics-informed neural network_) ; VMC (_variational Monte Carlo_, Monte-Carlo variationnel quantique) ; EWC (_elastic weight consolidation_, régularisation par la Fisher en apprentissage continu) ; NTK (_neural tangent kernel_) ; FIM (_Fisher information matrix_, la Fisher) ; RMT (_random matrix theory_, théorie des matrices aléatoires) ; NLA (algèbre linéaire numérique) ; PSD (semi-définie positive) ; SDP (_semidefinite program_, programme semi-défini — à ne pas confondre avec le précédent) ; SVD (décomposition en valeurs singulières) ; EMA (_exponential moving average_, moyenne mobile exponentielle) ; JVP / VJP (produit jacobienne-vecteur en mode direct / vecteur-jacobienne en mode inverse) ; R-op, l'opérateur différentiel de Pearlmutter réalisant un JVP (§3.2) ; GQA (_grouped-query attention_) ; ADMM (_alternating direction method of multipliers_) ; ADI (_alternating direction implicit_) ; DMRG (_density matrix renormalization group_, l'algorithme de référence des réseaux de tenseurs en physique) ; MVDR (_minimum variance distortionless response_, le beamformer de Capon) ; EnKF (_ensemble Kalman filter_) ; QDWH (_QR-based dynamically weighted Halley_, itération de décomposition polaire) ; HODLR / $\mathcal H^2$ / HSS / HIF / BLR, formats de matrices hiérarchiques définis en §2.C.6 ; FOOF, l'optimiseur « gradient sur les neurones » de Benzing (§3.7) ; HBM, la mémoire haute bande passante d'un GPU ; NCCL, la bibliothèque de collectives inter-GPU de NVIDIA ; FSDP (_fully sharded data parallel_).

### 1.2 Les cinq propriétés structurelles de la Fisher qui décident de tout

1. **Rang déficient par construction.** Sur un batch de $B$ échantillons, $\mathrm{rang}(F)\le B(C-1)$ pour softmax + entropie croisée (le $-1$ vient de $\Lambda_n\mathbf1=0$), et $\mathrm{rang}(\tilde F)\le B$. Pour $P\sim10^6$–$10^9$, c'est un déficit de plusieurs ordres de grandeur. **Ce n'est pas un défaut d'implémentation, c'est structurel** — et c'est aussi l'opportunité centrale de la §3.3.
2. **Spectre à décroissance rapide, structuré en « bulk + valeurs aberrantes ».** Établi empiriquement à grande échelle : Sagun et al. (arXiv:1706.04454), Ghorbani, Krishnan & Xiao (arXiv:1901.10159, ICML 2019), Papyan (arXiv:2008.11865, JMLR 2020). Papyan établit en outre que **le rapport valeurs aberrantes/bulk du spectre de la Fisher est prédictif du taux d'erreur** et propose une correction de K-FAC fondée sur cette structure. Le gradient se concentre dans le sous-espace des valeurs aberrantes : un préconditionneur n'a besoin d'être _précis_ que sur $O(C)$ directions et seulement bien _régularisé_ ailleurs. K-FAC dépense un budget uniforme sur tout le spectre.
3. **Décroissance spectrale des facteurs de Kronecker, _imposée par l'EMA elle-même_.** Puiu (arXiv:2206.15397) établit que la construction des facteurs par moyenne mobile exponentielle force la décroissance de leur spectre, et la mesure indépendante de la largeur de couche. Sketchy (Feinberg, Chen, Sun, Anil, Hazan, arXiv:2302.03764, NeurIPS 2023) mesure une **dimension intrinsèque des facteurs de Kronecker inférieure à 105 pour une dimension nominale supérieure à 1024**. C'est le régime exact où les méthodes de rang faible randomisées sont optimales.
4. **Non-stationnarité.** $F(\theta_t)$ dérive à chaque pas ; toute estimation devient obsolète. C'est ce qui distingue le problème d'une estimation de covariance statique classique, et c'est ce que l'EMA traite — sans garantie (§2.D).
5. **Corrélations inter-couches ignorées.** K-FAC, EKFAC, Shampoo, SOAP et AdaFisher sont tous bloc-diagonaux par couche. Cette bloc-diagonalité n'est dérivée de rien : c'est un artefact de tractabilité, et elle est équivalente à un _taper_ binaire non calibré au sens de l'assimilation de données (§2.F).

Un point mérite d'être isolé, parce qu'il est contre-intuitif et qu'il vient d'un travail récent : Abreu, Vyas, Kakade & Morwani (arXiv:2510.09378, oct. 2025) calculent la GGN **complète** sur des transformeurs jusqu'à 150 M paramètres et trouvent **5,4× moins d'itérations** que SOAP et Muon — mais aussi qu'**un préconditionneur GGN par couche, ignorant l'information inter-couches, égale presque la GGN complète** `[ÉTABLI]`. Si ce résultat se confirme, la bloc-diagonalité serait la moins coûteuse des approximations de K-FAC, et l'effort devrait porter ailleurs (sur le rang de Kronecker intra-couche, §2.B). C'est une des rares mesures directes disponibles et elle doit orienter les priorités. Réserve : les auteurs ne fournissent **aucune comparaison en temps mural**.

### 1.3 Anatomie chiffrée du coût wall-time sur GPU

C'est la section la plus déterminante du rapport : elle fixe le critère de tri de toutes les techniques qui suivent.

#### 1.3.1 Les routines spectrales denses n'atteignent jamais plus de 1 à 2 % du pic

|Mesure|Débit atteint|Source|
|---|---|---|
|`cusolverDnDsyevd` FP64, $n=4096$ (Titan XP, CUDA 11.0)|**88,3 GFLOP/s**|forum développeurs NVIDIA `[ÉTABLI]`|
|idem, $n=1024$|19,7 GFLOP/s|idem|
|idem, $n=256$|**1,74 GFLOP/s**|idem|
|MAGMA `syevd` multi-GPU, $n=49,152$, 8×A100|2,18 TFLOPS = **1,3 % du pic**|Wang et al., arXiv:2511.16174 `[ÉTABLI]`|
|cuSOLVERMp, même configuration|2,37 TFLOPS = **1,5 % du pic**|idem|
|`torch.svd`, lot de 100 matrices 10×10|**70× plus lent sur GPU que sur CPU**|pytorch/pytorch#41306 `[ÉTABLI]`|

Wang et al. l'écrivent dans leur résumé : _« all of the libraries only utilize around 1.5 % of the peak multi-GPU performance »_. Trois causes distinctes, chacune documentée :

- **Phases BLAS-2 limitées par la bande passante.** La tridiagonalisation repose sur des produits matrice-vecteur d'intensité arithmétique ≈ 2 FLOP/octet, contre $\approx 2n/3$ pour un produit matriciel tuilé. Répartition mesurée sur MAGMA — pour la SVD, dont la structure en deux phases est celle de `syevd` à la réduction bidiagonale près : 43,2 %→24,0 % du temps en réduction, 51,8 %→73,3 % en divide-and-conquer (arXiv:2508.11467) `[ÉTABLI ; transposé par analogie à` syevd`]`.
- **Dépendances fines dans le _bulge-chasing_.** Ringoot, Alomairy & Edelman (arXiv:2510.12705) profilent un débit de calcul de **13–23 %** contre une utilisation de bande passante L1/L2 de **52–64 %** : la phase est limitée par la mémoire, pas par le calcul `[ÉTABLI]`.
- **Aucun usage possible des tensor cores.** Réflecteurs de Householder, rotations de Givens et itérations QR sont des séquences de rang faible avec réorthogonalisation ; il n'existe pas de primitive tensor-core exploitable.

**Le cas qui nous concerne est le pire.** K-FAC/AdaFisher ne font pas _une_ grande décomposition mais _des centaines de petites_, une par couche. À $n=256$, on est à 1,74 GFLOP/s, soit ~0,0002 % du pic BF16 d'un H100 `[DÉRIVÉ ; les deux termes sont hétérogènes — FP64 sur Pascal contre BF16 sur Hopper — voir la réserve §6.3]`.

#### 1.3.2 Où partent les millisecondes : décompositions publiées

- **K-FAC, part de l'inversion.** SKFAC (Tang et al., CVPR 2021) mesure que **l'inversion des facteurs de Kronecker représente 38,9 % du temps de calcul du gradient naturel**, ramenée à 3,5 % par une astuce de Sherman-Morrison-Woodbury ; gain wall-clock ImageNet/ResNet-50 sur 4×RTX 2080Ti : 1544,9 → 1189,5 min, soit **−23 %** `[ÉTABLI]`.
- **K-FAC distribué : la communication reprend le gain.** SPD-KFAC (Shi, Zhang & Li, arXiv:2107.06533), ResNet-50 sur 64 GPU : K-FAC mono-GPU ≈ **4× plus lent que SGD** ; calcul des inverses 292 ms en D-KFAC, ramené à 51 ms en distribuant l'inversion — mais **134 ms de diffusion des inverses réapparaissent**, soit **≈ 56 % du gain repris par le réseau**, la diffusion représentant à elle seule 72 % du coût résiduel `[DÉRIVÉ]`. Cause dimensionnelle : pour ResNet-50 (25,5 M paramètres), les facteurs $A$ totalisent 62,3 M éléments et les facteurs $G$ 14,6 M — **le trafic des facteurs de Kronecker dépasse celui des gradients** `[ÉTABLI]`.
- **Shampoo : le chiffre le plus net.** DASH (arXiv:2602.02016), Llama-953M, forward 1000 ms + backward 3000 ms. À taille de bloc 1024 et fréquence de recalcul 1 : décomposition propre **3080 ms** (soit **+77 % de temps d'itération**) ; Newton couplé FP16 empilé en 3D : **119 ms** (soit **+3,0 %**). Rapport **25,9×** `[ÉTABLI/DÉRIVÉ]`. À fréquence 10, les neuf pas qui séparent deux recalculs ne coûtent plus qu'≈ 35 ms chacun : **le coût est entièrement concentré dans le pas de recalcul** `[ÉTABLI]`.
- **Muon : le poste devient l'orthogonalisation.** Dion3 (Amsel et al., arXiv:2608.11612) : sur un modèle 7B / 4×GH200, le pas d'optimiseur Muon coûte **26× celui d'AdamW**, ramené à 4× `[ÉTABLI]`. Fourchette honnête publiée par les mêmes auteurs : 1,9 % du pas total en pré-entraînement Kimi K2 (2048 GPU, batch 67 M tokens, pipeline recouvrant) contre **17 % en fine-tuning supervisé Llama3-70B** (32 H100, FSDP) — **un facteur 9 selon le régime**. Sur la même famille, Moonlight (Kimi Team, arXiv:2502.16982) mesure une latence d'optimiseur _« usually 1 % to 3 % »_ du forward-backward, avec une charge de communication dans $(1,;1{,}25]$ relativement à AdamW distribué `[ÉTABLI]`.

**Corollaire méthodologique.** Toute affirmation de surcoût d'un optimiseur du second ordre sans préciser (fréquence de recalcul, taille de batch, stratégie de parallélisme) est vide. Les quatre chiffres publiés — ≤ 10 % pour Distributed Shampoo, +77 % pour DASH-EVD, 26× pour Dion3, 1–3 % pour Moonlight — sont mutuellement cohérents parce qu'ils mesurent des choses différentes.

**Point à charge contre AdaFisher, exploitable directement.** Le papier (arXiv:2405.16397, ICLR 2025) ne publie **aucun tableau de temps par pas** : sa méthodologie repose sur un « Wall-Clock-Time avec cutoff » et une figure qualitative, sans valeurs numériques, sans modèle de GPU spécifié, sans décomposition du pas. L'affirmation « memory and time requirements on par with first-order methods » n'est donc **pas vérifiable sur les sources primaires accessibles** `[ÉTABLI comme constat d'absence]`.

#### 1.3.3 Grille de verdict par primitive algébrique

C'est le filtre à appliquer à toute technique candidate.

| Primitive                                                                        | Intensité arithmétique                       | Tensor cores                               | Verdict                                           | Justification chiffrée                                                                                                    |
| -------------------------------------------------------------------------------- | -------------------------------------------- | ------------------------------------------ | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **Polynôme matriciel** (Newton-Schulz, Newton couplé, Denman–Beavers, Chebyshev) | $O(n)$ FLOP/octet — régime produit matriciel | **Oui, plein régime**                      | ✅ **La primitive de choix**                       | 119 ms vs 3080 ms ; aucune dépendance interne, $T$ itérations = $T{\cdot}k$ produits batchables                           |
| **GEMM symétrique (SYRK)**                                                       | idem, moitié des tuiles utiles               | Oui, **mais cuBLAS ne l'exploite pas**     | ✅ excellente, noyau maison requis                 | Dion3/CuteDSL : 360,6 → 274,4 ms (**1,31×**)                                                                              |
| **Cholesky** (`potrf`)                                                           | récursion par blocs → GEMM                   | oui pour les mises à jour de blocs         | ✅ la seule factorisation dense tolérable          | utilisée par Zolo-pd pour cette raison                                                                                    |
| **QR** (`geqrf`+`orgqr`)                                                         | panel BLAS-2, trailing update BLAS-3         | partiellement                              | ~ acceptable si très rectangulaire                | 88,6 % du temps en QR pour $m=20,000$ ; « slightly more expensive than `torch.linalg.eigh` » en carré (Purifying Shampoo) |
| **FFT / Hadamard**                                                               | $O(\log n)$ FLOP/élément                     | oui **après reformulation en blocs 16×16** | ~ viable après reformulation                      | HadaCore : 1,1–1,4× (A100), 3,5× au pic                                                                                   |
| **Produit matrice-vecteur itératif** (Lanczos, power iteration, CG)              | ≈ 2 FLOP/octet                               | non                                        | ~ bon pour 1–2 valeurs propres, jamais un spectre | séquentiel + 1 synchronisation par itération                                                                              |
| **Eigendécomposition dense** (`syevd`, `syevj`)                                  | BLAS-2                                       | **non**                                    | ❌ **à proscrire du chemin critique**              | 1,3–1,5 % du pic ; 3080 ms/pas                                                                                            |
| **SVD dense** (`gesvd`)                                                          | idem, pire                                   | **non**                                    | ❌ à proscrire                                     | 70× plus lent que le CPU en petit lot                                                                                     |
| **Scatter-add creux (atomiques)**                                                | ≈ 1 FLOP/octet + sérialisation               | non                                        | ❌ médiocre                                        | `[EXTRAPOLATION]`                                                                                                         |
| **Récursion arborescente** (divide & conquer, élimination creuse)                | faible, dominée par la latence de cache      | non                                        | ❌ mauvaise sauf pipelining explicite              | occupation qui s'effondre aux niveaux hauts                                                                               |
| **Collectives NCCL petites**                                                     | latence-bound                                | —                                          | ❌ toujours coalescer                              | plancher **25 µs** (dont 15–17 µs hôte) ; 80–90 % du pic seulement à 16–32 Mio/lien                                       |
| **Lancement de noyaux (boucle Python par couche)**                               | —                                            | —                                          | ❌ résolu par CUDA graphs                          | temps CPU **0,1 ms constant** sous capture (Dion3)                                                                        |

**Synthèse.** L'écart entre les deux régimes est de un à trois ordres de grandeur en débit effectif. Aux tailles rencontrées ($n\le8192$), il **domine toute considération de complexité asymptotique**. C'est le critère qui gouverne le classement de la §4, et il explique pourquoi plusieurs pistes théoriquement séduisantes (matrices hiérarchiques à admissibilité forte, factorisation butterfly, rang de déplacement) sont classées basses malgré une complexité asymptotique favorable.

### 1.4 État de l'art et failles documentées

|Méthode|Référence vérifiée|Principe|Faille structurelle|
|---|---|---|---|
|K-FAC|Martens & Grosse, arXiv:1503.05671, ICML 2015|$A_\ell\otimes G_\ell$ par couche ; **prescrit la Fisher échantillonnée**|troncature au rang 1 de Kronecker (§2.B.1) ; bloc-diagonalité ; coût de la décomposition|
|K-FAC conv.|Grosse & Martens, arXiv:1602.01407|extension aux convolutions|idem|
|EKFAC|George, Laurent, Bouthillier, Ballas, Vincent, arXiv:1806.03884, NeurIPS 2018|ré-estime les valeurs propres dans la base de Kronecker|corrige **les valeurs propres, pas la base** ; garantie explicitement en norme de Frobenius (§2.B.3) ; décomposition coûteuse|
|K-FAC expand/reduce|Eschenhagen, Immer, Turner, Schneider, Hennig, arXiv:2311.00636, NeurIPS 2023|formulations pour le partage de poids|énonce le péché originel : « approxime une somme de produits de Kronecker par un produit de Kronecker de sommes »|
|TEKFAC|Gao et al., arXiv:2011.13609, AAAI 2021|trace conservée|même bloc-diagonalité ; substitue l'empirique|
|SINGD|Lin, Dangel, Eschenhagen, Neklyudov, Kristiadi, Turner, Makhzani, arXiv:2312.05705, ICML 2024|structure creuse **à l'intérieur** de chaque facteur ; sans inversion|reste au rang de Kronecker 1|
|Shampoo|Gupta, Koren & Singer, ICML 2018|préconditionneur Kronecker (borne AdaGrad matriciel)|racines $p$-ièmes coûteuses ; bloc-diagonalité|
|Distributed Shampoo|Shi et al., arXiv:2309.06497|DTensor + AllGather|≤10 % de surcoût **seulement à fréquence 100–1000**|
|SOAP|Vyas, Morwani, Zhao, Kwun, Shapira, Brandfonbrener, Janson, Kakade, arXiv:2409.11321, ICLR 2025|Adam dans la base propre de Shampoo|dégradation documentée quand la fréquence de recalcul de la base baisse|
|AdaFisher|Martins Gomes, Zhang, Belilovsky, Wolf, Hosseini, arXiv:2405.16397, ICLR 2025|Fisher bloc-Kronecker diagonale substituée au second moment d'Adam|hérite de tout K-FAC/EKFAC ; validité restreinte au cas $F=$ GGN ; **aucun chiffre de wall-time publié**|
|FisherAdapTune|Rostami, Chen & Hosseini, **arXiv:2606.10196**, _Fisher-Guided Progressive Parameter Selection for Adaptive Fine-Tuning_, 2026-06-08|gel progressif guidé par la dérive de Fisher|dérive mesurée sur l'empirique brute, sans test de significativité (§2.E.4, §3.5)|
|Fisher empirique brute|—|réutilise le gradient déjà calculé|**change d'objet mathématique** (§3.1) ; biais de « projection à échelle inversée »|

_Note bibliographique._ Le nom « FisherAdapTune » n'apparaît dans aucun titre arXiv : la méthode est introduite dans le corps du résumé de arXiv:2606.10196. Mahdi S. Hosseini est co-auteur des deux articles (AdaFisher et FisherAdapTune), même laboratoire — ce qui explique le partage de l'estimateur $H_D\otimes S_D$ entre les deux, et justifie de traiter leurs faiblesses comme corrélées plutôt qu'indépendantes.

---

## 2. Question 1 — Approximer la Fisher empirique : techniques candidates

Les techniques sont regroupées **par faille attaquée**, pas par domaine d'origine, parce que c'est la faille qui décide de leur pertinence. Chaque sous-section indique le domaine source.

Rappel de la question posée : ces techniques résolvent-elles le problème de wall-time, et sont-elles adaptées au matériel ? La grille de la §1.3.3 fournit le verdict matériel ; il est rappelé pour chaque entrée.

---

### 2.A — Le mur spectral : supprimer toute décomposition propre

_Domaines sources : algèbre linéaire numérique (théorie des fonctions de matrices), HPC._

C'est le levier de loin le mieux établi, et le seul dont le gain soit déjà mesuré à grande échelle.

#### 2.A.1 Itérations polynomiales pour la racine (inverse) matricielle `[ÉTABLI]`

**Mécanisme.** Calculer $A^{-1/2}$ ou $A^{-1/p}$ par une itération ne contenant que des produits matriciels.

- **Newton-Schulz** : $X_{k+1}=\tfrac12 X_k(3I-AX_k^2)$, convergence quadratique sous $|I-A|<1$ (normalisation préalable requise). Muon (Jordan et al., 2024) l'utilise pour l'orthogonalisation ; l'auteur justifie explicitement le choix par le matériel : la SVD est _« far too slow »_ et l'itération de Newton couplée _« must be run in at least float32 precision… which makes it slow on modern GPUs »_ — Newton-Schulz est retenu **parce qu'il tourne en bfloat16**. Surcoût FLOP mesuré : **0,7 %** sur le record NanoGPT.
- **Newton couplé** (Guo & Higham, _SIAM J. Matrix Anal. Appl._ 28(3):788–804, 2006) : forme couplée inconditionnellement stable, $2n^3(2+\theta\log_2 p)$ flops/itération, ~5 itérations.
- **Denman–Beavers** : itération couplée pour la racine carrée ; DASH optimise la première itération en forme close, ramenant le coût à **3 produits par itération**.
- **Polynômes minimax.** _Polar Express_ (Amsel, Persson, Musco & Gower, arXiv:2505.16932, ICLR 2026 Oral) résout à chaque itération un problème minimax sur l'intervalle courant des valeurs singulières, avec traitement explicite des erreurs d'arrondi le rendant praticable en bfloat16. _CANS_ (Grishina, Smirnov & Rakhuba, arXiv:2506.10935) dérive les coefficients optimaux par le théorème d'alternance de Chebyshev puis l'algorithme de Remez.
- **Schéma cubique relâché.** Huang (NVIDIA), _How Much Orthogonalization Does Muon Need?_, arXiv:2606.00371 : la qualité d'entraînement **n'est pas monotone en la précision de la décomposition polaire** ; il suffit d'amener les valeurs singulières dans la bande $[0{,}7,;1{,}3]$. Un schéma cubique en 5 pas y parvient avec **10 produits dominants contre 15**, soit **1,16–1,52× vs Polar Express** sur H200 ; l'écart de loss entre quatre méthodes (dont la SVD explicite en FP32) reste **inférieur à 0,003** sur GPT-2 Small.

**Ce que ça résout.** Le poste dominant du wall-time, en totalité. Zéro passe arrière supplémentaire : c'est du post-traitement algébrique sur des facteurs déjà calculés.

**Gain mesuré.** DASH, Llama-953M, bloc 1024, fréquence 1 : décomposition propre **3080 ms** → Newton couplé FP16 empilé **119 ms** (**25,9×**) ; Newton–Denman–Beavers avec normalisation par itération de puissance **177 ms** contre 740 ms en boucle Python (**4,18×**), **et perplexité 11,68 contre 11,80 pour la décomposition exacte** `[ÉTABLI]`.

**Le résultat qu'il faut retenir.** Une approximation matmul-only **bat la décomposition exacte en qualité**. Ce n'est pas un compromis vitesse/précision : l'inexactitude contrôlée de l'itération régularise le spectre. Toute défense d'AdaFisher fondée sur la fidélité de sa décomposition doit répondre à ce point.

**Verdict matériel.** ✅ Optimal. Produits matriciels exclusivement, aucune synchronisation, aucune réduction globale, empilable par lots sur toutes les couches.

**Réserve de précision — résultat négatif important `[ÉTABLI]`.** DASH mesure que le Newton couplé **converge en FP16 sans perte de qualité mais diverge en BF16**, « despite having the same range as FP32 » ; Newton–Denman–Beavers ne converge ni en FP16 ni en BF16. Le blog Gram-Newton-Schulz (Dao AI Lab) fait le même choix explicite de float16 plutôt que bfloat16. **La mantisse, pas la dynamique, est la ressource critique pour les itérations de racine matricielle.** C'est contre-intuitif dans une pratique bf16-first, et directement applicable à l'accumulation des covariances de Kronecker.

#### 2.A.2 Approximations rationnelles de Zolotarev `[ÉTABLI en NLA — EXTRAPOLATION en DL]`

**Mécanisme.** Les approximations rationnelles optimales au sens minimax de Zolotarev (1877) pour $\mathrm{sign}(x)$ et $x^{-1/2}$ ont une propriété unique : elles se **composent** — composer deux fonctions de degré $r$ donne l'approximant optimal de degré $r^2$. Nakatsukasa & Freund (_SIAM Review_ 58(3):461–493, 2016) obtiennent la décomposition polaire en **2 itérations en double précision**, l'itération étant d'**ordre de convergence 17** — contre 6 itérations pour QDWH (_QR-based dynamically weighted Halley_, ordre 3) et ~9 pour l'itération de Newton mise à l'échelle (ordre 2). Coût ≈ 3× en arithmétique totale, mais les $r$ factorisations sont **indépendantes** donc batchables.

**Statut.** Toute la littérature Muon/Shampoo 2025-2026 explore l'espace des polynômes ; **personne n'explore les rationnelles**. Le pari à mesurer : 2 itérations de QR/Cholesky par lots contre 10 itérations de produits matriciels. Sur des facteurs de taille LoRA ($d\le1024$) au conditionnement catastrophique de la Fisher, c'est le régime où le polynomial stagne le plus. **Réserve honnête** : Zolotarev utilise QR et Cholesky, primitives que la grille §1.3.3 classe « acceptable » et non « excellente ». À mesurer avant de croire.

#### 2.A.3 Restructurations GPU du même calcul `[ÉTABLI]`

- **Empilement 3D des blocs** (DASH) : au lieu d'une boucle Python couche par couche, empiler tous les blocs en tenseurs 3D traités par `bmm`. Gain **4,18× à 4,90×** — mais **aucun gain sur la décomposition propre** (2850 ms, identique que l'empilement soit fait par couche ou par GPU ; la configuration de référence à bloc 1024 et fréquence 1 est celle de la §1.3.2), confirmation directe que le batching ne sauve que ce qui est déjà en forme de produit matriciel.
- **Gram Newton-Schulz** (Dion3 / blog Dao AI Lab) : comme les polynômes utilisés sont impairs, $p(x)=x,h(x^2)$, on itère entièrement sur la matrice de Gram $n\times n$ au lieu de la rectangulaire. **−42 à −68 % de FLOPs** ; gains wall-clock 1,3× (Llama-430M) à 2× (Kimi K2, H100 et B300), perplexité préservée à ±0,01. Instabilité en demi-précision corrigée par un redémarrage après l'itération 2.
- **Noyaux GEMM symétriques (SYRK) sur tensor cores** : $A=a^\top a$ et $A^2$ sont symétriques ; un ordonnanceur triangulaire n'assigne que le triangle inférieur. Deux équipes indépendantes (Dion3/CuteDSL, NVIDIA/Triton) ont dû **écrire leurs propres noyaux parce que cuBLAS `syrk` n'exploite pas correctement les tensor cores**. Gain 1,31× dans Dion3.
- **CUDA graphs** : capture-replay du pas entier en un seul lancement. Temps CPU **0,1 ms constant** quelles que soient les configurations, y compris celles émettant « many small operations and kernel launches ». C'est le levier le plus sous-utilisé dans les implémentations K-FAC académiques, qui bouclent en Python couche par couche.
- **Megabatching des collectives** : regrouper toutes les matrices de même forme en un seul all-to-all, $O(1)$ rounds au lieu de $O(N/\text{world size})$. Justifié par micro-benchmark NCCL : plancher de latence **25 µs**, 80–90 % du pic seulement à partir de 16–32 Mio par lien, **~10 % du pic à 256 Kio**.

**Levier sous-exploité `[EXTRAPOLATION]`.** Le SYRK est _exactement_ l'opération de construction des facteurs K-FAC ($A_\ell=a^\top a$, $G_\ell=g^\top g$). Les noyaux SYRK tensor-core écrits pour Muon n'ont jamais été réutilisés pour K-FAC, alors que c'est la même opération, et que KAISA identifie ce poste comme **invariant au sharding** — donc irréductible par les techniques distribuées existantes. Gain attendu 1,3–2× sur un poste aujourd'hui incompressible.

#### 2.A.4 Décider _quand_ recalculer plutôt que fixer un intervalle `[ÉTABLI]`

Le levier complémentaire du précédent : §2.A.1 réduit le coût _par_ recalcul, celui-ci réduit leur _nombre_, et les deux se composent.

**L'état de l'art actuel est un intervalle fixe.** K-FAC sépare deux fréquences, $T_{\rm stats}$ (mise à jour EMA des facteurs) et $T_{\rm inv}$ (recalcul de l'inversion), toutes deux réglées à la main — le pseudocode figure explicitement dans plusieurs travaux (arXiv:1810.12281). Les gains sont réels : passer l'intervalle de courbure de 1 à 10 réduit le temps d'exécution de K-FAC d'environ 62 % sans dégrader la perte de test (arXiv:2603.29108) ; sur ImageNet, courbure tous les 10 pas et inverses tous les 200 ramènent le surcoût de K-FAC de 2,5× à 1,25× celui de SGD (arXiv:2203.08120). KAISA utilise $K_{\rm freq}=500$ / $F_{\rm freq}=50$. Mais SOAP documente que **la dégradation s'aggrave avec l'espacement**, et Purifying Shampoo mesure que le réglage fixe gagnant est loin de l'optimum.

**Purifying Shampoo** (Eschenhagen, Defazio, Lee, Turner & Shi, arXiv:2506.03595). Deux idées. (i) **Découpler la mise à jour des valeurs propres** — indispensable à chaque pas, c'est ce que le _grafting_ d'Adam compensait — **de celle de la base propre**, qui peut vieillir. (ii) Un **critère d'arrêt** inspiré de l'algorithme QR à démarrage chaud, bornant $\lVert L_t-\hat L_t\rVert_F/\lVert L_t\rVert_F$ par la norme des termes hors-diagonaux de $\hat Q^\top L_t\hat Q$ — quantité **moins chère à calculer que la décomposition qu'elle évite**, évaluée **par facteur**, donc autorisant des fréquences différentes par couche _et_ par facteur. Gain mesuré : **−20 % de wall-clock** face au réglage AlgoPerf gagnant (recalcul tous les 100 pas) sur Imagewoof/ViT, avec **5/5 seeds convergentes contre 3/5** et 7117±328 pas contre 8320±1203 `[ÉTABLI]`.

**Résultat négatif très utile du même travail `[ÉTABLI]`** : l'algorithme QR à démarrage chaud est _légèrement plus cher_ que `torch.linalg.eigh` dès que le critère n'est pas satisfait, et le réglage par défaut de SOAP (une itération de la méthode d'itération simultanée tous les 10 pas) est **plus lent en wall-clock** que le recalcul adaptatif de `eigh`, avec une perte finale pire — en contradiction avec la figure correspondante de Vyas et al.

**FOAM** (Nam & Ahn, arXiv:2606.02365) pilote la même décision par une boucle de rétroaction sur le damping, avec un proxy d'erreur d'opérateur évaluable **dans les bases propres périmées** : mise à jour du facteur gauche, du facteur droit, puis un critère $\mathcal U(\cdot)$ qui gouverne conjointement l'adaptation du damping et le rafraîchissement de la base. Directement transposable à EKFAC/AdaFisher, et complémentaire de l'estimation spectrale par quadrature de Lanczos stochastique (§2.E.5).

**Verdict matériel.** ✅ Le critère est un produit matriciel $\hat Q^\top L\hat Q$ suivi d'une norme — trivialement batchable sur toutes les couches. Il n'introduit aucune primitive nouvelle.

**La combinaison non faite `[EXTRAPOLATION]`.** Purifying Shampoo compare QR à `eigh`, **jamais QR à Newton–Denman–Beavers**, alors que DASH montre que NDB bat la décomposition exacte en perplexité tout en étant 16× plus rapide. Empiler (i) le critère de fraîcheur par facteur, (ii) un solveur matmul-only, (iii) l'empilement 3D des facteurs « frais » d'un même pas en un seul `bmm` : les trois leviers sont orthogonaux et leurs gains devraient se composer multiplicativement. C'est le résultat de nature systèmes le plus accessible du rapport.

**Lien direct avec FisherAdapTune.** C'est exactement sa structure — un signal de dérive décide quoi geler — transposée du gel de paramètres au gel de préconditionneur. Deux différences exploitables comme critique : le critère y est **borné et calculable a priori**, non heuristique ; et il est **par facteur**, non par couche. Même remarque pour CacheMuon (Dev, Bohara, Takáč & Horváth, arXiv:2606.16371), qui exploite la corrélation temporelle des facteurs polaires successifs avec une analyse en « Muon inexact » où l'erreur est bornée par celle du solveur frais plus la péremption du cache.

---

### 2.B — La structure : sortir du rang 1 de Kronecker

_Domaines sources : algèbre matricielle structurée (calcul scientifique), décompositions tensorielles, apprentissage de matrices structurées._

#### 2.B.1 Le réarrangement de Van Loan–Pitsianis : le cadre exact qui contient K-FAC `[ÉTABLI en algèbre — EXTRAPOLATION sur la Fisher]`

**Mécanisme.** Le _réarrangement_ $\mathcal R(M)$ d'une matrice bloc empile $\mathrm{vec}(M_{ij})^\top$ en lignes. C'est une **isométrie de Frobenius** qui transforme $$|M-B\otimes C|_F \longmapsto |\mathcal R(M)-\mathrm{vec}(B)\mathrm{vec}(C)^\top|_F.$$ Donc : le meilleur produit de Kronecker est le meilleur rang 1 de $\mathcal R(M)$, et **la meilleure somme de $r$ produits de Kronecker est exactement la SVD tronquée au rang $r$ de $\mathcal R(M)$** (Van Loan & Pitsianis, Cornell TR, nov. 1992 ; symétrie, définie-positivité et structure de bande sont héritées par les facteurs approchants).

**Application au bloc de Fisher.** Le gradient par échantillon d'une couche linéaire est $ga^\top$, donc $\mathrm{vec}(\nabla_{W_\ell})=a\otimes g$ et le bloc **exact** vaut $F_\ell=\mathbb{E}[(aa^\top)\otimes(gg^\top)]$, d'où $$\mathcal R(F_\ell)=\frac1N\sum_{n=1}^N \mathrm{vec}(a_na_n^\top),\mathrm{vec}(g_ng_n^\top)^\top.$$ **Réserve de validité, à ne pas escamoter.** Cette identité suppose qu'un échantillon produit **une seule** paire $(a,g)$ par couche. Dès qu'il y a partage de poids — séquence de longueur $T$ dans un transformeur, convolution — le gradient par échantillon vaut $\sum_t g_ta_t^\top$ et son produit extérieur contient les termes croisés $t\ne t'$, que $\sum_t(a_ta_t^\top)\otimes(g_tg_t^\top)$ ne reproduit pas. Tout ce qui suit doit donc être lu dans la convention _expand_ d'Eschenhagen et al. (arXiv:2311.00636), où les couples $(n,t)$ sont traités comme des échantillons distincts — le rang de Kronecker devient alors $\le\min(NT,d^2,p^2)$. La convention _reduce_, qui moyenne sur $t$ avant de former les facteurs, correspond à un autre objet, et l'écart entre les deux est précisément ce qu'Eschenhagen et al. formalisent. **En PEFT sur transformeur — le terrain visé en §5.2 — c'est la convention `expand` qu'il faut employer pour que le diagnostic ci-dessous ait un sens.**

Deux conséquences rigoureuses, sous cette convention :

1. Le **rang de Kronecker** de $F_\ell$ (le rang de $\mathcal R(F_\ell)$) vaut au plus $\min(N,d^2,p^2)$. La Fisher d'une couche est donc, en général, non pas un produit de Kronecker mais une **somme de $N$ produits de Kronecker** ; elle ne se réduit à un produit unique que si $\mathcal R(F_\ell)$ est de rang 1.
2. **K-FAC n'est même pas la meilleure troncature au rang 1.** Il prend le produit extérieur des _moyennes_ des facteurs ; l'optimum de Frobenius est le produit extérieur des _vecteurs singuliers dominants_. Sous l'hypothèse d'indépendance $\mathbb{E}[X\otimes Y]=\mathbb{E}[X]\otimes\mathbb{E}[Y]$, $\mathcal R(F_\ell)$ est de rang 1 et les deux coïncident. Hors de ce cas, K-FAC n'est pas l'optimum de rang 1 ; la réciproque exacte (rang 1 $\Rightarrow$ K-FAC optimal) n'est pas démontrée ici et ne l'est pas non plus dans Van Loan–Pitsianis `[à établir]`.

**Le diagnostic qui en découle — la recommandation la moins chère de tout le rapport.** $\sigma_2/\sigma_1$ mesure le biais d'indépendance ; $\sum_{i>1}\sigma_i^2/\sum_i\sigma_i^2$ mesure l'énergie de courbure que K-FAC jette. Couche par couche, au fil de l'entraînement. On ne forme **jamais** $\mathcal R(F_\ell)$ : le produit implicite $\mathcal R(F_\ell)x=\frac1N\sum_n\mathrm{vec}(a_na_n^\top)(g_n^\top X g_n)$ coûte $O(N(d^2+p^2))$ par vecteur, entièrement en produits matriciels par lots. Une SVD randomisée de rang $r$ coûte $r$ fois le coût de constitution des facteurs K-FAC.

**Sous-produit pour FisherAdapTune.** La dérive du **sous-espace de Kronecker dominant** (angles principaux entre les $u_1,v_1$ à deux instants) est un critère de gel strictement plus informatif que la dérive d'une norme scalaire de $H_D\otimes S_D$ : il distingue une couche dont la courbure change d'_échelle_ d'une couche dont elle change de _géométrie_. L'errata du projet (H7) note déjà que la distance de Jensen-Shannon sur les histogrammes ne voit qu'un aspect de la dérive ; le spectre singulier du réarrangement en voit un autre, et il est directement interprétable.

**Le piège honnête : l'inversion.** $(\sum_{i=1}^r B_i\otimes C_i)^{-1}$ **n'a pas de forme de Kronecker fermée** pour $r>1$. Trois issues connues du calcul scientifique : (i) $r=2$ → équation de Sylvester généralisée, résoluble par ADI (_alternating direction implicit_) — voir Voet, arXiv:2307.07884, _Numer. Linear Algebra Appl._ 2025 ; (ii) gradient conjugué **préconditionné par le terme de rang 1** (donc littéralement par K-FAC), le produit matrice-vecteur exact coûtant $r$ paires de produits via $(B_i\otimes C_i)\mathrm{vec}(X)=\mathrm{vec}(C_iXB_i^\top)$, typiquement 3 à 5 itérations ; (iii) série de Neumann tronquée autour du terme dominant. **C'est le vrai arbitrage : on gagne en fidélité, on perd la solution en forme fermée.**

**Verdict matériel.** ✅ Excellente — produits matriciels par lots sur des tenseurs réguliers, aucune récursion, aucune taille hétérogène. La structure la plus GPU-native du rapport.

**État de virginité.** DyKAF (Yudin, Grishina, Veprikov, Beznosikov, Rakhuba, arXiv:2511.06477) suit dynamiquement le rang-1 optimal par intégrateurs à séparation de projecteurs et **s'arrête là**. KoPA (Cai, Chen & Xiao, JMLR 23, 2022) fait du rang $K$ avec sélection automatique de configuration, mais sur des matrices statiques hors ML. Voet & De Novellis (arXiv:2510.25292) comblent le point aveugle « quelles tailles de facteurs ». **Personne ne fait $r>1$ sur la Fisher.**

#### 2.B.2 Structures conçues pour le GPU : Monarch, BTT `[ÉTABLI comme couches — EXTRAPOLATION comme classe de courbure]`

**Monarch** (Dao et al., arXiv:2204.00595, ICML 2022) : $M=PLP^\top R$ avec $L,R$ bloc-diagonales et $P$ une permutation de transposition ; le produit se ramène à deux produits matriciels par lots séparés par des `reshape`. Trois propriétés décisives : (i) _Théorème 1_ — la projection $\min_{M\in\mathcal M}|A-M|_F$ admet une **solution analytique**, décomposée en $b^2$ problèmes de rang 1 indépendants (analogue exact d'Eckart–Young), donc utilisable comme classe d'approximation de courbure sans optimisation itérative ; (ii) $A\otimes G$ est le cas **dégénéré** de Monarch où tous les blocs sont proportionnels — Monarch les libère et capture donc de la corrélation activation/gradient ; (iii) **l'inverse d'un produit de bloc-diagonales inversibles reste un produit de bloc-diagonales inversibles** — on récupère l'inversion en forme close que le Kronecker de rang $r$ perd. **Réserve** : une matrice de Monarch n'est en général ni symétrique ni semi-définie positive, donc cette classe perd la garantie que §2.C.1 érige en critère pour un préconditionneur ; il faudrait la restreindre à un sous-ensemble symétrique.

**BTT** (Qiu, Potapczynski, Finzi, Goldblum, Wilson, arXiv:2406.06248, ICML 2024 ; Potapczynski et al., arXiv:2410.02117, NeurIPS 2024) paramètre **tout** l'espace des opérateurs linéaires exprimables en Einsum par $\theta\in[0,1]^7$ et en extrait trois quantités : $\omega$ (partage de paramètres), $\psi$ (exposant de rang), $\nu$ (ressemblance au dense). Conclusions mesurées : les structures **sans partage de paramètres** ($\omega=0$) et de **rang plein** ($\psi=1$) dominent ; **$\nu$ — la « GPU-friendliness » brute — a un impact minime sur les lois d'échelle**, ce qui est un contre-argument sérieux à l'intuition « il faut du dense ».

**Transposition.** $A\otimes G$ est de rang plein ($\psi=1$, bon) mais avec un **partage de paramètres maximal** ($\omega\gg0$, mauvais : chaque bloc $A_{ij}G$ réutilise entièrement $G$). Le diagnostic de Wilson et al. prédit donc directement qu'une somme de $r$ Kronecker ou une matrice Monarch bat K-FAC à budget mémoire égal `[EXTRAPOLATION bien fondée]`.

**Nuance matérielle à ne pas escamoter `[ÉTABLI]`.** Gonon, Zheng, Carrivain & Le (arXiv:2405.15013) donnent le premier benchmark GPU systématique des produits Kronecker-creux : les implémentations existantes passent **jusqu'à 50 % de leur temps en réécritures mémoire** (permutations limitées par la bande passante). Leur noyau CUDA fusionné donne 1,4× de gain médian sur 600 motifs et −22 % de latence sur ViT-S/16. **La structure est GPU-friendly, les permutations ne le sont pas** : sans noyau fusionné on perd la moitié du gain. Il faut appliquer symétriquement à toute alternative proposée la critique de wall-time qu'on adresse à AdaFisher.

#### 2.B.3 Le mauvais critère : Frobenius contre norme spectrale `[ÉTABLI]`

Argument théorique décisif et systématiquement ignoré côté ML. L'isométrie du réarrangement vaut **en norme de Frobenius uniquement**. Dressler, Uschmajew & Chandrasekaran (arXiv:2207.03186) exhibent (exemple 2.1) un opérateur pour lequel la solution SVD a une erreur spectrale $1$ alors que l'optimum spectral vaut $\sigma_1/\sqrt{(m-1)(n-1)}$ — **un écart qui croît avec la dimension**.

Or, pour un préconditionneur, la quantité qui gouverne la convergence est le conditionnement de $\tilde F^{-1}F$, soit une erreur **relative en norme spectrale**, pas une erreur de Frobenius. Toute la littérature — K-FAC, EKFAC (dont la garantie est explicitement $|\cdot|_F$), Van Loan–Pitsianis lui-même, et le cadre unificateur de Gong, Scetbon, Ma & Meeds (arXiv:2502.07752, qui montre qu'Adam, Shampoo et SOAP sont tous des solutions de $\min_{\tilde F\in\mathcal H}|\tilde F-F|_F^2$ pour différentes classes $\mathcal H$) — **optimise le mauvais critère**.

**Programme concret.** Calculer hors ligne, sur de petites couches, l'optimum spectral par programme semi-défini (SDP) alterné et mesurer de combien K-FAC et le rang-$r$ Frobenius s'en écartent. Si l'écart est grand, c'est une critique théorique de fond contre AdaFisher, pas une remarque de forme. Verdict matériel du programme semi-défini : ❌ mauvais — c'est un outil d'analyse hors-ligne, pas une brique d'entraînement.

#### 2.B.4 Décompositions tensorielles `[ÉTABLI en calcul scientifique — EXTRAPOLATION sur la Fisher]`

- **Kilmer & Saibaba** (arXiv:2105.01170) généralisent §2.B.1 en trois directions : plus de deux facteurs (pertinent pour conv/attention : canaux × spatial × tête) ; le format **Tucker introduit un noyau de couplage** entre facteurs gauche et droite — précisément l'objet capable de représenter la corrélation activation/gradient que K-FAC annule ; et le Lemme 4 garantit la conservation exacte de l'erreur de Frobenius entre tenseur et matrice.
- **Tensor-Train** (Oseledets, _SIAM J. Sci. Comput._ 33(5), 2011) : deux usages à ne pas confondre. Sur les _facteurs_ (compression intra-facteur, comme SINGD) : n'attaque pas le biais d'indépendance. Sur l'**axe des couches** : le rang TT joue le rôle d'entropie d'intrication entre couches voisines et représente les couplages inter-couches à décroissance rapide — l'analogue exact de ce que DMRG fait en physique à $N$ corps. Verdict matériel : ~ moyen (cœurs petits, chaîne de contractions séquentielle en profondeur) ; une chaîne **courte** ($d=3$–4) à gros cœurs est en revanche parfaitement GPU-friendly — c'est exactement ce qu'est BTT.
- **Tensor Normal Training** (Ren & Goldfarb, **arXiv:2106.02925**, NeurIPS 2021 Spotlight) : généralisation tensorielle complète de la structure Kronecker de K-FAC pour la courbure — la référence ML la plus proche de cette piste.
- **K-FOC** (Schnaus, Lee & Triebel, _Kronecker-Factored Optimal Curvature_, NeurIPS 2021 Bayesian Deep Learning Workshop, DLR elib 145806 — **pas d'identifiant arXiv**) : somme de produits de Kronecker résolue par itération de puissance comme meilleure approximation de rang 1 itérée. C'est le point de départ à battre pour §2.B.1.

---

### 2.C — Le rang déficient et les corrélations inter-couches

_Domaines sources : algèbre linéaire numérique randomisée (RandNLA), problèmes inverses à grande échelle, statistiques spatiales._

#### 2.C.1 Nyström randomisé et sa variante matmul-only `[ÉTABLI en RandNLA]`

**Mécanisme.** Pour $F\succeq0$ : tirer $\Omega\in\mathbb{R}^{P\times s}$, calculer $Y=F\Omega$, poser $\hat F=Y(\Omega^\top Y)^\dagger Y^\top$. C'est l'une des rares constructions par esquisse qui **préserve la semi-définie positivité par construction** — propriété critique pour un préconditionneur, que la troncature d'Eckart-Young et Frequent Directions possèdent aussi mais que le Nyström _généralisé_ (ci-dessous) perd. L'inverse par Woodbury coûte $O(Pm+m^3)$ au lieu de $O(P^3)$.

Références vérifiées : Halko, Martinsson & Tropp (arXiv:0909.4061, _SIAM Review_ 2011) ; Tropp, Yurtsever, Udell & Cevher (arXiv:1609.00048, SIMAX 2017) ; Frangella, Tropp & Udell, _Randomized Nyström Preconditioning_ (arXiv:2110.02820, SIMAX).

**Preuve de concept déjà en production sur la courbure** : RS-KFAC / SRE-KFAC (Puiu, arXiv:2206.15397) exploitent la décroissance spectrale des facteurs — **qu'ils démontrent imposée par l'EMA** — pour remplacer l'inversion exacte par une SVD/EVD randomisée, ramenant le coût cubique par couche à quadratique, **jusqu'à 3× plus rapide que K-FAC**. SketchySGD et PROMISE (Frangella, Rathore, Zhao & Udell, arXiv:2211.08597 et arXiv:2309.02014) font de même sur le Hessien sous-échantillonné avec **un rang $r\approx10$ suffisant** et une règle de pas dérivée du spectre esquissé — mais leurs garanties et expériences sont **convexes**, le passage au deep learning n'est pas validé.

**La variante qui manque : le Nyström généralisé `[EXTRAPOLATION]`.** Nakatsukasa (arXiv:2009.11392) : deux esquisses indépendantes $X\in\mathbb{R}^{P\times s}$, $Y\in\mathbb{R}^{P\times t}$ avec $t\approx1{,}5s$, et $\hat F=(FX)(Y^\top FX)^\dagger(Y^\top F)$ — le pseudo-inverse porte sur une matrice $t\times s$, **plus aucune QR sur la grande dimension**. Or la QR est précisément ce qui empêche les méthodes RandNLA appliquées à K-FAC d'atteindre le gain annoncé, et Milligan, Xu, Lacoste-Julien, Dangel & Lin (arXiv:2605.26327) documentent que la QR de Shampoo/SOAP **exige du FP32 et bloque le stockage bfloat16**. Personne n'a fait le lien. Risque identifié : perte de la PSD exacte, il faut resymétriser et quantifier l'effet sur le damping.

**Verdict matériel.** ✅ Excellente pour Nyström, ✅✅ pour le Nyström généralisé (le seul noyau non-GEMM est un $t\times s$ avec $s\le128$, complètement fusible en un graphe CUDA). En distribué, $Y=F\Omega$ est un all-reduce de $P\times s$ flottants, de même forme que celui des gradients donc fusionnable avec lui — mais **$s$ fois plus volumineux**, ce qui n'est acceptable qu'à $s$ petit et à fréquence amortie (la même critique que celle adressée à K-FAC en §1.3.2).

#### 2.C.2 Diagonale + rang faible, estimées conjointement `[ÉTABLI]`

**Mécanisme.** $F\approx D+UU^\top$, inversion par Woodbury en $O(Pm+m^3)$. C'est la réponse structurelle la plus directe au biais de Kronecker : $A\otimes G$ impose une structure produit sur _toute_ la matrice, tandis que $D+UU^\top$ n'impose rien sur la diagonale (ce qu'Adam capture bien) et met tout le budget de rang sur les corrélations dominantes, **y compris inter-couches** si $U$ est global.

**Résultat clé.** Fernandez, Dangel, Hennig & Schneider, _Sketching Low-Rank Plus Diagonal Matrices_ (SKETCHLORD, arXiv:2509.23587, sept. 2025) : l'estimation **jointe** diagonale + rang faible **domine strictement** toute estimation séquentielle (diagonale-puis-low-rank ou l'inverse), avec pour cible explicite le Hessien du deep learning. C'est un argument technique fort contre toute méthode qui empile un correctif de rang faible sur une diagonale préalablement estimée.

Précédents sur la Fisher : SENG (Yang, Xu, Wen, Chen & Xu, arXiv:2006.05924, ICML 2022 — ResNet-50/ImageNet à 75,9 % en 41 époques) ; TENGraD (Soori, Can, Mu, Gürbüzbalaban & Mehri Dehnavi, arXiv:2106.03947 — inversion **exacte** par bloc via Sherman-Morrison-Woodbury).

**Verdict matériel.** ✅✅ La plus « tensor-core-native » du rapport : Woodbury = produits matriciels $P\times m$ + un solve $m\times m$, aucune décomposition spectrale. **Mémoire $O(Pm)$ : rédhibitoire en pré-entraînement complet** (7·10⁹ paramètres × 16 en bf16 = 224 Go), **parfaitement viable en LoRA** ($10^7$–$10^8$ → 0,3–3 Go). C'est un argument de plus pour se placer en PEFT.

#### 2.C.3 Cholesky à pivots aléatoires piloté par le second moment d'Adam `[EXTRAPOLATION — briques ÉTABLIES]`

**Mécanisme.** RPCholesky (Chen, Epperly, Tropp & Webber, arXiv:2207.06503, _Comm. Pure Appl. Math._) : factorisation de Cholesky partielle où le $i$-ème pivot est tiré proportionnellement à la **diagonale résiduelle**. Version accélérée par blocs et rejet : arXiv:2410.03969.

**Pourquoi c'est un pont non exploité.** RPCholesky n'a jamais besoin de produits $Fv$ génériques : il lui suffit d'entrées $F_{ij}$ et de **colonnes** $Fe_j$ — or une colonne de la Fisher est un produit Fisher-vecteur avec un vecteur canonique, soit un unique R-op. Et surtout : **la diagonale initiale, qui fixe la distribution du premier pivot, est exactement le second moment que tout entraînement Adam maintient déjà** ($\tilde F_{jj}=\mathbb{E}[(\partial\ell/\partial\theta_j)^2]$). Deux réserves à ne pas escamoter : les pivots suivants exigent la diagonale **résiduelle**, qu'Adam ne fournit pas ; et cette diagonale est celle de la Fisher **empirique** $\tilde F$, dont la §3.1 établit qu'elle diffère de $F$ — la correction iEF (§3.5) s'applique en amont. On obtient une approximation **PSD, en facteur, sans aucune hypothèse de Kronecker**, donc capable de corrélations inter-couches, à partir d'une statistique gratuite. Risque : l'analyse suppose une décroissance spectrale ; la queue lourde du spectre de la Fisher pourrait exiger un $k$ élevé — à tester en premier. Verdict matériel : ✅ en version par blocs (la version séquentielle pivot-par-pivot est ❌).

#### 2.C.4 Sketching de produits de Kronecker : la régression de Kronecker ridge `[ÉTABLI en théorie — EXTRAPOLATION sur K-FAC]`

Diao, Song, Sun & Woodruff (arXiv:1712.09473, AISTATS 2018) ; Diao, Jayaram, Song, Sun & Woodruff (arXiv:1909.13384, NeurIPS 2019) ; Fahrbach, Fu & Ghadiri (arXiv:2209.04876, NeurIPS 2022) donnent des algorithmes de régression et d'approximation de rang faible pour un design $A_1\otimes\cdots\otimes A_q$ **sans jamais former le produit complet**, en temps $O(\sum_i \mathrm{nnz}(A_i))$.

**Le lien que personne n'a fait.** Le pas naturel de K-FAC, $(A\otimes G+\lambda I)^{-1}\mathrm{vec}(\nabla W)$, **est littéralement une instance de régression de Kronecker régularisée** — et le terme $\lambda I$, qui casse la factorisation exacte, est précisément le cas traité par la version _ridge_. Le résultat structurel central de ces travaux est que **les leverage scores d'un produit de Kronecker se factorisent** (produit des leverage scores des facteurs), ce qui fait disparaître le besoin de toucher au produit complet. Si le pas naturel se calcule sans diagonaliser, la justification centrale d'AdaFisher (approximer le spectre par une statistique bon marché) perd son motif. La cloison entre les deux littératures est totale : aucun de ces papiers ne cite le gradient naturel, aucun papier K-FAC ne les cite.

**Verdict matériel.** ~ Mitigé pour TensorSketch (dépend de FFT) et KFJLT (dépend de Hadamard) ; ✅ pour la version par échantillonnage de leverage scores, qui se réduit à des gathers + produits matriciels.

#### 2.C.5 Le cas contre les esquisses structurées classiques `[EXTRAPOLATION argumentée]`

Point méthodologique important, souvent mal évalué. La transformée de Hadamard rapide sous-échantillonnée (SRHT) réduit le coût d'application d'une esquisse de $O(Ps)$ à $O(P\log P)$. Mais c'est un noyau **limité par la bande passante** à motif papillon, qui n'utilise pas les tensor cores : sur H100/B200, une esquisse **gaussienne dense** $P\times s$ avec $s\le128$ est souvent **plus rapide** malgré un facteur $s/\log P$ de FLOPs en plus, parce qu'elle tourne à 400+ TFLOPS quand la FWHT tourne à la bande passante HBM. SRHT était le bon choix sur CPU en 2010. Nuance : HadaCore (blog PyTorch) reformule la FWHT en blocs 16×16 sur tensor cores et obtient 1,1–1,4× en moyenne, 3,5× au pic — la primitive redevient viable, mais après réécriture.

**Corollaire général, valable pour tout ce rapport : l'asymptotique induit systématiquement en erreur sur GPU aux tailles rencontrées.**

#### 2.C.6 Matrices hiérarchiques : ce qui survit réellement `[ÉTABLI — nuancé]`

_Définitions._ Une matrice $\mathcal H$ partitionne les indices par un arbre de clusters et comprime en rang faible les blocs « bien séparés ». **HODLR** est le cas faiblement admissible : tout bloc hors-diagonale est de rang faible, arbre binaire, aucune géométrie requise. **$\mathcal H^2$** ajoute des bases emboîtées ($O(n)$ au lieu de $O(n\log n)$). **HSS** = HODLR + bases emboîtées. **BLR** = un seul niveau, blocs uniformes.

**Le précédent direct existe.** Chen, Reiz, Yu, Bungartz & Biros (arXiv:1910.12184) construisent une approximation $\mathcal H$ de la GGN d'un perceptron multicouche via GOFMM (_Geometry-Oblivious FMM_) — point crucial, puisque les paramètres d'un réseau n'ont aucune géométrie et que le critère d'admissibilité classique est inapplicable ; GOFMM construit l'arbre à partir de distances induites par la matrice elle-même. Précalcul $O(Nnd)$, factorisation $O(Nr_o^2)$. Limite : MLP seulement, hors ligne, aucune comparaison wall-time à K-FAC.

**L'argument transposable.** Hartland, Stadler, Perego, Liegeois & Petra (arXiv:2301.03644, _Inverse Problems_ 2023) et Ambartsumyan et al. (arXiv:2003.10173, SISC 2020) établissent que quand les données deviennent informatives, **le rang global du Hessien croît mais les rangs locaux des blocs hors-diagonale restent bornés** — jusqu'à un ordre de grandeur de gain sur les approximations globalement de rang faible. Le format $\mathcal H^2$ y supporte matvec, résolution, inversion **et racine carrée inverse**, ce qui est exactement ce dont un préconditionneur a besoin.

**Le verdict GPU, avec la nuance décisive.** HODLR se traite **niveau par niveau** (donc à plat, pas récursivement) : Chen & Martinsson (arXiv:2208.06290) mesurent une factorisation de $N=2,097,152$ en **1,89 s** et ≈ **2 TFlop/s sur une V100**, avec `batched LU` + `batched GEMM`. H2Opus (Zampini, Boukaram, Turkiyyah, Knio & Keyes, arXiv:2109.05451) atteint **2,3 Tflop/s/GPU sur 1024 GPU V100** — **mais avec 64 vecteurs concurrents ; en mono-vecteur on tombe à ≈150 Gflop/s/GPU, soit 6 % du régime saturé**. Le matvec hiérarchique est limité par la mémoire et n'atteint la performance crête qu'en amortissant sur de nombreux seconds membres. **Pour un optimiseur qui applique $F^{-1}$ à un seul gradient par itération, on est structurellement dans le régime à 150 Gflop/s.** C'est l'objection wall-time la plus sérieuse à toute méthode hiérarchique en entraînement — et son contre-argument : en PEFT à micro-batchs multiples, ou en approximation de Laplace, les seconds membres multiples existent naturellement.

Boukaram, Liu, Ghysels & Li (arXiv:2506.16759) rendent la construction $\mathcal H^2$ **matrix-free** (esquisses + évaluation d'entrées) avec **256 échantillons aléatoires constants** contre jusqu'à 18 920 pour H2Opus en 3D, et 13× de gain sur leur propre CPU. C'est ce qui rend la piste réaliste : on ne sait faire, sur une Fisher, que des produits Fisher-vecteur.

**Ce qui ne survit pas.** HSS (bases emboîtées → dépendances séquentielles ascendantes/descendantes), HIF (squelettisation récursive à pivotage), factorisation butterfly ($O(\log N)$ facteurs appliqués en séquence, jusqu'à 50 % du temps en permutations), élimination creuse exacte / dissection emboîtée (supernœuds hétérogènes, les implémentations GPU efficaces ne le sont que sur les nœuds racines et laissent le reste au CPU). **BLR est le compromis à privilégier si l'on veut du hiérarchique sans payer la récursion.**

**Piste `[EXTRAPOLATION]`.** Ordonner les paramètres par profondeur, prendre $\mathrm{dist}(\ell,\ell')=|\ell-\ell'|$ comme géométrie 1-D (naturelle : le graphe de calcul est une chaîne) et poser $F=\mathrm{blkdiag}(A_\ell\otimes G_\ell)+\mathcal H_{\text{off}}$ avec $\mathcal H_{\text{off}}$ en HODLR faiblement admissible sur l'arbre binaire des couches. Justification de la faible admissibilité : la corrélation de gradient décroît avec $|\ell-\ell'|$ par contraction du produit de jacobiennes. La combinaison « K-FAC en diagonale + HODLR hors-diagonale sur la profondeur » n'existe pas. **Réserve sérieuse** : le résultat d'Abreu et al. (§1.2) suggère que l'information inter-couches rapporte peu — cette piste doit être précédée d'une mesure.

#### 2.C.7 Facteur de Cholesky creux de $F^{-1}$ par minimisation de Kullback-Leibler `[ÉTABLI en GP/EDP — EXTRAPOLATION totale sur la Fisher]`

**Mécanisme.** Schäfer, Katzfuss & Owhadi (arXiv:2004.14455, SISC 43(3), 2021) : au lieu de comprimer $\Theta$, chercher directement un facteur de Cholesky **creux de l'inverse** $L$ minimisant $\mathrm{KL}(\mathcal N(0,\Theta)|\mathcal N(0,(LL^\top)^{-1}))$ sous contrainte de motif de creux. Le minimiseur est **en forme close, colonne par colonne** : $L_{s_i,i}=\Theta^{-1}_{s_i,s_i}e_1/\sqrt{e_1^\top\Theta^{-1}_{s_i,s_i}e_1}$.

**Pourquoi c'est potentiellement la plus élégante des pistes.** C'est la seule méthode du rapport qui produit **directement le facteur de $F^{-1}$** — or c'est $F^{-1}$ ou $F^{-1/2}$ que l'optimiseur veut, jamais $F$. Elle contourne donc entièrement le problème d'inversion du Kronecker de rang $r$ (§2.B.1). Elle ne demande que des sous-blocs $F_{s_i,s_i}$ de petite taille, estimables par produits Fisher-vecteur, avec un motif $S$ **dicté par l'architecture** (bloc-diagonal par couche + bande sur l'axe des couches + connexions résiduelles). L'ordre « maximin » se transpose en un ordre par distance de corrélation de gradient.

**Verdict matériel.** ✅✅ Excellente et sous-estimée : chaque colonne est un petit système SPD **indépendant** ⇒ `batched Cholesky` sur des milliers de blocs de taille 20–50, le motif d'accès idéal pour cuBLAS batched. Aucune récursion, aucune dépendance inter-colonnes. Contraste instructif avec §2.C.6 : c'est **l'abandon de l'exactitude qui restaure la régularité**.

---

### 2.D — La mise à jour en ligne : sortir de l'EMA

_Domaines sources : filtrage de Kalman, identification de systèmes, traitement du signal adaptatif, algèbre linéaire en flux._

Toutes les méthodes de cette section attaquent le même point : K-FAC, EKFAC, AdaFisher et FisherAdapTune lissent les facteurs par une moyenne mobile exponentielle scalaire, puis paient périodiquement une décomposition qui est **périmée entre deux recalculs**.

#### 2.D.1 Le cadrage : l'EMA n'est pas la mise à jour canonique `[ÉTABLI]`

Ollivier (_Online Natural Gradient as a Kalman Filter_, arXiv:1703.00209) prouve que le gradient naturel en ligne **est** un filtre de Kalman étendu, la covariance a posteriori jouant le rôle de $\hat F^{-1}$. La mise à jour canonique est donc la récursion de Riccati de l'information, $P_t^{-1}=\lambda P_{t-1}^{-1}+J_t^\top R^{-1}J_t$ : une **accumulation additive dans le domaine de la précision**, avec un bruit de processus $Q$ qui modélise explicitement la dérive. L'EMA sur $A$ et $G$ _séparément_ n'est équivalente à **aucune** récursion de Riccati exacte : c'est une heuristique. C'est le critère de correction contre lequel juger tout le reste.

#### 2.D.2 Mise à jour exacte de la décomposition — équation séculaire `[ÉTABLI en algèbre — EXTRAPOLATION sur K-FAC]`

**Mécanisme.** Si $A=U\Lambda U^\top$ est connue, les valeurs propres de $A+\sigma vv^\top$ sont les racines de l'**équation séculaire** $1+\sigma\sum_i z_i^2/(\lambda_i-\lambda)=0$ avec $z=U^\top v$ ; elles s'entrelacent strictement avec les anciennes, et les nouveaux vecteurs propres s'écrivent en forme close (matrice de Cauchy). Bunch, Nielsen & Sorensen (_Numer. Math._ 31:31–48, 1978) ; Gu & Eisenstat (SIMAX, version stable) ; Brand (_Linear Algebra Appl._ 415(1), 2006) pour la SVD mince.

**Pourquoi c'est direct.** L'EMA de K-FAC, $\hat A_t=\beta\hat A_{t-1}+(1-\beta)a_ta_t^\top$, est **littéralement un scaling ($\Lambda\leftarrow\beta\Lambda$, gratuit, base inchangée) plus une modification de rang 1**. Les **valeurs propres exactes** sont donc disponibles à $O(n^2)$ par pas au lieu de $O(n^3)$ tous les $T$ pas, et elles ne sont jamais périmées. La base propre, elle, reste à $O(n^3)$ : voir la réserve de coût dans le verdict matériel ci-dessous.

**Le sous-produit théorique, directement branchable sur `fisher_drift_analysis`.** L'entrelacement de Cauchy donne des bornes **gratuites et en forme close** sur le déplacement du spectre : $\lambda_i\le\tilde\lambda_i\le\lambda_{i-1}$ et $\sum_i(\tilde\lambda_i-\lambda_i)=\sigma|v|^2$. Autrement dit **la dérive spectrale d'une couche est prédictible avant d'être mesurée**, à partir de la seule norme du batch d'activations. Cela remplacerait le critère de gel de FisherAdapTune par un critère analytique en $O(n)$, et fournirait de surcroît un critère de rafraîchissement adaptatif de la décomposition (rafraîchir quand la borne cumulée dépasse un seuil, pas tous les $T$ pas).

**Verdict matériel — argument détaillé, car c'est le point délicat.** Favorable : la résolution des $n$ racines séculaires est _embarrassingly parallel_ (un thread par racine, 3–5 Newton, intervalles disjoints) ; la reconstitution $U\leftarrow UC$ est un produit matriciel dense au pic FLOPS — mais elle coûte $n^3$, donc **elle annule le gain asymptotique si on la fait à chaque pas** (le schéma viable est d'accumuler $\prod C_t$ et de ne l'appliquer que périodiquement, ce que LAPACK `dlaed*` exploite déjà). Défavorable : la **déflation** (traitement des $z_i\approx0$ et des valeurs propres quasi-confondues), indispensable à la stabilité, est un filtrage à branchement dépendant des données → divergence de warps ; il faut la remplacer par une compaction par tri/scan (primitives CUB). Verdict honnête : **matmul-friendly à 80 %, avec une phase de déflation hostile**. C'est jouable, ce n'est pas gratuit, c'est un vrai travail de noyau.

**Précédent partiel.** Puiu, _Brand New K-FACs_ (arXiv:2210.08494) exploite exactement cette idée par mise à jour de type Brand, ramenant le coût de mise à jour **de cubique à linéaire** en taille de couche (contre quadratique pour RS-KFAC), avec un compromis précision/vitesse quantifié par quatre métriques d'erreur — non gratuit, donc à valider.

#### 2.D.3 Suivi continu de la variété de rang faible `[ÉTABLI — EXTRAPOLATION sur la Fisher]`

**Mécanisme.** Contraindre la covariance à une variété de rang $r$ et faire évoluer **la variété elle-même** (base + valeurs propres) par une équation différentielle projetée, au lieu de recalculer une décomposition tronquée. Bonnabel & Sepulchre (arXiv:1203.4049) ; Schmidt, Hennig, Nick & Tronarp (arXiv:2306.07774) ; Nobile & Trigo Trindade (arXiv:2509.11210).

**Réponse de principe au recalcul périodique.** Coût $O(nr^2+r^3)$ par pas contre $O(n^3)$ tous les $T$ pas — pour $n=4096$, $r=128$, $T=100$ : $6{,}7\cdot10^7$ contre $6{,}9\cdot10^8$ amorti, **et sans base périmée**. Verdict matériel : ✅ produits matriciels $n\times r$ + petite factorisation $r\times r$ ; réserve sur les intégrateurs à splitting projeté (3 sous-étapes séquentielles avec une QR chacune — regrouper les couches en lots).

**DyKAF** (arXiv:2511.06477) fait exactement cela sur la Fisher, intégré à SOAP, testé en pré-entraînement et fine-tuning de LLM — **mais reste au rang de Kronecker 1**. La combinaison naturelle est : suivi dynamique **de la SVD du réarrangement au rang $r$** (§2.B.1), qui n'existe pas.

Variante voisine et intéressante : _Stochastic Reconfiguration with Warm-Started SVD_ (Zhou, Chen, Ho, Liu & Ortner, arXiv:2512.05749) réutilise la SVD du pas précédent comme point de départ — algorithmiquement plus honnête que l'EMA, qui suppose implicitement que les Fisher de mini-batchs successifs sont commensurables dans une base fixe, ce qui est faux dès que la base propre tourne.

#### 2.D.4 Subspace tracking et esquisses en flux `[ÉTABLI]`

- **PAST / Oja / GROUSE** : suivre en ligne le sous-espace dominant à $O(nr)$ par échantillon avec facteur d'oubli intégré, sans jamais former la covariance. Yang (IEEE TSP 43(1), 1995) ; Balzano, Nowak & Recht (arXiv:1006.4046) ; Balzano (PMLR 151, 2022, équivalence Oja ↔ GROUSE) ; Huang, Niles-Weed & Ward (arXiv:2102.03646, « beyond rank-one updates »). Point souvent ignoré : **Kumar & Sarkar (arXiv:2305.02456) analysent Oja sous données markoviennes**, c'est-à-dire corrélées dans le temps — le régime réel des activations, que l'analyse i.i.d. sous-jacente à l'EMA ignore. Verdict matériel : ✅ si l'on traite le mini-batch entier comme une mise à jour de rang $b$ (produits matriciels) ; ❌ pour PASTd, séquentiel en $r$.
- **Frequent Directions** (Liberty, arXiv:1206.0594 ; Ghashami, Liberty, Phillips & Woodruff, arXiv:1501.01711) : esquisse **déterministe** avec garantie bilatérale $0\preceq A^\top A-B^\top B\preceq(|A-A_k|_F^2/(\ell-k))I$ — l'esquisse sous-estime uniformément, ce qu'aucune esquisse aléatoire ne garantit. **Sketchy** (arXiv:2302.03764, NeurIPS 2023) l'applique aux facteurs de Shampoo et **mesure la dimension intrinsèque des facteurs de Kronecker à moins de 105 pour une dimension nominale > 1024**. C'est le concurrent direct le plus sérieux d'AdaFisher, et c'est aussi le fondement empirique qui manque à AdaFisher lui-même. Point faible : la SVD périodique de FD — remplaçable par un Nyström généralisé (§2.C.1), au prix de la garantie déterministe.
- **Fenêtre glissante à garantie spectrale `[EXTRAPOLATION]`.** Le modèle _sliding window_ (Braverman, Drineas, Musco, Musco, Upadhyay, Woodruff & Zhou, arXiv:1805.03765, FOCS 2020 ; Yao, Chen & Chen, arXiv:2502.18830 pour le résultat optimal du produit matriciel approché sur fenêtre) maintient une approximation $(1\pm\varepsilon)$ sur les $W$ derniers pas via $O(\log W)$ esquisses imbriquées. **L'EMA est un oubli implicite sans garantie** : aucun résultat ne borne l'écart entre $\mathrm{EMA}_\beta$ et la vraie covariance de fenêtre, et $\beta$ est calibré à l'œil. Conséquence directement exploitable : **la « dérive de Fisher » mesurée par FisherAdapTune est confondue avec l'artefact de l'EMA** ; avec une fenêtre glissante on peut mesurer la dérive _vraie_ (différence entre deux fenêtres disjointes, chacune à garantie $(1\pm\varepsilon)$) et vérifier si le signal de gel survit. C'est une expérience falsifiante nette. Verdict matériel : ✅ les $O(\log W)$ esquisses sont indépendantes donc batchables ; coût $\times\log W\approx14$ pour $W=10^4$.

#### 2.D.5 Oubli directionnel : la pathologie de l'EMA scalaire `[ÉTABLI en identification — EXTRAPOLATION sur K-FAC]`

**Le diagnostic.** Les moindres carrés récursifs à facteur d'oubli $\lambda$ souffrent d'une pathologie connue et documentée depuis quarante ans : le _covariance blow-up_ / _estimator windup_. Dans les directions où le régresseur n'est pas excité, l'oubli exponentiel divise la précision par $\lambda$ à chaque pas **sans qu'aucune information nouvelle ne la reconstitue** ; la covariance diverge exponentiellement dans ces directions.

Or les activations d'une couche sont **fortement anisotropes et concentrées dans un sous-espace de dimension effective $\ll n$**. Donc : _l'EMA scalaire de K-FAC fait diverger l'estimation de $A$ dans les directions non excitées ; le damping $\varepsilon I$ masque le symptôme sans traiter la cause, et l'échelle relative entre directions excitées et non excitées est arbitraire._

**Les remèdes, transposables et non transposés.** Oubli directionnel / SIFt-RLS (Lai & Bernstein, arXiv:2404.10844) : n'oublier que dans le sous-espace effectivement excité par le batch courant. Oubli à taux variable piloté par l'innovation (Bruce, Goel & Bernstein, arXiv:2003.02737). Oubli multiple, un $\lambda$ par groupe de paramètres dérivant à des vitesses différentes (Fraccaroli, Peruffo & Zorzi, arXiv:1503.07338) — **c'est exactement le problème que pose FisherAdapTune, mais avec une théorie de convergence**. Régularisation _fading_ avec convergence en temps fini **sans excitation persistante** (Lai, Panagou & Bernstein, arXiv:2501.04566), qui est la condition réaliste en deep learning.

**Expérience décisive et bon marché.** Mesurer $\kappa(\hat A_t)$ **sans damping** en fonction de $\beta$ et de la dimension effective des activations. Si le windup est confirmé, remplacer l'EMA scalaire par une mise à jour à oubli directionnel rend le damping largement superflu et **supprime un hyperparamètre**.

#### 2.D.6 Formes racine carrée : ne jamais former ni inverser `[ÉTABLI en filtrage — EXTRAPOLATION sur K-FAC]`

**Mécanisme.** Cinquante ans de pratique aérospatiale : ne jamais propager $P$ ni $P^{-1}$, mais un facteur $S$ tel que $P=SS^\top$, chaque mise à jour se faisant par QR ou rotations de Givens — la définie-positivité devient **structurelle** et le conditionnement effectif est divisé par deux ($\kappa(S)=\sqrt{\kappa(P)}$). Bierman (1977) ; Ramos, Brink, Ganesh & Hurtado (arXiv:2203.06105) ; Tracy, _A Square-Root Kalman Filter Using Only QR Decompositions_ (arXiv:2208.06452).

**Transposition.** K-FAC calcule $A^{-1/2}$ par décomposition spectrale — la voie la plus coûteuse et la plus mal conditionnée. La forme racine carrée dit : **maintenir directement $L_A$ tel que $A=L_AL_A^\top$**, et faire de l'EMA une mise à jour de Cholesky de rang $b$ (`cholupdate`), à $O(bn^2)$ au lieu de $O(n^3)$, sans jamais former $A$. Le préconditionnement s'applique ensuite par deux résolutions triangulaires. Ratio théorique pour $n=4096$, $b=32$ : ~$10^2$.

**Verdict matériel, avec la nuance qui décide.** `cholupdate` classique est une séquence de $n$ rotations de Givens dépendantes → **séquentiel, mauvais**. Deux contournements sains : (i) mise à jour de Cholesky **par blocs** ($b\in[64,256]$), qui exprime l'essentiel du travail en produits matriciels avec seulement $n/b$ étapes séquentielles ; (ii) mise à jour **par QR** (Tracy) : empiler $[\sqrt\beta L^\top;\sqrt{1-\beta},a^\top]$ et prendre le facteur $R$ — une QR « tall-skinny » est un noyau GPU mature. **Matmul-only atteignable à condition de traiter le mini-batch entier comme une mise à jour de rang $b$, pas $b$ mises à jour de rang 1.** C'est un choix de conception, pas une fatalité.

#### 2.D.7 Filtre de Kalman de rang faible et PEFT `[ÉTABLI]`

Chang, Durán-Martín, Shestopaloff, Jones & Murphy (arXiv:2305.19535) remplacent le bloc-diagonal du filtre de Kalman découplé par **rang faible + diagonale**, strictement plus expressif à mémoire comparable, à $O(Pr)$ mémoire et $O(Pr^2+r^3)$ par pas, **sans aucune décomposition spectrale**. Verdict matériel : ✅✅.

L'état de l'art de l'intersection Kalman × PEFT est **faible** : LoKO (Abdi, Sun, Zhang, Kaski & Pan, arXiv:2410.11551) retombe sur une covariance diagonale, donc perd exactement ce que K-FAC apporte. **Personne n'a construit un filtre de Kalman dont l'état de covariance est contraint à la variété de Kronecker ${A\otimes G}$**, avec une récursion de Riccati projetée et un bruit de processus par facteur ($Q_A$, $Q_G$) estimé par covariance matching (§2.E.3). C'est une piste vierge de forme claire.

---

### 2.E — Le damping et la significativité : quatre théories matures jamais transposées

_Domaines sources : théorie des matrices aléatoires, finance quantitative, traitement d'antennes, assimilation de données._

Le damping $\varepsilon$ de K-FAC/AdaFisher est réglé par grid search ou par une heuristique de Levenberg-Marquardt. Quatre corpus indépendants le dérivent.

#### 2.E.1 Shrinkage et estimateurs rotationnellement invariants `[ÉTABLI — EXTRAPOLATION sur K-FAC]`

Quand $n/N$ n'est pas petit, les valeurs propres échantillonnales sont **biaisées de manière déterministe et connue** (dispersion de Marchenko-Pastur : les grandes surestiment, les petites sous-estiment). Trois niveaux :

1. **Shrinkage linéaire** (Ledoit & Wolf, _J. Multivariate Anal._ 88(2), 2004) : $\alpha$ minimisant le risque quadratique attendu, **en forme close**, calculable en $O(n^2N)$ à partir de moments déjà disponibles. C'est un remplaçant direct et gratuit du $\varepsilon$ réglé à la main. Variante dominante sous gaussianité : OAS (Chen, Wiesel, Eldar & Hero, arXiv:0907.4698).
2. **Shrinkage non-linéaire** (Ledoit & Wolf, arXiv:1207.5322, _Ann. Statist._ 2012 ; version **analytique** en forme close, _Ann. Statist._ 48(5), 2020) : une correction valeur propre par valeur propre, base propre inchangée, $O(n^2)$ après diagonalisation.
3. **Estimateur rotationnellement invariant optimal** (Bun, Bouchaud & Potters, arXiv:1610.08104, _Physics Reports_, 165 p.) : la formule d'oracle — garder les vecteurs propres, remplacer les valeurs propres par leur estimation asymptotiquement optimale via la transformée de Stieltjes du spectre empirique.

**Deux raffinements qui changent la conclusion.**

- Donoho, Gavish & Johnstone (arXiv:1311.0851, _Ann. Statist._ 46(4), 2018) montrent qu'à **chaque fonction de perte** (Frobenius, opérateur, Stein, entropie, nombre de conditionnement) correspond un shrinker optimal **unique et admissible**. Le point crucial pour un préconditionneur : la perte pertinente n'est pas la perte de Frobenius sur $\hat F$, mais une perte sur $\hat F^{-1/2}$ — et le shrinker optimal pour cette perte est **différent** de $\varepsilon I$ _et_ du shrinkage linéaire. **Aucun optimiseur du second ordre publié n'utilise le bon shrinker.**
- La FIM d'un mini-batch a un rapport d'aspect $P/N\gg1$, régime dégénéré. Le shrinkage RMT y est mal posé sur $F$ — mais **parfaitement posé sur le noyau dual $N\times N$** (§3.3), où le rapport d'aspect est sain et où la diagonalisation coûte quelques millisecondes. C'est le pont non construit le plus prometteur de cette section `[EXTRAPOLATION]`.

**Précédent le plus proche.** Granziol & Baskerville (arXiv:2011.08181, _J. Phys. Complexity_ 2022) montrent que la constante de damping se décompose en une réduction du taux d'apprentissage et un **shrinkage linéaire** de la courbure estimée. Nino-Ruiz & Sandu (arXiv:1502.00301) et Popov, Sandu, Nino-Ruiz & Evensen (arXiv:2003.00354) substituent Ledoit-Wolf à la covariance échantillonnale dans un filtre d'ensemble **et améliorent la performance** — le précédent empirique le plus proche du transfert visé.

#### 2.E.2 Diagonal loading en beamforming robuste : la théorie min-max du damping `[ÉTABLI — EXTRAPOLATION]`

En formation de voies adaptative (MVDR/Capon), le poids optimal $w=R^{-1}s/(s^\top R^{-1}s)$ est catastrophiquement sensible à l'erreur d'estimation de $R$ ; on remplace $R$ par $R+\varepsilon I$. La théorie moderne établit que $\varepsilon$ **n'est pas un facteur d'ajustement mais le multiplicateur de Lagrange exact d'un problème d'optimisation au pire cas sur une boule d'incertitude** (Vorobyov, Gershman & Luo, IEEE TSP 51(2), 2003). Trois conséquences directes :

- **Un budget d'erreur, pas un grid search.** Borner l'incertitude sur l'estimation de $A$ (donnée par $b$ et $n$) et en déduire $\varepsilon$ par une équation scalaire. Li, Stoica & Wang (2003) le résolvent par une recherche 1-D monotone sur l'équation séculaire — **exactement la même structure qu'en §2.D.2**, donc mutualisable.
- **Une dépendance testable à la taille de batch.** Carlson (IEEE TAES 24(4), 1988) relie le niveau de charge à l'erreur d'estimation due au nombre fini de snapshots. Transposé : **le damping optimal doit décroître en $1/\sqrt b$ à $n$ fixé** (et croître en $\sqrt n$ à $b$ fixé) et non être constant. **Aucune implémentation K-FAC connue n'a cette dépendance** — c'est une prédiction falsifiable en une expérience.
- **Un damping anisotrope.** $\varepsilon I$ suppose une erreur isotrope. Tsai, Su, Tsao & Wang (arXiv:1602.02690) montrent que charger **uniquement le complément du sous-espace signal** est strictement meilleur. Transposé : charger uniquement la queue du spectre, en laissant les directions dominantes intactes — proche du shrinkage non-linéaire, et strictement différent du $\varepsilon I$ uniforme. Cet objet n'existe pas dans la littérature du second ordre.

Verdict matériel : ✅ parfait (bissection scalaire sur une somme de $n$ termes, réductible).

#### 2.E.3 Inflation adaptative et covariance matching `[ÉTABLI — EXTRAPOLATION]`

En assimilation de données, l'inflation de covariance compense la sous-dispersion due à la taille finie d'ensemble. Deux apports que le second ordre n'a pas :

- **Elle est estimée en ligne à partir des innovations** (Anderson, _Tellus A_ 2007 et 2009 : un filtre scalaire sur $\lambda$ alimenté par $|y-H\bar x|^2$ contre sa variance prédite). Transposé : **le facteur d'oubli $\beta$ de l'EMA ne devrait pas être un hyperparamètre fixe mais estimé à partir du désaccord entre le gradient observé et la variance prédite par $\hat F_{t-1}$.**
- **Elle est spatialement variable** — un $\lambda_i$ par variable d'état, donc un $\beta$ par couche voire par bloc, piloté par la dérive locale. C'est très exactement le terrain de FisherAdapTune, mais avec un estimateur fondé plutôt qu'un seuil.

Luo & Hoteit (arXiv:1108.0158) établissent en outre que l'inflation **est** le filtre $H_\infty$ (robuste au pire cas) : le damping n'est pas un bricolage numérique, c'est une garantie min-max — deuxième dérivation indépendante du même objet, en écho à §2.E.2.

Enfin, l'identification en ligne de $Q$ et $R$ par covariance matching (Mehra, IEEE TAC 1970 ; Lai & Bernstein, arXiv:2404.10914) fournit un mécanisme de mesure de la dérive : **mesurer la dérive de Fisher via les innovations plutôt que par différence explicite de matrices**. L'innovation est un vecteur de taille $b$ ; la statistique de matching est $O(b^2)$ — deux ordres de grandeur moins cher que la mesure $O(n^2)$ de FisherAdapTune, pour la même information.

#### 2.E.4 Significativité statistique : quelles directions le budget d'échantillons permet-il d'affirmer ? `[ÉTABLI en physique — EXTRAPOLATION]`

C'est le point le plus incisif de la section, et il vient du Monte-Carlo variationnel quantique. Schmitt & Heyl (_Phys. Rev. Lett._ 125, 100503, 2020 ; arXiv:1912.08828) régularisent la métrique de Fisher non par $\lambda I$ mais en **jetant les composantes dont l'estimateur Monte-Carlo n'est pas statistiquement significatif**. Dans la base propre de $S$, avec $\rho_k$ la projection de la force : $$\mathrm{SNR}(\rho_k)=\sqrt{\frac{N_{\text{MC}}}{1+(\sigma_k^2/\rho_k^2),\mathrm{Var}(H)}},$$ et l'erreur introduite en jetant la composante $k$ est **bornée explicitement** par $\Delta_k=\mathrm{SNR}(\rho_k)^2/N_{\text{MC}}$ — une borne que le damping de Tikhonov ne fournit pas. (Cette borne se simplifie en $1/(1+(\sigma_k^2/\rho_k^2)\mathrm{Var}(H))$ : elle est explicite mais **ne décroît pas** avec le budget d'échantillons ; c'est le seuil de coupure, et non la borne, qui se déplace quand $N_{\text{MC}}$ augmente.)

Le seuil de coupure n'est pas un hyperparamètre esthétique : c'est **une frontière de détectabilité statistique avec une borne d'erreur explicite**. La bonne question n'est pas « quel $\varepsilon$ ? » mais « quelles directions de courbure mon budget d'échantillons me permet-il d'affirmer ? ».

**Transposition à FisherAdapTune, sous forme de prédiction falsifiable.** FisherAdapTune gèle des paramètres selon une dérive de Fisher. Mais un drift mesuré est un drift _estimé_ : la question préalable est de savoir s'il est distinguable du bruit d'échantillonnage. Le critère SNR fournit exactement ce test. **Prédiction : une part significative des décisions de gel porte sur des directions dont le SNR est sous le seuil, c'est-à-dire sur du bruit.** C'est directement testable dans `fisher_drift_analysis`, et cela recoupe l'errata H3 du projet, qui identifie déjà que la condition de négligeabilité du reste dans l'équation de dérive **échoue précisément quand une couche se stabilise** — le régime visé par le critère.

Deux compléments du même registre :

- **Bornes de concentration matricielle** (Roosta-Khorasani & Mahoney, arXiv:1601.04737 / 1601.04738) : la taille d'échantillon $|S|$ garantissant $|\hat G-G|\le\varepsilon|G|$. Transposé : ne geler une couche que lorsque la dérive observée est _significativement_ inférieure au bruit d'estimation borné par la théorie.
- **Le biais d'inversion `[ÉTABLI]`, critique de fond.** Niu, Liao, Ling & Mahoney (arXiv:2502.13583) établissent que l'inverse d'une matrice sous-échantillonnée est un estimateur **structurellement biaisé** de l'inverse (inégalité de Jensen), et que ce biais **ne disparaît pas quand le batch grandit**. Or tout optimiseur de type Fisher inverse une Fisher estimée. Personne dans la littérature K-FAC/AdaFisher ne le quantifie ni ne le corrige. Zhang & Pilanci (arXiv:2402.01956) donnent l'estimateur à rétrécissement asymptotiquement non biaisé dans le cadre esquissé. **Mesurer et corriger ce biais sur une Fisher de réseau est une contribution théorique autonome, de nature entièrement différente de « l'approximation Kronecker est trop grossière », et plus difficile à réfuter.**

#### 2.E.5 Estimation stochastique de trace et de diagonale `[ÉTABLI]`

Hutchinson estime $\mathrm{tr}(F)$ en $O(\varepsilon^{-2})$ produits ; **Hutch++** (Meyer, Musco, Musco & Woodruff, arXiv:2010.09649) déflate d'abord le sous-espace dominant, calcule sa trace exactement, et n'applique Hutchinson qu'au résidu : $O(\varepsilon^{-1})$, **gain quadratique**. XTrace (Epperly, Tropp & Webber, arXiv:2301.07825, SIMAX 2024) et Nyström++ (Persson, Cortinovis & Kressner, arXiv:2109.10659) raffinent encore. La Fisher, avec son spectre à décroissance rapide, est **exactement le cas favorable**.

Toute méthode utilisant $\mathrm{tr}(F)$ — normalisation de pas, damping adaptatif, critères de gel — devrait utiliser Hutch++/XTrace plutôt que Hutchinson nu. **C'est un gain gratuit, et son absence de la littérature PEFT est une lacune exploitable immédiatement.** Verdict matériel : ✅✅ les $k$ sondes se batchent en un seul produit matriciel, une seule synchronisation.

Sur la diagonale, deux résultats à charge : Soen & Sun (arXiv:2402.05379) analysent la variance des estimateurs de Fisher diagonale et établissent (Remarque 4.4) que **l'estimateur par Hessien négatif a une variance nulle sur la dernière couche** — Rao-Blackwellisation implicite. Et Li, Dangel, Tam & Raffel, _Fishers for Free?_ (arXiv:2507.18807, ICML 2025 spotlight) montrent qu'**un accumulateur de gradients au carré d'Adam, gratuit, égale la vraie diagonale de Fisher sur cinq applications** (fusion de modèles, élagage, masquage, apprentissage continu). C'est **le témoin de contrôle à exiger de FisherAdapTune** : si une statistique gratuite suffit là où on paie une estimation de Fisher, le budget consacré au drift demande une justification empirique explicite.

---

### 2.F — La bloc-diagonalité comme _taper_ non calibré

_Domaine source : assimilation de données, statistiques en grande dimension._

**Mécanisme.** La _localisation_ multiplie élément par élément la covariance échantillonnale par une matrice de corrélation à support compact ; le produit de Schur de deux matrices semi-définies positives (PSD) étant PSD, on annule les corrélations lointaines (dominées par le bruit) tout en préservant la définie-positivité et en **augmentant le rang** de la covariance estimée. Gaspari & Cohn (_Q. J. R. Meteorol. Soc._ 125(554), 1999) ; Furrer & Bengtsson (_J. Multivariate Anal._ 98(2), 2007) pour le taper optimal ; Bickel & Levina (arXiv:0901.3079, _Ann. Statist._ 2008) pour la version par seuillage, consistante en norme opérateur si $(\log p)/N\to0$.

**La théorie dit exactement ce que la bloc-diagonalité imite maladroitement.** Quand une covariance $p\times p$ est estimée à partir de $N\ll p$ échantillons, les entrées hors-diagonale ont une erreur $O(N^{-1/2})$ constante : **les petites entrées sont entièrement du bruit**, et les annuler réduit l'erreur en norme opérateur. La bloc-diagonalité de K-FAC est donc un **taper binaire, fixé a priori, et non calibré sur le rapport $N/p$** (taille de mini-batch / dimension de couche). Un taper de Gaspari-Cohn est continu, à support réglable, et le rayon de localisation est _estimable_. Bickel-Levina donne le critère chiffré pour décider quelles entrées de $A$ K-FAC a le droit de jeter.

**L'objection honnête.** La localisation classique repose sur une distance géométrique ; un réseau n'en a pas. Ait-El-Fquih & Hoteit (arXiv:2603.03926) attaquent précisément cette objection en assimilation, ce qui rend le transfert scientifiquement intéressant plutôt que mécanique. La distance candidate pour un facteur de Kronecker est **fonctionnelle** (corrélation moyenne des activations, proximité dans le graphe d'attention, indice de tête), pas spatiale.

**Verdict matériel.** ✅ excellent pour le produit de Schur ($O(n^2)$, élément par élément) et pour la version par domaines locaux (résolutions indépendantes = `batched potrf`). Mais **la sparsification qui en résulte n'accélère pas les produits matriciels sur GPU sauf structure bloc régulière** : viser un taper à structure blocs-bandes, pas un seuillage non structuré.

**Complément — restaurer le rang gratuitement `[EXTRAPOLATION]`.** Le _spatial smoothing_ du traitement d'antennes (Shan, Wax & Kailath, IEEE TASSP 33(4), 1985) restaure le rang d'une covariance structurée en moyennant ses sous-blocs décalés, **sans échantillons supplémentaires**. Pour une couche d'attention multi-têtes ou une convolution, les sous-blocs de $A$ associés à des têtes ou positions différentes sont des « sous-réseaux » au sens exact du terme. Le moyennage restaurerait le rang à coût $O(n^2)$, là où K-FAC paie ce déficit avec du damping.

**Deux compléments d'estimation structurée qui portent une critique de fond `[ÉTABLI]`.**

- Roś, Bijma, de Munck & de Gunst (arXiv:1410.2118) établissent l'existence et l'unicité du **maximum de vraisemblance sous contrainte de Kronecker**, calculé par l'algorithme _flip-flop_. **K-FAC n'utilise pas ce MLE** : il utilise l'espérance factorisée $\mathbb{E}[aa^\top]\otimes\mathbb{E}[gg^\top]$, qui n'est le MLE que sous une hypothèse d'indépendance explicitement fausse. C'est une faiblesse chiffrable, et le flip-flop est bon marché (alternance de deux produits matriciels + deux Cholesky).
- Greenewald & Hero (arXiv:1402.5568) approximent par une **somme de $r$ produits de Kronecker plus une correction diagonale**, avec régularisation — c'est §2.B.1 avec quinze ans d'avance, dans un autre domaine.

---

### 2.G — Trois résultats négatifs argumentés

Il est aussi utile de savoir ce qui ne marchera pas.

1. **Robust PCA / décomposition rang faible + creux sur la Fisher.** L'intuition « quelques directions globales + un bruit creux » est séduisante, mais Tanner, Thompson & Vary (arXiv:1811.05919) établissent que la décomposition $L+S$ est **mal posée si la partie de rang faible est cohérente** (énergie localisée) — ce qui est précisément le cas des blocs diagonaux de la Fisher, dont l'énergie est localisée par couche. De plus chaque itération d'ADMM contient un seuillage de valeurs singulières, donc une SVD : ❌ éliminatoire pour un optimiseur en boucle. Ce qui reste valide : $L+\text{bloc-diagonal}$ avec $S$ remplacée par une structure connue — problème bien posé et non traité.
2. **Coloriage de graphe pour Hessiens creux.** La Fisher est **structurellement dense** : $F_{ij}=\mathbb{E}[\partial_i\ell,\partial_j\ell]$ est non nul pour presque tout $(i,j)$. Le coloriage (Coleman & Moré, _Math. Prog._ 28, 1984 ; Gebremedhin, Manne & Pothen, _SIAM Review_ 47(4), 2005) exige un motif de creux connu **et réellement creux**. La seule voie défendable serait de **postuler** un motif (bloc-diagonal + bande sur l'axe des couches) et d'en recouvrer exactement les entrées en $q$ produits Fisher-vecteur — donnant une estimation **sans biais sur le motif choisi**, contrairement à K-FAC qui est biaisé partout. Le coloriage se calcule une fois hors ligne, coût amorti à zéro.
3. **Rang de déplacement / matrices de type Toeplitz.** Suppose une invariance par translation, qui n'a de sens que pour des couches convolutives ou à biais positionnel relatif — et SINGD le fait déjà. Les solveurs superfast reposent sur la FFT et des récursions de Schur séquentielles : ❌ contre-exemple parfait d'« asymptotiquement rapide, pratiquement lent ».

---

### 2.H — Réponse synthétique à la question 1

**Résolvent-elles le problème de wall-time ?** Oui, mais une seule famille le résout _déjà et de façon mesurée_ : les itérations polynomiales matmul-only avec empilement par lots (§2.A), gain 25,9× sur le pas d'optimiseur, sans perte de qualité — parfois avec un gain de qualité. Les familles §2.C (rang faible randomisé) et §2.D.2–2.D.3 (mise à jour incrémentale ou continue de la décomposition) attaquent le même poste sous un angle différent et sont largement **cumulables** avec la première, puisqu'elles agissent sur des leviers indépendants : la première réduit le coût _par_ recalcul, les secondes réduisent le _nombre_ de recalculs ou le rang effectif.

**Sont-elles adaptées au matériel ?** La grille §1.3.3 tranche : oui pour les polynômes matriciels, le Nyström (surtout généralisé), la diagonale + rang faible par Woodbury, le Kronecker de rang $r$, le Cholesky creux par minimisation KL, le tapering, le shrinkage et l'estimation de trace ; conditionnellement pour l'équation séculaire (déflation hostile), les formes racine carrée (à faire par blocs), Monarch (noyaux de permutation fusionnés requis), les matrices hiérarchiques (seulement si plusieurs seconds membres) ; non pour HSS/HIF, butterfly, l'élimination creuse irrégulière, le rang de déplacement et Robust PCA.

**Ce qui améliore l'approximation sans rien coûter en temps** : le diagnostic $\sigma_2/\sigma_1$ du réarrangement (§2.B.1), le shrinkage en forme close à la place du damping (§2.E.1), Hutch++ à la place de Hutchinson (§2.E.5), et le test de significativité SNR avant toute décision de gel (§2.E.4).

---

## 3. Question 2 — Calculer et exploiter la Fisher exacte

### 3.1 Le fait central : sous lien canonique, la Fisher exacte ne demande aucun échantillonnage

**Condition exacte $F=G$ `[ÉTABLI]`.** Martens, _New insights and perspectives on the natural gradient method_ (arXiv:1412.1193, JMLR 21(146), 2020), §9 : $$F=\frac{1}{|\mathcal S_x|}\sum_x \mathbb{E}_{P_{y|x}}\big[H_{\mathcal L}(y,f(x,\theta))\big]=G.$$ **Condition suffisante précise** : $r(y\mid z)$ appartient à une famille exponentielle dont $z$ est le **paramètre naturel** (lien canonique). C'est le cas de softmax-entropie croisée sur les logits et de la gaussienne à variance fixe sur la moyenne — soit la quasi-totalité du deep learning supervisé.

**Ce que cela achète, et que la littérature applicative ignore.** $F$, objet statistique défini par une espérance sur $y\sim$ modèle, devient calculable **sans aucun échantillonnage**, par pure algèbre linéaire sur $J_n$ et un $\Lambda_n$ **analytique** :

|Modèle de sortie|$\Lambda_n$|rang|
|---|---|---|
|Softmax + entropie croisée, $z$ = logits|$\mathrm{diag}(p_n)-p_np_n^\top$|$C-1$|
|Gaussienne $\mathcal N(z,\sigma^2 I)$ (MSE)|$\sigma^{-2}I_C$|$C$|
|Bernoulli / sigmoïde|$p_n(1-p_n)$|1|
|Poisson, lien log|$\exp(z_n)$|1|

Le rang $C-1$ du cas softmax est immédiat ($\Lambda_n\mathbf1=p_n-p_n=0$) et **c'est le fait qui gouverne toute la section 3.3**.

**Ce que la condition ne couvre pas `[à surveiller]`** : lien non canonique, sortie post-softmax plutôt que logits, pertes qui ne sont pas des log-vraisemblances (hinge, contrastif), mélanges, VAE. Là $F\ne G$ et il faut choisir explicitement lequel on veut. Aucune des pistes de ce rapport ne lève cette limite de portée — c'est aussi la limite de validité d'AdaFisher.

**Bilan des substitutions.**

|Substitution|Gain|Perte|Référence|
|---|---|---|---|
|$H\to G$|PSD garanti, invariance, pas de dérivée seconde du réseau|terme de courbure du réseau ; exact seulement en régime « lazy »/résidus faibles|Martens §8|
|$G\to F$|**aucun coût** (identité) sous lien canonique|**rien — substitution gratuite et exacte**|Martens §9|
|$F\to\hat F_K$|$K$ backward au lieu de $C-1$|variance $O(1/K)$, rang $\le BK$|—|
|$F\to\tilde F$|1 backward, réutilise le gradient|**change d'objet mathématique**|arXiv:1905.12558 ; arXiv:2406.06420|

**La seule substitution réellement gratuite est $G\leftrightarrow F$, et c'est précisément celle que la littérature applicative n'exploite pas.** Elle transforme un problème d'échantillonnage en un problème d'algèbre linéaire exacte.

**Ce que coûte la Fisher empirique, précisément.** Kunstner, Balles & Hennig (arXiv:1905.12558, NeurIPS 2019) établissent que $\tilde F$ ne capture pas d'information de second ordre en général : l'identité $F=\mathbb{E}[-\nabla^2\log r]$ repose sur $\mathbb{E}_{y\sim\text{modèle}}[\nabla_\theta\log r]=0$, et en substituant $y\sim$ données cette espérance vaut $-\nabla_\theta\mathcal L$, non nulle hors optimum. Wu, Yu, Zhang & Woodland (arXiv:2406.06420, NeurIPS 2024) donnent le mécanisme **précis** de l'échec, plus exploitable que le contre-exemple d'impossibilité : la _projection à échelle inversée_. La réduction de perte induite par la mise à jour empirique sur l'échantillon $n$ vaut $$(\kappa_n)_{\rm EF}=-\frac{\eta}{|\nabla_\theta l_n|_2},$$ soit une amplification **inversement proportionnelle à la norme du gradient** : la réduction de perte induite, $\kappa_n|\nabla_\theta l_n|=-\eta$, est alors **identique pour tous les échantillons**, quel que soit leur degré de convergence : $\tilde F$ est structurellement biaisée vers les échantillons **déjà bien appris**, et la norme de la mise à jour diverge à mesure que l'entraînement progresse.

**Conséquence directe pour FisherAdapTune, non publiée `[EXTRAPOLATION]`.** FisherAdapTune utilise la Fisher empirique pour **mesurer une dérive** et décider d'un gel. Le biais de projection à échelle inversée y est directement délétère : il surpondère exactement les échantillons déjà convergés, donc **sous-estime la dérive là où elle compte**. Le critère de gel hérite d'un biais que personne n'a mesuré.

### 3.2 Les produits exacts coûtent trois passes, et ce n'est jamais le problème

**Opérateur $\mathcal R$ de Pearlmutter** (_Neural Computation_ 6(1):147–160, 1994) : $Hv$ exact en $O(1)$ passes (≈ 2× un gradient), sans former $H$ — différentiation en mode direct appliquée à la fonction gradient (_forward-over-reverse_).

**Produits GGN de Schraudolph** (_Neural Computation_ 14(7):1723–1738, 2002) : $Gv=J^\top(\Lambda(Jv))$ en **1 JVP + 1 application de $\Lambda$ analytique + 1 VJP**. Le résultat est **exact**, pas approché — et comme $\Lambda$ est en forme close (§3.1), **aucun échantillonnage de label n'intervient**.

**Le fait central de cette section : un produit Fisher-vecteur exact coûte 2–3 passes, quel que soit $C$. La difficulté n'est jamais le produit matrice-vecteur, c'est la résolution du système linéaire.**

**Outillage disponible, vérifié.**

|Bibliothèque|Référence|Ce qu'elle fournit|
|---|---|---|
|BackPACK|Dangel, Kunstner & Hennig, arXiv:1912.10985, ICLR 2020|gradients par échantillon, variance, blocs diagonaux exacts de la GGN, KFAC/KFRA|
|ASDL|Osawa, Ishikawa, Yokota, Li & Hoefler, arXiv:2305.04684|interface unifiée de préconditionnement : K-FAC, Shampoo, SMW-NG, Fisher exacte/MC/empirique|
|**curvlinops**|Dangel, Eschenhagen, Ormaniec, Fernandez, Tatzel & Kristiadi, arXiv:2501.19183|Hessien, GGN, Fisher, KFAC comme `LinearOperator` SciPy, branchables sur CG/Lanczos/LOBPCG|
|laplace-torch|Daxberger et al., arXiv:2106.14806, NeurIPS 2021|GGN/Fisher dernière couche, KFAC, diagonale, complète|

`curvlinops` est le bon point d'ancrage expérimental : il expose la distinction $F$ / $\tilde F$ / $\hat F_K$ / $G$ / $H$ derrière une interface matvec commune, ce qui permet de **comparer les cinq objets à code identique**.

**Gradients par échantillon.** Pour une couche linéaire, le gradient par échantillon est le produit extérieur $\delta_na_n^\top$ (Goodfellow, arXiv:1510.01799) ; `vmap(grad(...))` en JAX / `torch.func`, ou BackPACK `BatchGrad`. **Le mur est mémoire, pas calcul** : $O(BP)$ si l'on matérialise — 12,8 Go pour $B=32$, $P=10^8$ en fp32, rédhibitoire en pré-entraînement, **anodin en LoRA**.

### 3.3 La formulation duale : le levier décisif, et il vient de la physique

**Notation.** On note $\mathbf J\in\mathbb{R}^{m\times P}$ l'empilement vertical des $\Lambda_n^{1/2}J_n$ sur les $B$ échantillons du batch, chaque bloc étant tronqué au rang effectif $C-1$ de $\Lambda_n$ ; on a alors exactement $F=\frac1B\mathbf J^\top\mathbf J$ avec $m=B(C-1)\ll P$.

**Mécanisme.** Woodbury donne alors $$(F+\lambda I)^{-1}g=\lambda^{-1}\Big[g-\mathbf J^\top\big(\lambda B I_m+\mathbf J\mathbf J^\top\big)^{-1}\mathbf Jg\Big].$$ **Inversion exacte au prix d'une factorisation $m\times m$** — pas d'approximation, pas de CG, pas de Kronecker. Le coût passe de $O(P^3)$ à $O(m^3+m^2P)$.

#### 3.3.1 La physique l'a poussée à 10⁶ paramètres pendant que le ML empilait des Kronecker

La _reconfiguration stochastique_ (Sorella, _Phys. Rev. B_ 64, 024512, 2001 ; arXiv:cond-mat/0009149) est **littéralement** la descente de gradient naturel, publiée indépendamment d'Amari la même année : le préconditionneur $S_{ij}=\langle O_i^_O_j\rangle-\langle O_i^_\rangle\langle O_j\rangle$
avec $O_i=\partial_{\theta_i}\ln\psi_\theta$ **est** la Fisher empirique de la distribution de Born (la partie réelle de la métrique de Fubini–Study, aussi appelée tenseur géométrique quantique ; la partie imaginaire, forme symplectique de Berry, n'a pas d'analogue ML et est jetée).

**MinSR** — Chen & Heyl, _Empowering deep neural quantum states through efficient optimization_, **Nature Physics 20, 1476–1481 (2024)**, arXiv:2302.01941 `[ÉTABLI]`. La solution de norme minimale s'écrit entièrement dans l'espace des échantillons : $\delta\theta=O^\top(OO^\top)^{-1}\varepsilon$. Complexités données dans le papier :

- SR standard : $O(N_p^2N_s+N_p^3)$
- **MinSR : $O(N_pN_s^2+N_s^3)$**

Dès que $N_p\gg N_s$, **le coût redevient linéaire en $N_p$**. Échelle atteinte : 146 320 paramètres dans la v1, **jusqu'à 10⁶ paramètres** annoncés dans Nature Physics, convergence à précision machine sur des modèles de spin frustrés. Matériel : A100 80 Go, supercalculateur JUWELS — **explicitement cité**. L'opération dominante $OO^\top$ est un produit matriciel dense, le cas d'usage idéal du tensor core.

**Le contraste avec la voie Kronecker est frontal.** K-FAC/AdaFisher acceptent une **erreur d'approximation structurelle non contrôlée** pour éviter $O(N_p^3)$ ; MinSR obtient **le même gain de coût sans aucune approximation**, en payant seulement la contrainte de rang $N_s$ — laquelle est de toute façon inévitable, puisque la Fisher estimée sur $N_s$ échantillons **est** de rang $\le N_s$. _La factorisation de Kronecker invente du rang qui n'a pas été mesuré._

**Confirmation indépendante et simultanée.** Rende, Viteritti, Bardone, Becca & Goldt, _A simple linear algebra identity to optimize Large-Scale Neural Network Quantum States_, **Communications Physics 7, 260 (2024)**, arXiv:2310.05715 : même identité de Woodbury, découverte par un autre groupe le même trimestre, poussée à ≈ 268 000 paramètres avec un Vision Transformer profond, là où le plafond antérieur du SR était « quelques milliers ». Deux découvertes simultanées de la même identité dans deux groupes de physique, alors que la communauté ML continuait d'empiler des variantes de Kronecker, est en soi un signal.

**SPRING** — Goldshlager, Abrahamsen & Lin, _J. Comput. Phys._ 516, 113351 (2024), arXiv:2401.10190 `[ÉTABLI]`. Chaque pas de MinSR résout un sous-système différent du système idéal ; SPRING conserve le résidu précédent et le projette sur le nouveau sous-espace d'échantillons (Kaczmarz randomisé avec momentum), **accumulant l'information de courbure à travers les mini-lots au lieu de la jeter**. Résultat frontal : _sur l'atome d'oxygène, SPRING atteint la précision chimique après quarante mille itérations, là où MinSR **et K-FAC** échouent tous deux même après cent mille_, et ×2,5 d'accélération à taux d'apprentissage optimalement réglés.

**C'est une comparaison directe, publiée, K-FAC contre formulation duale, sur un vrai problème, où la duale gagne.** Et le mécanisme importe : la non-stationnarité de la courbure y est traitée non par une EMA sur des facteurs de Kronecker (qui mélange des géométries de mini-lots différents dans un objet structurellement contraint) mais par un **momentum sur le résidu dans l'espace dual**, dont le sens géométrique est défini quel que soit le mouvement de la base propre.

**PRIME-SR** (Wang & Liu, arXiv:2604.18357, avril 2026) : le paramètre de momentum de SPRING est fragile — convergence prouvée pour $0\le\mu<1$, **contre-exemples de divergence à $\mu=1$**. PRIME-SR le fixe **automatiquement** à partir de la _dimension spectrale effective_ et du **recouvrement de sous-espaces entre itérations consécutives**. C'est exactement ce qui manque à AdaFisher : une règle dérivée pour l'hyperparamètre de mémoire de la courbure, indexée sur une mesure de non-stationnarité mesurable en ligne. Et le recouvrement de sous-espaces est par ailleurs **une mesure de dérive de Fisher meilleure que celle de FisherAdapTune** : invariante par reparamétrisation, insensible à l'échelle des valeurs propres.

**Contre-exemple utile `[ÉTABLI, 2026]`.** Attar, Aboussalah & Hibat-Allah (arXiv:2608.18065) rapportent MinSR instable sur les réseaux récurrents autorégressifs, stabilisable par régularisation simple. La formulation duale n'est pas magique : elle interagit avec l'architecture. Un décodeur causal a une géométrie de gradients mal conditionnée et un régime $N_s$ petit rend le noyau presque singulier — directement pertinent pour le PEFT sur décodeurs.

#### 3.3.2 Le même levier, redécouvert côté PINNs et côté ML

|Méthode|Référence vérifiée|Mécanisme|Résultat|
|---|---|---|---|
|**D-NGD**|Jnini & Vella, arXiv:2505.21404|pas de Gauss-Newton résolu dans l'**espace des résidus** au lieu de l'espace des paramètres|NGD jusqu'à **12,8 M paramètres sur un seul GPU** ; erreur $L^2$ 1 à 3 ordres de grandeur meilleure|
|**ENGD + Woodbury**|Guzmán-Cordero, Dangel, Goldshlager & Zeinhofer, arXiv:2505.12149|Woodbury + SPRING + randomisation|**jusqu'à ×75** plus rapide à erreur $L^2$ égale|
|**TENGraD**|Soori, Can, Mu, Gürbüzbalaban & Mehri Dehnavi, arXiv:2106.03947|inverse **exact** de chaque bloc par Woodbury, factorisation réutilisée|~1 passe + algèbre $B\times B$|
|**SENG**|Yang, Xu, Wen, Chen & Xu, arXiv:2006.05924|esquisse du sous-espace de rang faible + version distribuée|ResNet-50/ImageNet, 75,9 % en 41 époques|
|**EGN**|Korbit, Adeoye, Bemporad & Zanon, arXiv:2405.14402|identité de Duncan-Guttman : factorise une matrice **de la taille du mini-batch**|1 fwd + $C$ bwd + $O(m^3)$|
|**Woodbury de Benzing**|Benzing, arXiv:2201.12250, ICML 2022|$(\lambda I+F)^{-1}u$ **sans former** la matrice des gradients|1 passe + $O(B^3)$|
|**Analyse sketch-and-project**|Goldshlager, Hu & Lin, arXiv:2508.21022|théorie : le gradient naturel sous-échantillonné **est** du sketch-and-project ; convergence globale à mini-batch de taille **quelconque** ; SPRING = version accélérée|attribue l'avantage à l'exploitation de la **décroissance spectrale de la jacobienne**|

Ces communautés ont résolu le problème dual **à l'échelle exacte où vit LoRA** ($P\sim10^6$–$10^7$), et le transfert n'a pas été fait. Les jacobiennes LoRA sont notoirement à spectre décroissant — le régime où la théorie de Goldshlager et al. prédit le gain maximal. La brique manquante est un préconditionneur Nyström sur le Gram $\mathbf J\mathbf J^\top$ quand $m$ dépasse quelques milliers.

**Verdict matériel.** ✅✅ $\mathbf J\mathbf J^\top$ est un produit matriciel, suivi d'un Cholesky $m\times m$. Aucune décomposition sur la grande dimension.

### 3.4 Le régime PEFT/LoRA : où la Fisher exacte devient calculable, avec les chiffres

C'est la section opérationnelle du rapport.

#### 3.4.1 Nombre de paramètres LoRA

$P_{\rm LoRA}=\sum r(d_{\rm in}+d_{\rm out})$ :

|Modèle|Configuration|$P_{\rm LoRA}$|
|---|---|---|
|RoBERTa-base ($d=768$, 12 couches)|$r=8$, q&v|≈ **0,29 M**|
|T5-base|$r=8$, q&v|≈ **0,6 M**|
|ViT-B/16|$r=8$, q&v|≈ **0,29 M**|
|LLaMA-2-7B ($d=4096$, 32 couches)|$r=8$, q&v|≈ **4,2 M**|
|LLaMA-3-8B|$r=16$, q&v|≈ **6,8 M** (GQA : `v_proj` est $4096\to1024$)|
|LLaMA-2-7B|$r=16$, q,k,v,o|≈ **16,8 M**|

Facteur $10^3$–$10^4$ par rapport au modèle complet. Les deux dernières lignes tiennent compte de l'attention à requêtes groupées (GQA) de LLaMA-3, qui réduit la dimension de sortie de `v_proj`.

#### 3.4.2 Rang de la GGN et taille du système dual — la ligne de fracture

$\mathrm{rang}(F)\le\min(P_{\rm LoRA},m)$ avec $m=B(C-1)$ en classification, $m=B\cdot T\cdot(V-1)$ en modélisation de langue.

|Tâche|$B$|$C$ ou $T{\times}V$|$m$|Système dual|
|---|---|---|---|---|
|GLUE binaire (SST-2, MRPC)|32|$C=2$|**32**|trivial (1024 entrées)|
|MNLI|32|$C=3$|**64**|trivial|
|CIFAR-100 (ViT+LoRA)|128|$C=100$|**12 672**|lourd mais faisable|
|ImageNet-1k|64|$C=1000$|**63 936**|limite haute|
|LM causal (LLaMA)|8|$T{=}512$, $V{=}128,000$|≈ $5\cdot10^8$|**impossible**|

**C'est la ligne de fracture.** En classification à peu de classes, $F$ est une matrice de rang $\le32$ dans un espace de dimension $3\cdot10^5$ : elle est **exactement représentable et exactement inversible**. En modélisation de langue, le facteur $T\cdot V$ rend l'approche duale caduque et l'échantillonnage redevient obligatoire (voir §3.6 pour la seule voie qui subsiste).

#### 3.4.3 Mémoire du facteur $\mathbf J\in\mathbb{R}^{m\times P_{\rm LoRA}}$ en fp32

|Cas|$m$|$P_{\rm LoRA}$|Mémoire|
|---|---|---|---|
|RoBERTa-base + LoRA $r{=}8$, GLUE binaire, $B{=}32$|32|0,29 M|**37 Mo**|
|T5-base + LoRA $r{=}8$, MNLI, $B{=}32$|64|0,6 M|**154 Mo**|
|LLaMA-3-8B + LoRA $r{=}16$, tâche binaire, $B{=}32$|32|6,8 M|**0,87 Go**|
|LLaMA-2-7B + LoRA $r{=}16$ q,k,v,o, $C{=}2$, $B{=}32$|32|16,8 M|**2,15 Go**|
|ViT-B + LoRA $r{=}8$, CIFAR-100, $B{=}128$|12 672|0,29 M|**14,7 Go** — le mur|

**Conclusion opérationnelle.** Sur toute la suite GLUE en PEFT, **la Fisher vraie, exacte, non approximée tient en quelques centaines de mégaoctets et s'inverse exactement**. Il n'y a aucune raison technique d'y utiliser la Fisher empirique, ni Kronecker, ni EMA. Coût en passes : $C-1$ rétropropagations pour $\mathbf J$ complète — donc **une seule en classification binaire**, exactement le prix de la Fisher empirique.

**Ce que cela permet immédiatement `[EXTRAPOLATION sur le résultat, ÉTABLI sur la faisabilité]`.** Personne n'a publié la comparaison systématique $F$ exacte / $\hat F_K$ / $\tilde F$ / iEF / K-FAC **avec $F$ réellement calculée** sur une suite PEFT complète. La métrique existe (§3.5), les cinq opérateurs sont exposés derrière une interface unique par `curvlinops`, le budget est de quelques centaines de mégaoctets. **Livrable : une carte quantitative de l'erreur d'approximation par tâche, par couche, par étape d'entraînement** — précisément ce qui manque pour critiquer AdaFisher et FisherAdapTune sur pièces.

#### 3.4.4 Régimes où la Fisher exacte est calculable par structure

- **Architectures réversibles** : Buffelli, McGowan, Xu, Cioba, Shiu, Hennequin & Bernacchia (arXiv:2411.07979) donnent les mises à jour Gauss-Newton **exactes** en forme close pour une classe assez expressive pour les benchmarks usuels.
- **Réseaux linéaires profonds** : Bernacchia, Lengyel & Hennequin, NeurIPS 2018 — gradient naturel exact.
- **K-FAC exact** : Eschenhagen et al. (arXiv:2311.00636) — les variantes _expand_ et _reduce_ sont exactes pour les réseaux linéaires profonds à partage de poids dans leur régime respectif. **Hors de là, K-FAC est une approximation non contrôlée.**
- **Dernière couche** : GGN exacte à coût marginal (laplace-torch) ; Galashov, Da Costa, Xu, Hennig & Gretton (arXiv:2510.04606) donnent même une solution en forme close sous perte quadratique.

### 3.5 iEF : corriger l'empirique pour un coût nul

**Mécanisme `[ÉTABLI]`.** Wu, Yu, Zhang & Woodland (arXiv:2406.06420, NeurIPS 2024), éq. 8 : $$\Delta\theta_{\rm iEF}=-\eta,\nabla_\theta\boldsymbol l^\top\big(\nabla_\theta\boldsymbol l,\nabla_\theta\boldsymbol l^\top+\lambda I\big)^{-1}\boldsymbol s_{\rm iEF},\qquad \boldsymbol s_{\rm iEF}=\big[|\nabla_{z_1}l_1|_2^2,\dots,|\nabla_{z_N}l_N|_2^2\big]^\top.$$ **On ne change pas la matrice** : on remplace le membre de droite $\mathbf 1$ (implicite dans la Fisher empirique) par les normes au carré des gradients **au niveau des logits**. La réduction de perte par échantillon devient $\approx-\eta|\nabla_{z_n}l_n|_2^2$, donc **proportionnelle au degré de non-convergence de l'échantillon** — exactement l'inverse du biais de §3.1.

**Coût : négligeable.** $\boldsymbol s_{\rm iEF}$ ne demande que les gradients au niveau des logits, obtenus par `retain_grad()` — **aucune passe supplémentaire**, contre $K$ rétropropagations pour la Fisher échantillonnée. Complexité de l'optimiseur exact : temps $O(M^2P)$, mémoire $O(MP)$ ; c'est cette mémoire qui restreint la méthode aux modèles à petit nombre de paramètres entraînables, et qui explique que les auteurs se placent en PEFT.

**Résultat le plus contre-intuitif du papier.** Sous leur cadre d'évaluation, **iEF approxime mieux la vraie mise à jour de gradient naturel que la Fisher échantillonnée** — qui est pourtant non biaisée et plus coûteuse. Explication : avec $K=1$ et $B$ modeste, la variance de $\hat F_K$ est telle que son rang effectif est très inférieur à celui de $F$. iEF est en outre remarquablement robuste au choix du damping, préservant sa qualité même avec un $\lambda$ proche de zéro.

**Validation directement dans le terrain visé** : T5-base avec LoRA et Prompt-Tuning sur GLUE, ViT avec LoRA sur CIFAR-100 ; appliqué comme optimiseur, l'iEF exact atteint les meilleures performances de test et la perte d'entraînement la plus basse sur la majorité des tâches, face à des baselines AdamW/Adafactor bien réglées.

**La métrique, utile en soi.** Le papier introduit $$\gamma(\Delta\theta)=\frac{(\Delta\theta^\top F\Delta\theta)^{1/2}}{|\Delta\theta^\top\nabla_\theta\mathcal L(\theta)|}$$ — plus bas = meilleure approximation de la mise à jour NGD exacte. **Elle ne coûte qu'un seul produit Fisher-vecteur**, donc elle est mesurable à grande échelle. C'est l'outil de diagnostic qui manquait, et il est directement branchable sur `fisher_drift_analysis` pour classer K-FAC, EKFAC, AdaFisher, iEF et l'empirique brute sur une échelle commune.

**Deux extensions non publiées `[EXTRAPOLATION]`.**

1. _Généraliser le facteur d'échelle._ iEF est un cas particulier d'une question ouverte : quel $\boldsymbol s$ rend $\mathbf J^\top(\mathbf J\mathbf J^\top+\lambda I)^{-1}\boldsymbol s$ le plus proche de la mise à jour NGD exacte ? En PEFT, $F$ exacte est disponible (§3.4) : on peut **résoudre ce problème de moindres carrés pour le $\boldsymbol s^\star$ optimal**, l'observer, et vérifier si la norme au carré des gradients de logits en est une bonne approximation, ou si une forme meilleure et tout aussi gratuite existe (par exemple pondérée par $\Lambda_n$).
2. _Appliquer iEF à un critère de gel plutôt qu'à une mise à jour._ Le biais de projection à échelle inversée est délétère pour FisherAdapTune (§3.1). Appliquer $\boldsymbol s_{\rm iEF}$ au critère de dérive est **gratuit et non publié**.

### 3.6 Réduire la variance de l'estimateur Monte-Carlo : un vide de littérature

**Constat négatif, à valeur de résultat.** Le seul travail dédié et vérifié sur la variance des estimateurs de Fisher en deep learning est Soen & Sun (arXiv:2402.05379), limité à la diagonale. **Aucun travail n'applique quasi-Monte Carlo, variables de contrôle, échantillonnage antithétique ou Rao-Blackwellisation systématique à l'estimateur de Fisher échantillonnée, et aucun ne mesure le $K$ nécessaire.** Le corpus classique existe hors deep learning (Spall, _J. Comput. Graph. Statist._ 14(4), 2005 ; Buchholz, Wenzel & Mandt, arXiv:1807.01604 pour le quasi-Monte Carlo en inférence variationnelle) mais n'a pas été transposé — alors que K-FAC et ses dérivés reposent tous sur un estimateur à $K=1$ non analysé.

**La piste la plus prometteuse, et la seule qui rende la Fisher quasi-exacte abordable en modélisation de langue `[EXTRAPOLATION]`.** L'espérance sur $y$ est une **somme finie de $C$ termes pondérés par $p_{n,c}$**. Rien n'oblige à choisir entre tout sommer ($C-1$ backward) et échantillonner ($K$ backward). **Partition exacte** : sommer analytiquement sur le top-$k$ des classes (masse $\pi_n=\sum_{c\in\text{top-}k}p_{n,c}$, souvent > 0,95 après quelques époques) et estimer par Monte-Carlo le résidu de masse $1-\pi_n$. L'estimateur reste **non biaisé** et sa variance est **multipliée** par $\approx(1-\pi_n)$, donc divisée par $\approx 1/(1-\pi_n)\gtrsim20$, pour un coût de $k+1$ backward avec $k\sim2$–5.

C'est exactement le régime des LLM, où $V\sim10^5$ interdit la somme complète mais où la distribution softmax est extrêmement piquée. La Remarque 4.4 de Soen & Sun (variance nulle en dernière couche quand l'espérance est analytique) en est le cas limite $k=C$.

Trois autres leviers directement transposables : échantillonnage antithétique sur les labels via inversion de la CDF catégorielle sur un couplage commun ; **variable de contrôle $\tilde F$** — la Fisher empirique est corrélée à $\hat F_K$ et **gratuite**, donc $\hat F_K-\beta(\tilde F-\mathbb E[\tilde F])$ avec $\beta$ estimé en ligne ; quasi-Monte Carlo (Sobol) sur les tirages de labels.

### 3.7 Trois contre-résultats à intégrer, sous peine de construire sur du sable

Le postulat « la Fisher empirique est un mauvais substitut » est établi. La conclusion « donc il faut la vraie Fisher » **ne l'est pas**. Trois travaux ont effectivement calculé les mises à jour exactes, et leurs résultats sont défavorables :

1. **Benzing (arXiv:2201.12250, ICML 2022)** calcule les mises à jour du second ordre **exactes** (Fisher vraie via Woodbury) sur MLP/MNIST, WideResNet18/CIFAR10, VGG11/SVHN, en comparant explicitement Fisher MC à 1 échantillon et Fisher complète sur toutes les combinaisons label-entrée. Résultat : **K-FAC surpasse significativement les mises à jour du second ordre vraies.** L'avantage de K-FAC ne viendrait donc pas de la courbure mais d'une descente de gradient « sur les neurones » (FOOF), ~1,5× plus rapide que K-FAC.
2. **Buffelli et al. (arXiv:2411.07979)** : la Gauss-Newton **exacte** dans des architectures réversibles **généralise mal** — saturation rapide en mini-batch, surajustement de chaque mini-batch, régime lazy, NTK quasi constant, pas de changement de représentation.
3. **Liu & Ghezelbash (arXiv:2607.26247, 2026)** : une famille à deux paramètres interpolant gradient brut et gradient naturel pour l'initialisation LoRA ; **le meilleur point est strictement à l'intérieur du continuum, pas aux extrémités**.

**Contrepoint favorable** : Abreu, Vyas, Kakade & Morwani (arXiv:2510.09378, 2025), transformeurs jusqu'à 150 M, GN complète, **5,4× moins d'itérations** que SOAP et Muon — mais **sans aucune mesure de temps mural**. C'est une borne supérieure de complexité d'itération, pas un optimiseur. Leur méthode ne fait d'ailleurs aucune inversion : elle linéarise le modèle, développe la perte au second ordre sur ce modèle linéarisé et minimise par un optimiseur **interne** (Muon) via des JVP avec recherche linéaire.

**Lecture conjointe.** La courbure exacte accélère nettement **l'optimisation** (Abreu) mais peut dégrader la **généralisation** (Buffelli) et perdre face à une approximation grossière à budget égal (Benzing). L'écart entre les trois se joue sur le régime (lazy vs feature learning), le damping, et le budget par pas — c'est exactement l'espace de conception à exploiter, et **toute architecture nouvelle issue de ce projet devra se positionner explicitement contre ces trois résultats**.

Un élément de réponse vient de la physique : Peng & Chan (arXiv:2502.19576) établissent qu'**en optimisation stochastique**, contrairement au cas déterministe, ce n'est pas la proximité du minimum qui détermine l'utilité d'une méthode du second ordre, mais **l'expressivité de l'ansatz** — les méthodes quasi-second-ordre gagnent quand le modèle est assez expressif pour représenter la cible, même loin de l'optimum. Transposé au PEFT : _le bénéfice d'un préconditionneur de courbure dépendrait du rang de l'adaptateur, pas du stade du fine-tuning._ Hypothèse testable et, à ma connaissance, non testée côté ML `[EXTRAPOLATION]`.

Enfin, il faut noter que le groupe qui a le plus sérieusement tenté de sauver K-FAC en physique en a conclu qu'il valait mieux en changer : Drissi, Keeble, Rozalén Sarmiento & Rios (arXiv:2401.17550, _Communications Physics_ 2024) corrigent l'échelle et la direction du pas de K-FAC en VMC (gains substantiels à coût négligeable), puis reformulent le problème en théorie des jeux et proposent un optimiseur qui **surclasse systématiquement toutes leurs améliorations de K-FAC** en stabilité, précision et vitesse.

### 3.8 Réponse synthétique à la question 2

**Comment calculer la Fisher réelle ?** Sous lien canonique — donc dans la quasi-totalité du deep learning supervisé — elle **est** la GGN, et son facteur de sortie $\Lambda$ est analytique : il n'y a rien à échantillonner. On l'obtient soit comme opérateur matvec exact en 2–3 passes (Pearlmutter/Schraudolph), soit explicitement comme $\mathbf J^\top\Lambda\mathbf J$ en $C-1$ rétropropagations.

**Comment l'exploiter à coût réel ?** Par la formulation duale : le rang de $F$ est borné par $B(C-1)$, donc Woodbury résout **exactement** en $O(m^3+m^2P)$. Cette voie est démontrée à 10⁶ paramètres en physique quantique, 12,8 M en PINNs sur un seul GPU, et n'a pas été transférée au PEFT.

**Dans quel régime ?** En classification à peu de classes avec adaptateurs LoRA : 37 Mo à 2,2 Go, une seule rétropropagation en binaire. En modélisation de langue causale, la dimension duale explose et seule une somme partielle sur le top-$k$ des classes (§3.6) reste viable.

**À quel prix quand ce n'est pas possible ?** iEF corrige le biais principal de l'empirique pour un coût strictement nul, et fait mieux que la Fisher échantillonnée en fidélité mesurée.

**Sous quelle réserve ?** Trois travaux publiés montrent que la courbure exacte n'est pas automatiquement meilleure. Le rapport ne recommande donc pas « la Fisher exacte comme optimiseur », mais **la Fisher exacte comme instrument de mesure** en PEFT — vérité terrain contre laquelle auditer les approximations — et la formulation duale comme voie d'optimisation là où elle est calculable.

---

## 4. Question 3 — Classement comparatif

### 4.1 Définition des trois critères

- **Pertinence scientifique (1–5)** : la technique attaque-t-elle une faiblesse _réelle et documentée_ de K-FAC/EKFAC/AdaFisher/FisherAdapTune, et le fait-elle d'une manière qui produit un résultat publiable ? 5 = attaque frontalement une faille structurelle et ouvre une contribution originale ; 1 = marginal ou déjà couvert.
- **Faisabilité d'implémentation (1–5)** : combien de travail entre l'état actuel du dépôt `fisher_drift_analysis` et un résultat mesurable ? 5 = quelques centaines de lignes, briques disponibles ; 1 = programme de thèse, primitives à écrire.
- **Gain estimé (1–5)** : effet attendu sur au moins un des trois axes wall-time / mémoire / qualité d'approximation. 5 = gain d'un ordre de grandeur mesuré ailleurs ; 1 = gain marginal ou incertain.

Une colonne **statut** rappelle si le gain est mesuré (`É` = établi, chiffre publié) ou anticipé (`X` = extrapolation).

### 4.2 Tableau de classement

|#|Technique|Domaine d'origine|Pert.|Fais.|Gain|Statut|Justification courte|
|---|---|---|---|---|---|---|---|
|1|**Racine inverse matmul-only (Newton couplé / Newton–Denman–Beavers) + empilement 3D par lots**|Théorie des fonctions de matrices ; HPC|5|5|5|É|Attaque le seul poste réellement dominant du wall-time. **25,9× mesuré** sur le pas d'optimiseur, et NDB **bat la décomposition exacte en perplexité** (11,68 vs 11,80). Zéro passe supplémentaire, produits matriciels purs. Réserve : converge en FP16, diverge en BF16.|
|2|**Formulation duale exacte par Woodbury sur le Gram $\mathbf J\mathbf J^\top$**|Monte-Carlo variationnel quantique ; PINNs|5|4|5|É|Inversion **exacte**, aucune approximation, coût $O(m^3+m^2P)$ avec $m=B(C-1)$. Démontrée à 10⁶ paramètres (MinSR) et 12,8 M sur un GPU (D-NGD). En PEFT/GLUE, $m\le64$ : trivial. Ligne de fracture nette sur les LLM causals.|
|3|**iEF — rééchelonnement par échantillon de la Fisher empirique**|ML (NeurIPS 2024)|5|5|3|É|Corrige le biais de « projection à échelle inversée » pour **coût strictement nul** ; bat l'empirique _et_ la Fisher échantillonnée en fidélité mesurée ; validé sur T5+LoRA/GLUE et ViT+LoRA/CIFAR-100. Gain en qualité, pas en temps. Apporte de surcroît la métrique $\gamma$.|
|4|**Diagnostic $\sigma_2/\sigma_1$ du réarrangement de Van Loan–Pitsianis**|Algèbre matricielle structurée (1992)|5|5|3|X|Mesure **scalaire, calculable, non heuristique** du biais d'indépendance de K-FAC, couche par couche et dans le temps. Coût : une SVD randomisée sur un produit implicite, jamais la matrice. C'est le test de falsification le moins cher du postulat central d'AdaFisher.|
|5|**Critère de fraîcheur adaptatif par facteur (Purifying Shampoo / FOAM)**|ML systèmes|4|5|4|É|Décider _quand_ recalculer plutôt que fixer un intervalle. Critère **moins cher que la décomposition qu'il évite** ; −20 % de wall-clock et 5/5 seeds convergentes contre 3/5. Structure identique à FisherAdapTune, mais avec une borne calculable.|
|6|**Kronecker de rang $r>1$ par SVD randomisée du réarrangement**|Algèbre matricielle structurée|5|3|4|X|Généralisation rigoureuse et exacte de K-FAC ; $r$ interpole entre K-FAC ($r=1$) et le bloc exact ($r=N$). Produits matriciels par lots exclusivement. **Piège réel** : pas d'inverse en forme close pour $r>1$ → CG préconditionné par le terme rang 1, 3–5 itérations.|
|7|**Diagonale + rang faible estimées conjointement (SKETCHLORD) + Woodbury**|RandNLA|4|4|4|É|Réponse structurelle directe au biais de Kronecker : rien d'imposé sur la diagonale, budget de rang sur les corrélations dominantes y compris inter-couches. Estimation **jointe** prouvée strictement supérieure à séquentielle. Mémoire $O(Pm)$ : viable en LoRA, pas en pré-entraînement.|
|8|**Nyström randomisé, et surtout Nyström _généralisé_ matmul-only**|RandNLA|4|5|4|É/X|Préserve la PSD ; rang $r\approx10$ suffisant en pratique (SketchySGD) ; RS-KFAC mesure déjà 3× sur K-FAC. La variante généralisée supprime la QR sur la grande dimension — la seule chose qui empêchait le gain annoncé — et n'a jamais été essayée sur la courbure.|
|9|**Shrinkage RMT (Ledoit–Wolf non linéaire / RIE) appliqué au noyau dual**|Théorie des matrices aléatoires ; finance|5|4|3|X|Remplace le damping par une correction **en forme close et prouvée optimale**, supprimant un hyperparamètre. Le rapport d'aspect $P/N$ rend l'application directe mal posée — mais le noyau dual $N\times N$ est le bon objet, petit et diagonalisable à chaque pas. Le shrinker doit correspondre à une perte sur $F^{-1/2}$, pas sur $F$.|
|10|**Noyaux SYRK tensor-core pour la construction des facteurs**|HPC|4|3|3|X|$A_\ell=a^\top a$ **est** un SYRK. cuBLAS n'exploite pas les tensor cores ; deux équipes ont écrit leurs noyaux pour Muon sans jamais les réutiliser pour K-FAC. Gain attendu 1,3–2× sur un poste que KAISA identifie comme **invariant au sharding**, donc autrement incompressible.|
|11|**Mise à jour exacte de la décomposition par équation séculaire (Bunch–Nielsen–Sorensen, Gu–Eisenstat, Brand)**|Algèbre linéaire numérique|4|3|4|É/X|L'EMA de K-FAC **est** un scaling + une modification de rang 1 : valeurs propres exactes à $O(n^2)$ par pas et base amortie en $O(n^3)$ périodique, au lieu de tout recalculer. Sous-produit gratuit : bornes analytiques d'entrelacement de Cauchy sur la dérive spectrale — un critère de gel en $O(n)$ pour FisherAdapTune. Réserve : la déflation est hostile au SIMT.|
|12|**Test de significativité SNR avant décision de gel (Schmitt–Heyl)**|Physique (Monte-Carlo variationnel)|5|4|2|X|Transforme « quel seuil de dérive ? » en « quelles directions mon budget d'échantillons me permet-il d'affirmer ? », avec **borne d'erreur explicite**. Prédiction falsifiable : une part significative des gels de FisherAdapTune porte sur du bruit. Recoupe l'errata H3 du projet.|
|13|**Suivi dynamique de la variété de rang faible (DLR / DyKAF étendu au rang $r$)**|Analyse numérique des EDO sur variétés|4|3|4|É/X|Supprime la base périmée : $O(nr^2)$ par pas contre $O(n^3)$ tous les $T$ pas, sans dérive de la base. DyKAF le fait déjà, mais au rang de Kronecker 1 ; la combinaison avec l'entrée #6 n'existe pas.|
|14|**CUDA graphs + megabatching des collectives**|HPC|3|4|3|É|Temps CPU **0,1 ms constant** sous capture, quel que soit le nombre de petits noyaux — exactement le régime des implémentations K-FAC académiques qui bouclent en Python par couche. Plancher NCCL de 25 µs : toute collective de préconditionneur doit être coalescée.|
|15|**Diagonal loading au pire cas (MVDR) / inflation adaptative**|Traitement d'antennes ; assimilation de données|4|4|2|X|Trois dérivations indépendantes du damping comme multiplicateur de Lagrange min-max, avec une **prédiction testable** : $\varepsilon^\star\sim O(\sqrt{n/b})$, dépendance qu'aucune implémentation K-FAC n'a. Variante forte : damping **anisotrope** (charger la queue du spectre seulement) — objet inexistant en second ordre.|
|16|**Cholesky creux de $F^{-1}$ par minimisation KL, motif dicté par l'architecture**|Processus gaussiens ; EDP elliptiques|4|2|4|X|Seule méthode qui produit **directement le facteur de $F^{-1}$** — ce que l'optimiseur veut, jamais $F$ — contournant le problème d'inversion de #6. Formule close colonne par colonne, colonnes indépendantes ⇒ `batched Cholesky`, le motif GPU idéal. Coût d'entrée : le choix du motif et de l'ordre.|
|17|**Hutch++ / XTrace à la place de Hutchinson**|RandNLA|3|5|2|É|$O(\varepsilon^{-1})$ produits au lieu de $O(\varepsilon^{-2})$, gain quadratique, dans le régime spectral exact de la Fisher. Gain gratuit pour toute normalisation de pas ou damping adaptatif. Son absence de la littérature PEFT est une lacune immédiate.|
|18|**Frequent Directions / esquisse à fenêtre glissante avec garantie**|Algorithmique des flux|4|3|3|É/X|L'EMA est un oubli **sans aucune garantie d'approximation** ; la fenêtre glissante donne du $(1\pm\varepsilon)$ pour $O(\log W)$ esquisses batchables. Permet de mesurer la dérive **vraie** (deux fenêtres disjointes) et de tester si le signal de gel de FisherAdapTune survit à l'élimination de l'artefact d'EMA.|
|19|**Représentation compacte L-SR1 comme correcteur additif du terme Kronecker**|Optimisation numérique classique|4|3|3|X|$\hat G=(A\otimes G)+\Psi M\Psi^\top$ avec $y_i=Gs_i-(A\otimes G)s_i$ : la correction apprend **exactement le résidu que Kronecker n'explique pas**, inversion exacte par Woodbury, aucune décomposition supplémentaire. $\lVert\Psi M\Psi^\top\rVert/\lVert A\otimes G\rVert$ est une mesure par couche de l'erreur de Kronecker. 0 à 1 GGNVP par pas.|
|20|**RPCholesky piloté par le second moment d'Adam**|RandNLA (matrices noyau)|4|4|3|X|La distribution de pivotage optimale **est** la statistique qu'Adam maintient déjà gratuitement ; une colonne de la Fisher est un seul R-op. Donne un facteur PSD sans hypothèse de Kronecker, donc capable de corrélations inter-couches. Risque : la queue lourde du spectre pourrait exiger un $k$ élevé.|
|21|**Recyclage de sous-espaces de Krylov + angles principaux comme mesure de dérive**|Algèbre linéaire numérique (EDP, problèmes inverses)|4|2|3|X|Une EMA lisse les _valeurs_, un recyclage transporte les _directions_ ; Davis-Kahan borne la rotation du sous-espace par $\lVert\Delta G\rVert/\text{gap}$, donc un grand écart spectral (que la Fisher a) implique un sous-espace quasi-stationnaire. Rend la dérive **directionnelle** : distingue changement d'échelle et changement de géométrie. 30–80 % d'itérations en moins en NLA.|
|22|**Précision : quantification sélective de la base propre + hiérarchie mantisse-first**|HPC / quantification|3|3|3|É/X|4-bit Shampoo : quantifier **les vecteurs propres** et non le préconditionneur, avec rectification d'orthogonalité — états ÷7, surcoût wall-clock nul à +4,8 %. S'y ajoute la leçon transversale de précision établie en §2.A.1 (la mantisse, pas la dynamique).|
|23|**Monarch / BTT comme classe d'approximation de courbure**|Apprentissage de matrices structurées|3|3|3|X|**Projection analytique optimale** (Théorème 1 de Dao et al.), $A\otimes G$ en est le cas dégénéré, et l'inverse d'un produit de bloc-diagonales reste un produit de bloc-diagonales — on récupère l'inversion close que le rang $r$ perd. Réserve mesurée : jusqu'à 50 % du temps en permutations sans noyau fusionné.|
|24|**Oubli directionnel (SIFt-RLS) à la place de l'EMA scalaire**|Identification de systèmes|4|3|2|X|Pathologie établie depuis 40 ans : l'oubli exponentiel fait diverger la covariance dans les directions non excitées — et les activations sont fortement anisotropes. Le damping masque le symptôme. Expérience décisive et bon marché : mesurer $\kappa(\hat A_t)$ sans damping en fonction de $\beta$.|
|25|**Sketching de Kronecker (Woodruff) : le pas naturel comme régression Kronecker ridge**|RandNLA théorique|4|2|3|X|$(A\otimes G+\lambda I)^{-1}\mathrm{vec}(\nabla W)$ **est** une régression Kronecker régularisée, résoluble en temps sous-quadratique sans diagonaliser, les leverage scores d'un produit de Kronecker se factorisant. Si le pas naturel se calcule sans décomposition, le motif d'AdaFisher disparaît. Cloison totale entre les deux littératures.|
|26|**Formes racine carrée : `cholupdate` par blocs au lieu de $A^{-1/2}$ spectral**|Filtrage de Kalman (aérospatial)|3|3|3|X|Définie-positivité **structurelle** et conditionnement effectif divisé par deux ; $O(bn^2)$ au lieu de $O(n^3)$. Viable sur GPU **à condition** de traiter le mini-batch comme une mise à jour de rang $b$ (blocs ou QR tall-skinny), jamais $b$ mises à jour de rang 1.|
|27|**Recouvrement intra-GPU du recalcul (stream séparé, launch dépendant)**|HPC|3|2|4|X|Les deux charges se disputent des ressources **différentes** : le forward-backward est compute-bound tensor-core, la décomposition est memory/latency-bound. Seul levier qui **ne dégrade pas la qualité**, contrairement à l'amortissement temporel. Aucun travail publié.|
|28|**Somme partielle top-$k$ sur les classes (Rao-Blackwellisation partielle)**|Monte-Carlo / statistique|4|4|3|X|Ni tout sommer ($C-1$ backward) ni échantillonner ($K$ backward) : sommer analytiquement le top-$k$ (masse > 0,95) et estimer le résidu. Estimateur **non biaisé**, variance × $(1-\pi_n)$, soit ÷ 20 environ, coût $k+1$ backward. **La seule voie qui rende la Fisher quasi-exacte abordable en modélisation de langue.**|
|29|**Préconditionnement polynomial (supprimer les synchronisations de CG)**|Algèbre linéaire numérique|3|3|3|X|Le coût de Hessian-free sur GPU vient des $2m$ réductions globales, pas des FLOPs. Un polynôme de degré $d$ remplace $d$ itérations et leurs $2d$ all-reduce par $d$ produits **sans une seule synchronisation**, et ce **sans approximer davantage la courbure**. Réserve sérieuse : stabilité des polynômes de haut degré en bf16.|
|30|**Tapering / seuillage de covariance (Gaspari–Cohn, Bickel–Levina), spatial smoothing**|Assimilation de données ; traitement d'antennes|3|4|2|X|La bloc-diagonalité est un _taper_ binaire non calibré sur $N/p$ ; la théorie donne le rayon optimal et le seuil chiffré. Le spatial smoothing restaure le rang en moyennant les sous-blocs par tête/position, **sans échantillons supplémentaires**. Réserve : la sparsification n'accélère pas sur GPU sans structure blocs-bandes.|
|31|**Zolotarev rationnel pour la racine inverse**|Algèbre linéaire numérique|3|3|3|X|2 itérations avec taux 17 contre 5–8 itérations pour les polynomiaux ; les $r$ branches sont indépendantes donc batchables. Toute la littérature Muon/Shampoo explore les polynômes, personne les rationnelles. Réserve honnête : repose sur QR/Cholesky (« acceptable ») et non sur du GEMM (« excellent ») — à mesurer avant de croire.|
|32|**Optimisation en norme spectrale (programme semi-défini alterné) comme étalon hors ligne**|Optimisation matricielle|4|2|2|É|Résultat théorique dur : l'écart entre optimum de Frobenius et optimum spectral **croît en $\sqrt{(m-1)(n-1)}$**. Or la convergence d'un pas préconditionné dépend d'une erreur _spectrale_. K-FAC, EKFAC et Shampoo optimisent tous le mauvais critère. Outil d'analyse seulement : ❌ sur GPU.|
|33|**Filtre de Kalman contraint à la variété de Kronecker**|Filtrage bayésien|4|2|3|X|Récursion de Riccati **projetée sur ${A\otimes G}$**, avec bruit de processus par facteur estimé par covariance matching : la version principielle de ce que l'EMA fait à la main. LoKO retombe sur du diagonal, DyKAF n'a pas de modèle de bruit. Objet inexistant.|
|34|**Tensor-Train sur l'axe des couches**|Physique à $N$ corps (DMRG)|2|2|2|X|Le rang TT joue le rôle d'entropie d'intrication inter-couches. Mais le résultat d'Abreu et al. suggère que l'information inter-couches rapporte peu, et les cœurs TT sont petits (faible occupation) avec une chaîne séquentielle en profondeur.|
|35|**HODLR / $\mathcal H^2$ sur l'axe des couches**|Problèmes inverses à grande échelle|3|2|2|É/X|Précédent réel sur la GGN d'un MLP (GOFMM), construction matrix-free à **256 esquisses constantes**, 2 TFlop/s mesurés sur V100. **Objection dirimante** : 2,3 Tflop/s à 64 vecteurs concurrents mais **≈150 Gflop/s en mono-vecteur** — un optimiseur applique $F^{-1}$ à un seul gradient par pas.|
|36|**EnKF / Ensemble Kalman Inversion (courbure sans gradient)**|Assimilation de données|2|3|1|É/X|Obtient la direction Gauss-Newton par $N$ passes avant, sans rétropropagation. Mais le sous-espace d'itération reste **piégé dans l'enveloppe linéaire de l'ensemble initial** : dimension $10^2$ pour $P\sim10^9$. Acceptable en LoRA de rang faible, rédhibitoire ailleurs.|
|37|**HSS / HIF / butterfly / rang de déplacement / élimination creuse exacte**|Calcul scientifique|2|1|1|É|Asymptotiquement rapides, pratiquement lents : bases emboîtées et squelettisation séquentielles, $O(\log N)$ facteurs enchaînés dont jusqu'à 50 % du temps en permutations, supernœuds hétérogènes. **Le contre-exemple canonique d'« asymptotique trompeuse sur GPU ».**|
|38|**Robust PCA (rang faible + creux) sur la Fisher**|Vision / optimisation convexe|1|1|1|É|**Mal posé** : la décomposition $L+S$ n'est identifiable que si la partie de rang faible est incohérente, or l'énergie de la Fisher est localisée par couche (rigidité matricielle). De plus chaque itération d'ADMM contient une SVD. À écarter — sauf sous la forme $L+\text{bloc-diagonal}$, qui est bien posée et non traitée.|

### 4.3 Lecture du classement

**Trois régimes se dégagent nettement.**

_Les entrées 1 à 5_ sont des gains à faible risque : chacune est soit déjà mesurée ailleurs, soit implémentable en quelques centaines de lignes, et aucune ne demande d'invention théorique. Elles devraient être faites avant toute autre chose, et elles se composent : #1 réduit le coût _par_ recalcul, #5 réduit le _nombre_ de recalculs, #2 et #3 changent l'objet estimé, #4 mesure ce que les autres coûtent en fidélité.

_Les entrées 6 à 33_ sont l'espace de contribution réel : chacune attaque une faille structurelle avec un outil mature dans un autre domaine, et aucune n'a été appliquée à la Fisher. Le risque y est modéré et le rendement scientifique élevé.

_Les entrées 34 à 38_ sont classées basses **pour des raisons matérielles, pas théoriques**. C'est le point qu'il faut retenir de la §1.3 : plusieurs de ces techniques ont une complexité asymptotique meilleure que tout ce qui les précède dans le tableau, et perdent quand même — parce que l'écart de débit entre un produit matriciel dense et une récursion irrégulière est de trois ordres de grandeur aux tailles rencontrées.

**Une remarque de méthode sur les scores.** Les colonnes « gain » marquées `X` sont des estimations argumentées, pas des mesures ; elles ne doivent pas être lues comme comparables aux colonnes `É`. La colonne « Statut » qualifie le **mécanisme** (`É` = publié et vérifié), pas nécessairement le gain chiffré. Les entrées dont le **gain lui-même** a été mesuré sur un optimiseur de courbure sont les n° 1, 5, 8 (partiellement, via RS-KFAC), 14, 17, 22 et 35 ; les n° 2 et 3 ont un mécanisme et des résultats publiés, mais dans un autre domaine (physique/PINNs) ou sur un autre axe (fidélité, pas temps). Une seule entrée a été mesurée sur un optimiseur de courbure complet en conditions de pré-entraînement de LLM : la n° 1.

---

## 5. Synthèse et recommandations

### 5.1 Le fil conducteur

Les trois questions convergent sur une même observation. Les faiblesses d'AdaFisher/FisherAdapTune ne sont pas indépendantes les unes des autres, et elles ne sont pas non plus des « approximations un peu grossières » : ce sont **quatre décisions non dérivées** qui se renforcent mutuellement.

1. **Substituer l'empirique à la vraie Fisher** — décision d'implémentation, jamais justifiée théoriquement, et corrigeable gratuitement (iEF) ou supprimable entièrement (dualité en PEFT).
2. **Tronquer au rang 1 de Kronecker, et pas même optimalement** — décision de tractabilité, mesurable par $\sigma_2/\sigma_1$, généralisable au rang $r$.
3. **Lisser par une EMA scalaire euclidienne** — heuristique sans garantie, qui n'est équivalente à aucune récursion de Riccati, et dont la pathologie (windup directionnel) est documentée depuis quarante ans ailleurs.
4. **Amortir par un damping fixe et un recalcul périodique** — deux hyperparamètres qui ont chacun une théorie mature dans un autre domaine (shrinkage RMT, diagonal loading min-max, inflation adaptative ; critère de fraîcheur borné).

Et une cinquième décision, celle-là purement matérielle : **payer une décomposition spectrale** là où une itération polynomiale fait mieux 25 fois plus vite.

Le point important est que **corriger la cinquième est indépendant de corriger les quatre premières**, et de loin le moins risqué. C'est ce qui structure le plan ci-dessous.

### 5.2 Plan en trois vagues pour `fisher_drift_analysis`

**Vague 1 — instrumentation et gains gratuits (quelques jours ; aucun risque scientifique).**

- Implémenter la **Fisher exacte de référence en PEFT** (§3.4) sur deux tâches GLUE binaires + MNLI avec RoBERTa/T5 + LoRA rang 8. Budget : 37–154 Mo, une seule rétropropagation en binaire. Exposer $F$, $\tilde F$, $\hat F_K$, iEF et K-FAC derrière `curvlinops`.
- Mesurer $\gamma$ (§3.5) et l'erreur relative en normes de Frobenius **et spectrale** de chaque approximation contre $F$, par couche et par étape d'entraînement. C'est la carte quantitative qui manque à toute la littérature.
- Calculer le spectre singulier de $\mathcal R(F_\ell)$ (§2.B.1) sur les mêmes configurations. Livrable : la courbe $\sigma_2/\sigma_1$ par couche au fil de l'entraînement. **Si ce ratio est petit partout, la critique d'indépendance d'AdaFisher est faible et il faut réorienter le projet vers le wall-time ; s'il est grand, l'entrée #6 du classement devient la contribution principale.** C'est le point de décision du projet.
- Ajouter iEF comme option de l'estimateur (coût nul) et **appliquer $\boldsymbol s_{\rm iEF}$ au critère de dérive de FisherAdapTune** — extension non publiée, gratuite.
- Remplacer Hutchinson par Hutch++ partout où une trace intervient.

**Vague 2 — wall-time (une à deux semaines ; risque faible, gain mesuré ailleurs).**

- Remplacer `torch.linalg.eigh` par une itération de Newton couplée en FP16 (pas BF16) et par Newton–Denman–Beavers, avec normalisation par itération de puissance plutôt que par norme de Frobenius. Empiler les blocs de toutes les couches en tenseurs 3D traités par `bmm`. Capturer le pas entier en graphe CUDA.
- Ajouter le critère de fraîcheur par facteur (§2.A.4 / entrée #5) et mesurer, sur la même configuration, l'effet du couplage : critère adaptatif **× solveur matmul-only × empilement 3D**. Ces trois leviers sont orthogonaux et personne ne les a combinés — c'est en soi un résultat publiable de nature systèmes.
- Substituer un Nyström **généralisé** (entrée #8, la meilleure faisabilité du bloc 6–13) à la décomposition des facteurs : deux esquisses, aucune QR sur la grande dimension, tout en produits matriciels. C'est la variante qui débloque le stockage en demi-précision, et elle n'a jamais été essayée sur la courbure.
- Mesurer la dépendance $\varepsilon^\star(b,n)$ (entrée #15) : si elle ne suit pas $O(\sqrt{n/b})$, c'est un résultat ; si elle la suit, c'est un hyperparamètre supprimé.

**Vague 3 — contribution scientifique (le reste du stage/de la thèse ; risque réel).** Selon le résultat du point de décision de la vague 1, l'une des deux voies :

- _Si $\sigma_2/\sigma_1$ est significatif_ : Kronecker de rang $r$ — avec, comme variante de repli à moindre risque, la diagonale + rang faible estimée conjointement (entrée #7, mécanisme établi et mémoire viable en LoRA) — par SVD randomisée du réarrangement, inversion par CG préconditionné par le terme rang 1, comparaison rigoureuse à K-FOC, DyKAF et Tensor Normal Training sur la distance spectrale à $F$ exacte. Extension naturelle : suivi **dynamique** du rang $r$ (entrée #13), qui n'existe pas.
- _Si $\sigma_2/\sigma_1$ est négligeable_ : la faiblesse est ailleurs, et les deux candidats sérieux sont (i) l'EMA — mesurer le windup directionnel (entrée #24) puis remplacer par oubli directionnel ou fenêtre glissante à garantie (entrée #18), et (ii) le damping — shrinkage RMT sur le noyau dual (entrée #9).

Dans les deux cas, deux volets transversaux : le **test de significativité SNR avant décision de gel** (entrée #12), qui est la critique la plus directe de FisherAdapTune et recoupe l'errata H3 déjà établi ; et le **biais d'inversion** (§2.E.4), qui est une critique de nature différente — structurelle, non réductible par l'augmentation du batch, et jamais quantifiée sur une Fisher de réseau.

### 5.3 Ce contre quoi toute contribution devra se positionner

Les trois résultats de la §3.7 (Benzing, Buffelli et al., Liu & Ghezelbash) contredisent la prémisse implicite « plus la courbure est exacte, mieux c'est », et le seul contrepoint favorable ne fournit aucune mesure de temps mural.

La conséquence pour ce projet est précise : **ne pas viser « une meilleure approximation de la Fisher » comme objectif final**, mais soit un gain de wall-time à qualité constante (vague 2, où le terrain est net et les gains mesurables), soit une _mesure_ de ce que les approximations coûtent réellement (vague 1, où la vérité terrain est calculable en PEFT et où personne ne l'a établie). Une nouvelle architecture qui améliorerait la fidélité sans améliorer ni le temps ni la généralisation aurait raté sa cible au sens des instructions du projet.

Un dernier élément : l'hypothèse de Peng & Chan discutée en §3.7 — le bénéfice d'un préconditionneur de courbure dépendrait du **rang de l'adaptateur LoRA**, pas du stade du fine-tuning — est une expérience d'une journée qui, si elle se confirme, réorganise tout le champ des priorités.

---

## 6. Limites et réserves

### 6.1 Limites de portée scientifique

- **L'équivalence $F=$ GGN, donc la validité même d'AdaFisher, n'est exacte que sous lien canonique** (softmax-entropie croisée sur les logits, erreur quadratique gaussienne). Hors de ce cadre — lien non canonique, sortie post-softmax, perte non log-vraisemblance, mélanges, VAE — $F\ne G\ne H$, et seuls l'échantillonnage (§3.6) ou le sous-échantillonnage du GGN restent valables comme approximations de la Fisher vraie. **Aucune des pistes de ce rapport ne lève cette limite.**
- **Les qualificatifs « jamais appliqué » / « piste vierge » reflètent l'état de sept recherches indépendantes menées en parallèle**, croisées entre elles. Une revue systématique exhaustive pourrait révéler des travaux de niche non détectés, en particulier dans les actes d'ateliers et la littérature non anglophone.
- **Le résultat d'Abreu et al.** (un préconditionneur GGN par couche égale presque la GGN complète) est le seul à mesurer directement la valeur de l'information inter-couches. S'il se confirme, il déclasse les entrées 16, 21, 34 et 35 du classement — c'est-à-dire toute la famille « corriger la bloc-diagonalité ». Il repose sur une seule étude, à 150 M paramètres, sans mesure de temps mural. **C'est la plus grande source d'incertitude du classement.**
- **Les trois contre-résultats de la §3.7** (Benzing, Buffelli, Liu & Ghezelbash) contredisent la prémisse « plus exact = mieux » et n'ont pas de réfutation publiée.

### 6.2 Réserves de vérification bibliographique

**Vérifié.** Les 200 identifiants arXiv listés en §7.2 ont été résolus sur la source primaire (titre, auteurs, date de soumission v1 confirmés). Aucun ne s'est révélé inexistant ou incohérent.

**Corrections apportées à des identifications approximatives.**

- **FisherAdapTune** ne porte pas ce titre sur arXiv. La méthode est introduite dans **arXiv:2606.10196**, _Fisher-Guided Progressive Parameter Selection for Adaptive Fine-Tuning_, Ghodsiyeh Rostami, Po-Han Chen, Mahdi S. Hosseini, v1 du 2026-06-08, cs.CV/cs.AI, **aucune venue annoncée** (préprint, version unique). Le nom n'apparaît que dans le corps du résumé ; code sur github.com/AtlasAnalyticsLab/FisherAdapTune.
- **K-FOC** (Schnaus, Lee & Triebel, _Kronecker-Factored Optimal Curvature_) **n'a pas d'identifiant arXiv** : atelier Bayesian Deep Learning de NeurIPS 2021 (14 décembre 2021), dépôt institutionnel DLR elib 145806.
- **Tensor Normal Training** (Ren & Goldfarb) : **arXiv:2106.02925**, v1 2021-06-05, NeurIPS 2021 Spotlight.
- **KL-Shampoo** n'est pas un titre de papier : la méthode est introduite dans **arXiv:2509.03378**, _Understanding and Improving Shampoo and SOAP via Kullback-Leibler Minimization_ (Lin, Lowe, Dangel, Eschenhagen, Xu, Grosse), version étendue du papier ICLR 2026. Travaux dérivés réels : **arXiv:2605.06316** (Pro-KLShampoo) et **arXiv:2605.26327** (reparamétrage de Shampoo/SOAP pour le stockage bfloat16).
- **« Online KL Shampoo » : aucun papier arXiv de ce nom.** Les recherches plein texte ne retournent que des travaux d'apprentissage par renforcement KL-régularisé sans rapport. **Cette référence, si elle apparaît ailleurs, est non substantiable en l'état et ne doit pas être citée.**

**Non vérifié sur source primaire — à revérifier avant citation dans un livrable formel.**

- Gower & Richtárik, arXiv:1612.06013 (_Sketch and Project: … and Inverting Matrices_) : entrée référencée par des bases secondaires, non confirmée sur arxiv.org.
- Pennington & Worah, _Nonlinear random matrix theory for deep learning_, NeurIPS 2017.
- Grasedyck, _Hierarchical Singular Value Decomposition of Tensors_, SIMAX 2010 (DOI à confirmer).
- Sindhwani, Sainath & Kumar, _Structured Transforms for Small-Footprint Deep Learning_, NIPS 2015.
- Famille SEEK/SEIK (Pham, Verron, Roubaud) en océanographie : citée pour mémoire conceptuelle uniquement.
- Préconditionneur de Kronecker pour réseaux d'automates stochastiques (Pitsianis/Stewart, NCSU).
- Pagination exacte non revérifiée (titre, auteurs et revue confirmés) : Gu & Eisenstat (SIMAX), Li–Stoica–Wang (IEEE TSP 2003), Mehra (IEEE TAC 1970), Singhal & Wu (NIPS 1988), Puskorius & Feldkamp (IEEE TNN 1994), Bekas–Kokiopoulou–Saad (_Appl. Numer. Math._ 2007), Byrd–Nocedal–Schnabel (_Math. Prog._ 1994), Coleman & Moré (_Math. Prog._ 1984).
- Listes d'auteurs non vérifiées individuellement (existence et titre confirmés) : arXiv:2008.01440, arXiv:2208.07095, arXiv:2601.01575, arXiv:2608.22129.
- Références de journal sans préprint arXiv, vérifiées via l'éditeur uniquement : Ailon & Chazelle (SICOMP 2009), Datar–Gionis–Indyk–Motwani (SICOMP 2002), Braverman & Ostrovsky (FOCS 2007), Drineas–Kannan–Mahoney (SICOMP 2006).

### 6.3 Réserves méthodologiques sur les chiffres

- **Les mesures de wall-time citées ne sont pas comparables entre elles sans leur contexte.** Distributed Shampoo annonce ≤ 10 % de surcoût à fréquence de recalcul 100–1000 ; DASH mesure +77 % à fréquence 1 ; Dion3 mesure 26× sur le pas d'optimiseur isolé ; Moonlight annonce 1–3 %. Ces quatre chiffres sont mutuellement cohérents : ils mesurent des choses différentes. **Toute affirmation de surcoût sans (fréquence, taille de batch, stratégie de parallélisme) est vide.** L'illustration la plus nette est l'annexe E de Dion3 : 1,9 % contre 17 % pour la _même_ implémentation, selon le régime.
- **Les mesures de débit `syevd` par taille les plus détaillées disponibles datent de CUDA 11.0 sur une Titan XP** (architecture Pascal, sans tensor cores). L'ordre de grandeur est confirmé indépendamment sur A100 (1,3–1,5 % du pic, arXiv:2511.16174), ce qui suggère que la situation n'a pas changé qualitativement — mais **un benchmark H100/B200 récent manque dans la littérature**. C'est une lacune identifiée, et une mesure d'une heure à faire soi-même.
- **Les évaluations d'adéquation GPU du tableau §1.3.3 et du classement §4.2** sont, pour les primitives non mesurées directement (scatter-add creux, déflation de l'équation séculaire), des extrapolations raisonnées à partir de la littérature sur des opérations analogues, et non des mesures.
- **Les ratios de coût asymptotique des méthodes de la §2.D** (formes racine carrée, équation séculaire, subspace tracking) ne sont pas des mesures de temps mural : **aucun des papiers vérifiés ne rapporte de wall-time pour ces méthodes appliquées à une Fisher de réseau.** Tout chiffre de vitesse réelle doit être mesuré dans `fisher_drift_analysis`.
- **Les identifiants arXiv à préfixe 26xx** correspondent à des dépôts de 2026 antérieurs à septembre 2026 ; leur statut de publication en revue n'a généralement pas été contrôlé, et plusieurs n'ont pas encore été relus par les pairs (FOAM, Orth-Dion, DASH, Dion3, PRIME-SR, Pro-KLShampoo, _How Much Orthogonalization Does Muon Need?_, Curvature-Guided LoRA, FIM-LoRA, FisherAdapTune lui-même).

---

## 7. Bibliographie

### 7.1 Par domaine

**Fisher, gradient naturel, GGN — fondations et critique**

- S. Amari, _Natural Gradient Works Efficiently in Learning_, Neural Computation 10(2):251, 1998.
- J. Martens, _New insights and perspectives on the natural gradient method_, arXiv:1412.1193, JMLR 21(146), 2020.
- F. Kunstner, L. Balles, P. Hennig, _Limitations of the Empirical Fisher Approximation for Natural Gradient Descent_, arXiv:1905.12558, NeurIPS 2019.
- X. Wu, W. Yu, C. Zhang, P. Woodland, _An Improved Empirical Fisher Approximation for Natural Gradient Descent_ (iEF), arXiv:2406.06420, NeurIPS 2024.
- A. Soen, K. Sun, _Trade-Offs of Diagonal Fisher Information Matrix Estimators_, arXiv:2402.05379, 2024.
- Y. Li, F. Dangel, D. Tam, C. Raffel, _Fishers for Free? Approximating the Fisher Information Matrix by Recycling the Squared Gradient Accumulator_, arXiv:2507.18807, ICML 2025 (spotlight).
- R. Grosse et al., _Studying Large Language Model Generalization with Influence Functions_, arXiv:2308.03296, 2023.

**K-FAC et variantes**

- J. Martens, R. Grosse, _Optimizing Neural Networks with Kronecker-factored Approximate Curvature_, arXiv:1503.05671, ICML 2015.
- R. Grosse, J. Martens, _A Kronecker-factored approximate Fisher matrix for convolution layers_, arXiv:1602.01407, 2016.
- T. George, C. Laurent, X. Bouthillier, N. Ballas, P. Vincent, _Fast Approximate Natural Gradient Descent in a Kronecker-factored Eigenbasis_ (EKFAC), arXiv:1806.03884, NeurIPS 2018.
- K.-X. Gao et al., _Eigenvalue-corrected Natural Gradient Based on a New Approximation_, arXiv:2011.13609, AAAI 2021.
- D. Schnaus, J. Lee, R. Triebel, _Kronecker-Factored Optimal Curvature_ (K-FOC), NeurIPS 2021 Bayesian Deep Learning Workshop ; DLR elib 145806. **Pas d'arXiv.**
- R. Eschenhagen, A. Immer, R. E. Turner, F. Schneider, P. Hennig, _Kronecker-Factored Approximate Curvature for Modern Neural Network Architectures_, arXiv:2311.00636, NeurIPS 2023.
- W. Lin et al., _Structured Inverse-Free Natural Gradient Descent_ (SINGD), arXiv:2312.05705, ICML 2024.
- F. Dangel, J. Müller, M. Zeinhofer, _Kronecker-Factored Approximate Curvature for Physics-Informed Neural Networks_, arXiv:2405.15603, NeurIPS 2024.
- C. O. Puiu, _Randomized K-FACs: Speeding up K-FAC with Randomized Numerical Linear Algebra_, arXiv:2206.15397, 2022.
- C. O. Puiu, _Brand New K-FACs: Speeding up K-FAC with Online Decomposition Updates_, arXiv:2210.08494, 2022.
- Z. Tang et al., _SKFAC: Training Neural Networks with Faster Kronecker-Factored Approximate Curvature_, CVPR 2021.
- N. Yudin, E. Grishina, A. Veprikov, A. Beznosikov, M. Rakhuba, _DyKAF: Dynamical Kronecker Approximation of the Fisher Information Matrix_, arXiv:2511.06477, 2025.
- V. Yusupov, D. Cherniuk, E. Frolov, _Scalable Kronecker-Fisher Approximation_, arXiv:2609.02451, 2026.
- G. Zhang, C. Wang, B. Xu, R. Grosse, _Three Mechanisms of Weight Decay Regularization_, arXiv:1810.12281, 2018 (pseudocode $T_{\rm stats}/T_{\rm inv}$).
- G. Zhang, A. Botev, J. Martens, _Deep Learning without Shortcuts_, arXiv:2203.08120, 2022 (chiffres d'amortissement).
- D. Liao, F. Dangel, Y. Yu, _Efficient Bilevel Optimization with KFAC-Based Hypergradients_, arXiv:2603.29108, 2026.

**Shampoo, SOAP, AdaFisher, FisherAdapTune**

- V. Gupta, T. Koren, Y. Singer, _Shampoo: Preconditioned Stochastic Tensor Optimization_, ICML 2018.
- H.-J. Shi et al., _A Distributed Data-Parallel PyTorch Implementation of the Distributed Shampoo Optimizer_, arXiv:2309.06497, 2023.
- N. Vyas, D. Morwani, R. Zhao, M. Kwun, I. Shapira, D. Brandfonbrener, L. Janson, S. Kakade, _SOAP: Improving and Stabilizing Shampoo using Adam_, arXiv:2409.11321, ICLR 2025.
- R. Eschenhagen, A. Defazio, T.-H. Lee, R. E. Turner, H.-J. Shi, _Purifying Shampoo_, arXiv:2506.03595, 2025.
- K. Nam, S. Ahn, _FOAM: Frequency and Operator Error-Based Adaptive Damping Method… for Shampoo_, arXiv:2606.02365, 2026.
- W. Lin, S. C. Lowe, F. Dangel, R. Eschenhagen, Z. Xu, R. B. Grosse, _Understanding and Improving Shampoo and SOAP via Kullback-Leibler Minimization_ (KL-Shampoo), arXiv:2509.03378, version étendue ICLR 2026.
- R. Sun, E. Wei, _Pro-KLShampoo_, arXiv:2605.06316, 2026.
- A. Milligan, Z. Xu, S. Lacoste-Julien, F. Dangel, W. Lin, _Reparametrizing Shampoo and SOAP for Subspace Basis Updates and BFloat16 Storage_, arXiv:2605.26327, 2026.
- W. Gong, M. Scetbon, C. Ma, E. Meeds, _Towards Efficient Optimizer Design for LLM via Structured Fisher Approximation with a Low-Rank Extension_, arXiv:2502.07752, 2025.
- D. Martins Gomes, Y. Zhang, E. Belilovsky, G. Wolf, M. S. Hosseini, _AdaFisher: Adaptive Second Order Optimization via Fisher Information_, arXiv:2405.16397, ICLR 2025.
- D. Martins Gomes, _Towards Practical Second-Order Optimizers in Deep Learning: Insights from Fisher Information Analysis_, arXiv:2504.20096, 2025.
- G. Rostami, P.-H. Chen, M. S. Hosseini, _Fisher-Guided Progressive Parameter Selection for Adaptive Fine-Tuning_ (**FisherAdapTune**), arXiv:2606.10196, 2026.

**Fisher exacte, GGN exact, produits matrice-vecteur**

- B. A. Pearlmutter, _Fast Exact Multiplication by the Hessian_, Neural Computation 6(1):147–160, 1994.
- N. N. Schraudolph, _Fast Curvature Matrix-Vector Products for Second-Order Gradient Descent_, Neural Computation 14(7):1723–1738, 2002.
- J. Martens, _Deep learning via Hessian-free optimization_, ICML 2010 ; Martens & Sutskever, ICML 2011.
- I. Goodfellow, _Efficient Per-Example Gradient Computations_, arXiv:1510.01799, 2015.
- F. Dangel, F. Kunstner, P. Hennig, _BackPACK: Packing more into backprop_, arXiv:1912.10985, ICLR 2020.
- K. Osawa, S. Ishikawa, R. Yokota, S. Li, T. Hoefler, _ASDL: A Unified Interface for Gradient Preconditioning in PyTorch_, arXiv:2305.04684, 2023.
- F. Dangel, R. Eschenhagen, W. Ormaniec, A. Fernandez, L. Tatzel, A. Kristiadi, _Position: Curvature Matrices Should Be Democratized via Linear Operators_ (curvlinops), arXiv:2501.19183, 2025.
- E. Daxberger et al., _Laplace Redux — Effortless Bayesian Deep Learning_, arXiv:2106.14806, NeurIPS 2021.
- A. Bernacchia, M. Lengyel, G. Hennequin, _Exact natural gradient in deep linear networks_, NeurIPS 2018.
- D. Buffelli et al., _Exact, Tractable Gauss-Newton Optimization in Deep Reversible Architectures Reveal Poor Generalization_, arXiv:2411.07979, 2024.
- F. Benzing, _Gradient Descent on Neurons and its Link to Approximate Second-Order Optimization_, arXiv:2201.12250, ICML 2022.
- N. Abreu, N. Vyas, S. Kakade, D. Morwani, _The Potential of Second-Order Optimization for LLMs: A Study with Full Gauss-Newton_, arXiv:2510.09378, 2025.
- A. Galashov, N. Da Costa, L. Xu, P. Hennig, A. Gretton, _Closed-Form Last Layer Optimization_, arXiv:2510.04606, 2025.
- I. Emirahmetoglu, D. E. Stewart, _Training Autoencoders Using Stochastic Hessian-Free Optimization with LSMR_, arXiv:2504.13302, 2025.

**Formulation duale / Gram / Woodbury**

- M. Soori, B. Can, M. Mu, M. Gürbüzbalaban, M. Mehri Dehnavi, _TENGraD: Time-Efficient Natural Gradient Descent with Exact Fisher-Block Inversion_, arXiv:2106.03947, 2021.
- M. Yang, D. Xu, Z. Wen, M. Chen, P. Xu, _Sketchy Empirical Natural Gradient Methods for Deep Learning_ (SENG), arXiv:2006.05924, ICML 2022.
- M. Korbit, A. Adeoye, A. Bemporad, M. Zanon, _Exact Gauss-Newton Optimization for Training Deep Neural Networks_ (EGN), arXiv:2405.14402, 2024.
- A. Jnini, F. Vella, _Dual Natural Gradient Descent for Scalable Training of Physics-Informed Neural Networks_, arXiv:2505.21404, 2025.
- A. Guzmán-Cordero, F. Dangel, G. Goldshlager, M. Zeinhofer, _Improving Energy Natural Gradient Descent through Woodbury, Momentum, and Randomization_, arXiv:2505.12149, 2025.
- Y. Ren, D. Goldfarb, _Efficient Subsampled Gauss-Newton and Natural Gradient Methods for Training Neural Networks_, arXiv:1906.02353, 2019.
- Y. Ren, D. Goldfarb, _Tensor Normal Training for Deep Learning Models_, arXiv:2106.02925, NeurIPS 2021 (Spotlight).
- Y. Ren, A. Bahamou, D. Goldfarb, _Kronecker-factored Quasi-Newton Methods for Deep Learning_, arXiv:2102.06737, 2021.
- D. Goldfarb, Y. Ren, A. Bahamou, _Practical Quasi-Newton Methods for Training Deep Neural Networks_, arXiv:2006.08877, NeurIPS 2020.
- F. Roosta-Khorasani, M. W. Mahoney, _Sub-Sampled Newton Methods I & II_, arXiv:1601.04737, arXiv:1601.04738.
- C. Niu, Z. Liao, Z. Ling, M. W. Mahoney, _Fundamental Bias in Inverting Random Sampling Matrices with Application to Sub-sampled Newton_, arXiv:2502.13583, 2025.

**Physique : reconfiguration stochastique, MinSR, VMC**

- S. Sorella, _Generalized Lanczos Algorithm for Variational Quantum Monte Carlo_, PRB 64, 024512 (2001), arXiv:cond-mat/0009149.
- A. Chen, M. Heyl, _Empowering deep neural quantum states through efficient optimization_, **Nature Physics 20, 1476–1481 (2024)**, arXiv:2302.01941.
- R. Rende, L. L. Viteritti, L. Bardone, F. Becca, S. Goldt, _A simple linear algebra identity to optimize Large-Scale Neural Network Quantum States_, **Communications Physics 7, 260 (2024)**, arXiv:2310.05715.
- G. Goldshlager, N. Abrahamsen, L. Lin, _A Kaczmarz-inspired approach to accelerate the optimization of neural network wavefunctions_ (SPRING), J. Comput. Phys. 516, 113351 (2024), arXiv:2401.10190.
- G. Goldshlager, J. Hu, L. Lin, _A Sketch-and-Project Analysis of Subsampled Natural Gradient Algorithms_, arXiv:2508.21022, 2025.
- Y. Wang, X. Liu, _Momentum Stability and Adaptive Control in Stochastic Reconfiguration_ (PRIME-SR), arXiv:2604.18357, 2026.
- M. Schmitt, M. Heyl, _Quantum Many-Body Dynamics in Two Dimensions with Artificial Neural Networks_, PRL 125, 100503 (2020), arXiv:1912.08828 (critère SNR).
- D. Zhou, H. Chen, C. H. Ho, X. Liu, C. Ortner, _Stochastic Reconfiguration with Warm-Started SVD_, arXiv:2512.05749, 2025.
- D. Pfau, J. S. Spencer, A. G. de G. Matthews, W. M. C. Foulkes, _Ab-Initio Solution of the Many-Electron Schrödinger Equation with Deep Neural Networks_ (FermiNet), arXiv:1909.02487, PRResearch 2, 033429 (2020).
- M. Drissi, J. W. T. Keeble, J. Rozalén Sarmiento, A. Rios, _Second-order optimisation strategies for neural network quantum states_, arXiv:2401.17550, Communications Physics 2024.
- R. Peng, G. K.-L. Chan, _An Analysis of First- and Quasi-Second-Order Optimization Algorithms in Variational Monte Carlo_, arXiv:2502.19576, 2025.
- A. Attar, A. M. Aboussalah, M. Hibat-Allah, _Quantum Geometric Tensor Preconditioning for Stable Training of Recurrent Neural Quantum States_, arXiv:2608.18065, 2026.
- J. Stokes, J. Izaac, N. Killoran, G. Carleo, _Quantum Natural Gradient_, Quantum 4, 269 (2020), arXiv:1909.02108.
- F. Vicentini et al., _NetKet 3_, SciPost Phys. Codebases 7 (2022), arXiv:2112.10526.

**RandNLA, sketching, rang faible**

- P.-G. Martinsson, J. A. Tropp, _Randomized Numerical Linear Algebra: Foundations & Algorithms_, arXiv:2002.01387, Acta Numerica 2020.
- R. Murray et al., _RandNLA: A Perspective on the Field With an Eye to Software_, arXiv:2302.11474, 2023.
- N. Halko, P.-G. Martinsson, J. A. Tropp, _Finding structure with randomness_, arXiv:0909.4061, SIAM Review 2011.
- J. A. Tropp, A. Yurtsever, M. Udell, V. Cevher, _Practical sketching algorithms for low-rank matrix approximation_, arXiv:1609.00048, SIMAX 2017.
- Y. Nakatsukasa, _Fast and stable randomized low-rank matrix approximation_ (Nyström généralisé), arXiv:2009.11392, 2020.
- Z. Frangella, J. A. Tropp, M. Udell, _Randomized Nyström Preconditioning_, arXiv:2110.02820, SIMAX.
- Z. Frangella, P. Rathore, S. Zhao, M. Udell, _SketchySGD_, arXiv:2211.08597 ; _PROMISE_, arXiv:2309.02014.
- Y. Chen, E. N. Epperly, J. A. Tropp, R. J. Webber, _Randomly pivoted Cholesky_, arXiv:2207.06503, CPAM ; _Embrace rejection_ (version par blocs), arXiv:2410.03969.
- A. Fernandez, F. Dangel, P. Hennig, F. Schneider, _Sketching Low-Rank Plus Diagonal Matrices_ (SKETCHLORD), arXiv:2509.23587, 2025.
- M. Pilanci, M. J. Wainwright, _Newton Sketch_, arXiv:1505.02250, SIOPT 2017.
- J. Zhang, M. Pilanci, _Optimal Shrinkage for Distributed Second-Order Optimization_, arXiv:2402.01956, 2024.
- E. Liberty, _Simple and Deterministic Matrix Sketching_, arXiv:1206.0594 ; M. Ghashami, E. Liberty, J. Phillips, D. P. Woodruff, _Frequent Directions_, arXiv:1501.01711, SICOMP 2016.
- V. Feinberg, X. Chen, Y. J. Sun, R. Anil, E. Hazan, _Sketchy: Memory-efficient Adaptive Regularization with Frequent Directions_, arXiv:2302.03764, NeurIPS 2023.
- V. Braverman, P. Drineas, C. Musco, C. Musco, J. Upadhyay, D. P. Woodruff, S. Zhou, _Near Optimal Linear Algebra in the Online and Sliding Window Models_, arXiv:1805.03765, FOCS 2020.
- H. Yao, X. Chen, L. Chen, _Optimal Approximate Matrix Multiplication over Sliding Windows_, arXiv:2502.18830, 2025.
- R. A. Meyer, C. Musco, C. Musco, D. P. Woodruff, _Hutch++_, arXiv:2010.09649, SOSA 2021.
- E. N. Epperly, J. A. Tropp, R. J. Webber, _XTrace_, arXiv:2301.07825, SIMAX 2024.
- D. Persson, A. Cortinovis, D. Kressner, _Improved variants of the Hutch++ algorithm_ (Nyström++), arXiv:2109.10659, SIMAX 2022.
- M. B. Cohen, C. Musco, C. Musco, _Input Sparsity Time Low-Rank Approximation via Ridge Leverage Score Sampling_, arXiv:1511.07263, SODA 2017.
- D. P. Woodruff, _Sketching as a Tool for Numerical Linear Algebra_, Found. Trends TCS 10(1-2), 2014.
- H. Diao, Z. Song, W. Sun, D. P. Woodruff, _Sketching for Kronecker Product Regression and P-splines_, arXiv:1712.09473, AISTATS 2018.
- H. Diao, R. Jayaram, Z. Song, W. Sun, D. P. Woodruff, _Optimal Sketching for Kronecker Product Regression and Low Rank Approximation_, arXiv:1909.13384, NeurIPS 2019.
- M. Fahrbach, G. Fu, M. Ghadiri, _Subquadratic Kronecker Regression with Applications to Tensor Decomposition_, arXiv:2209.04876, NeurIPS 2022.
- D. P. Woodruff, S. Zhou, _Consistent Low-Rank Approximation_, arXiv:2603.02148, 2026.
- E. J. Candès, X. Li, Y. Ma, J. Wright, _Robust Principal Component Analysis?_, arXiv:0912.3599, JACM 2011.
- J. Tanner, A. Thompson, S. Vary, _Matrix rigidity and the ill-posedness of Robust PCA and matrix completion_, arXiv:1811.05919, 2018.

**Matrices structurées, Kronecker de rang $r$, tenseurs**

- C. Van Loan, N. Pitsianis, _Approximation with Kronecker Products_, Cornell CS Technical Report, nov. 1992 (eCommons 1813/5484).
- C. Cai, R. Chen, H. Xiao, _KoPA: Automated Kronecker Product Approximation_, JMLR 23, 2022.
- Y. Voet, L. De Novellis, _Identifying Kronecker product factorizations_, arXiv:2510.25292, 2025.
- M. Dressler, A. Uschmajew, V. Chandrasekaran, _Kronecker Product Approximation of Operators in Spectral Norm via Alternating SDP_, arXiv:2207.03186, 2022.
- M. E. Kilmer, A. K. Saibaba, _Structured Matrix Approximations via Tensor Decompositions_, arXiv:2105.01170, 2021.
- Y. Voet, _Preconditioning Techniques for Generalized Sylvester Matrix Equations_, arXiv:2307.07884, NLAA 2025.
- K. Greenewald, A. O. Hero, _Regularized Block Toeplitz Covariance Matrix Estimation via Kronecker Product Expansions_, arXiv:1402.5568, 2014.
- B. Roś, F. Bijma, J. C. de Munck, M. C. M. de Gunst, _Existence and uniqueness of the MLE for models with a Kronecker product covariance structure_, arXiv:1410.2118, 2014.
- T. Dao et al., _Monarch: Expressive Structured Matrices for Efficient and Accurate Training_, arXiv:2204.00595, ICML 2022.
- S. Qiu, A. Potapczynski, M. Finzi, M. Goldblum, A. G. Wilson, _Compute Better Spent: Replacing Dense Layers with Structured Matrices_, arXiv:2406.06248, ICML 2024.
- A. Potapczynski et al., _Searching for Efficient Linear Layers over a Continuous Space of Structured Matrices_, arXiv:2410.02117, NeurIPS 2024.
- A. Gonon, L. Zheng, P. Carrivain, Q.-T. Le, _Fast inference with Kronecker-sparse matrices_, arXiv:2405.15013, 2024.
- Y. Li, H. Yang, E. R. Martin, K. L. Ho, L. Ying, _Butterfly Factorization_, arXiv:1502.01379, MMS/SIAM.
- I. V. Oseledets, _Tensor-Train Decomposition_, SIAM J. Sci. Comput. 33(5):2295–2317, 2011.
- L. Grasedyck, D. Kressner, C. Tobler, _A literature survey of low-rank tensor approximation techniques_, arXiv:1302.7121.
- Q. Zhao, M. Sugiyama, A. Cichocki, _Learning Efficient Tensor Representations with Ring Structure Networks_, arXiv:1705.08286, 2017.
- A. Novikov, D. Podoprikhin, A. Osokin, D. Vetrov, _Tensorizing Neural Networks_, 2015.

**Matrices hiérarchiques et problèmes inverses**

- C. Chen, S. Reiz, C. Yu, H.-J. Bungartz, G. Biros, _Fast Approximation of the Gauss-Newton Hessian Matrix for the Multilayer Perceptron_, arXiv:1910.12184, 2019.
- I. Ambartsumyan et al., _Hierarchical Matrix Approximations of Hessians Arising in Inverse Problems Governed by PDEs_, arXiv:2003.10173, SISC 42(5), 2020.
- T. Hartland, G. Stadler, M. Perego, K. Liegeois, N. Petra, _Hierarchical off-diagonal low-rank approximation of Hessians in inverse problems_, arXiv:2301.03644, Inverse Problems 2023.
- C. Chen, P.-G. Martinsson, _Solving Linear Systems on a GPU with Hierarchically Off-Diagonal Low-Rank Approximations_, arXiv:2208.06290, 2022.
- S. Zampini, W. Boukaram, G. Turkiyyah, O. Knio, D. E. Keyes, _H2Opus_, arXiv:2109.05451, Adv. Comput. Math. 2022.
- W. H. Boukaram, Y. Liu, P. Ghysels, X. S. Li, _Adaptive Sketching Based Construction of H2 Matrices on GPUs_, arXiv:2506.16759, 2025.
- F. Schäfer, M. Katzfuss, H. Owhadi, _Sparse Cholesky Factorization by Kullback–Leibler Minimization_, arXiv:2004.14455, SISC 43(3), 2021.
- S. Ambikasaran, D. Foreman-Mackey, L. Greengard, D. W. Hogg, M. O'Neil, _Fast Direct Methods for Gaussian Processes_, arXiv:1403.6015, 2014.
- T. F. Coleman, J. J. Moré, _Estimation of sparse Hessian matrices and graph coloring problems_, Math. Prog. 28:243–270, 1984.
- A. H. Gebremedhin, F. Manne, A. Pothen, _What Color Is Your Jacobian?_, SIAM Review 47(4), 2005.

**Théorie des matrices aléatoires, shrinkage, spectre des réseaux**

- O. Ledoit, M. Wolf, _A well-conditioned estimator for large-dimensional covariance matrices_, J. Multivariate Anal. 88(2):365–411, 2004.
- O. Ledoit, M. Wolf, _Nonlinear shrinkage estimation…_, arXiv:1207.5322, Ann. Statist. 40(2), 2012 ; _Analytical nonlinear shrinkage…_, Ann. Statist. 48(5), 2020.
- J. Bun, J.-P. Bouchaud, M. Potters, _Cleaning large correlation matrices: tools from random matrix theory_, arXiv:1610.08104, Physics Reports.
- D. L. Donoho, M. Gavish, I. M. Johnstone, _Optimal Shrinkage of Eigenvalues in the Spiked Covariance Model_, arXiv:1311.0851, Ann. Statist. 46(4), 2018.
- Y. Chen, A. Wiesel, Y. C. Eldar, A. O. Hero, _Shrinkage Algorithms for MMSE Covariance Estimation_ (OAS), arXiv:0907.4698, IEEE TSP 2010.
- J. Alt, L. Erdős, T. Krüger, Y. Nemish, _Location of the spectrum of Kronecker random matrices_, arXiv:1706.08343, AIHP 55(2), 2019.
- D. Granziol, N. Baskerville, _A Random Matrix Theory Approach to Damping in Deep Learning_, arXiv:2011.08181, J. Phys. Complexity 2022.
- V. Papyan, _Traces of Class/Cross-Class Structure Pervade Deep Learning Spectra_, arXiv:2008.11865, JMLR 21, 2020.
- B. Ghorbani, S. Krishnan, Y. Xiao, _An Investigation into Neural Net Optimization via Hessian Eigenvalue Density_, arXiv:1901.10159, ICML 2019.
- L. Sagun, U. Evci, V. U. Güney, Y. Dauphin, L. Bottou, _Empirical Analysis of the Hessian of Over-Parametrized Neural Networks_, arXiv:1706.04454.
- Z. Yao, A. Gholami, K. Keutzer, M. Mahoney, _PyHessian_, arXiv:1912.07145, 2019.
- S. Ubaru, J. Chen, Y. Saad, _Fast Estimation of tr(f(A)) via Stochastic Lanczos Quadrature_, SIMAX 38(4), 2017.
- T. Rogers, I. Pérez Castillo, R. Kühn, K. Takeda, _Cavity Approach to the Spectral Density of Sparse Symmetric Random Matrices_, arXiv:0803.1553, PRE 78, 2008.

**Filtrage, identification, traitement du signal**

- Y. Ollivier, _Online Natural Gradient as a Kalman Filter_, arXiv:1703.00209, Electron. J. Statist. 12(2), 2018 ; _The EKF is a Natural Gradient Descent in Trajectory Space_, arXiv:1901.00696.
- P. G. Chang, G. Durán-Martín, A. Y. Shestopaloff, M. Jones, K. Murphy, _Low-rank extended Kalman filtering for online learning of neural networks_, arXiv:2305.19535, 2023.
- H. Abdi, M. Sun, A. Zhang, S. Kaski, W. Pan, _LoKO: Low-Rank Kalman Optimizer for Online Fine-Tuning of Large Models_, arXiv:2410.11551, 2024.
- K. Tracy, _A Square-Root Kalman Filter Using Only QR Decompositions_, arXiv:2208.06452, 2022.
- J. R. Bunch, C. P. Nielsen, D. C. Sorensen, _Rank-one modification of the symmetric eigenproblem_, Numer. Math. 31:31–48, 1978.
- M. Gu, S. C. Eisenstat, _A Stable and Efficient Algorithm for the Rank-One Modification of the Symmetric Eigenproblem_, SIMAX.
- M. Brand, _Fast low-rank modifications of the thin singular value decomposition_, Linear Algebra Appl. 415(1):20–30, 2006.
- S. Bonnabel, R. Sepulchre, _The geometry of low-rank Kalman filters_, arXiv:1203.4049, 2012.
- J. Schmidt, P. Hennig, J. Nick, F. Tronarp, _The Rank-Reduced Kalman Filter_, arXiv:2306.07774, 2023.
- F. Nobile, T. Trigo Trindade, _Dynamical Low-Rank Approximations for Kalman Filtering_, arXiv:2509.11210, 2025.
- B. Yang, _Projection approximation subspace tracking_ (PAST), IEEE TSP 43(1):95–107, 1995.
- L. Balzano, R. Nowak, B. Recht, _Online Identification and Tracking of Subspaces_ (GROUSE), arXiv:1006.4046, 2010.
- D. Huang, J. Niles-Weed, R. Ward, _Streaming k-PCA: … beyond rank-one updates_, arXiv:2102.03646, 2021.
- S. Kumar, P. Sarkar, _Streaming PCA for Markovian Data_, arXiv:2305.02456, 2023.
- B. Lai, D. S. Bernstein, _SIFt-RLS: Subspace of Information Forgetting Recursive Least Squares_, arXiv:2404.10844, 2024 ; _Generalized Forgetting RLS_, arXiv:2308.04259 ; _Adaptive Kalman Filtering from RLS Forgetting Algorithms_, arXiv:2404.10914.
- A. L. Bruce, A. Goel, D. S. Bernstein, _Convergence and Consistency of RLS with Variable-Rate Forgetting_, arXiv:2003.02737, 2020.
- F. Fraccaroli, A. Peruffo, M. Zorzi, _A New RLS Method with Multiple Forgetting Schemes_, arXiv:1503.07338, 2015.
- B. Lai, D. Panagou, D. S. Bernstein, _RLS with Fading Regularization…_, arXiv:2501.04566, 2025.
- S. A. Vorobyov, A. B. Gershman, Z.-Q. Luo, _Robust Adaptive Beamforming Using Worst-Case Performance Optimization_, IEEE TSP 51(2):313–324, 2003.
- B. D. Carlson, _Covariance matrix estimation errors and diagonal loading in adaptive arrays_, IEEE TAES 24(4):397–401, 1988.
- Y.-T. Tsai, B. Su, Y. Tsao, S.-S. Wang, _Robust Beamforming… Subspace-Constrained Diagonal Loading_, arXiv:1602.02690, 2016.
- T.-J. Shan, M. Wax, T. Kailath, _On spatial smoothing for direction-of-arrival estimation of coherent signals_, IEEE TASSP 33(4):806–811, 1985.
- G. Gaspari, S. E. Cohn, _Construction of correlation functions in two and three dimensions_, Q. J. R. Meteorol. Soc. 125(554):723–757, 1999.
- R. Furrer, T. Bengtsson, _Estimation of high-dimensional prior and posterior covariance matrices in Kalman filter variants_, J. Multivariate Anal. 98(2):227–255, 2007.
- P. J. Bickel, E. Levina, _Covariance regularization by thresholding_, arXiv:0901.3079, Ann. Statist. 2008.
- J. L. Anderson, _An adaptive covariance inflation error correction algorithm for ensemble filters_, Tellus A 59(2), 2007 ; _Spatially and temporally varying adaptive covariance inflation_, Tellus A, 2009.
- X. Luo, I. Hoteit, _Robust ensemble filtering and its relation to covariance inflation_, arXiv:1108.0158, 2011.
- G. Evensen, _Sequential data assimilation…_, J. Geophys. Res. 99(C5), 1994 ; C. H. Bishop, B. J. Etherton, S. J. Majumdar, ETKF, Mon. Wea. Rev. 129(3), 2001.
- M. A. Iglesias, K. J. H. Law, A. M. Stuart, _The Ensemble Kalman Filter for Inverse Problems_, arXiv:1209.2736 ; C. Schillings, A. M. Stuart, arXiv:1602.02020.
- E. D. Nino-Ruiz, A. Sandu, arXiv:1502.00301 ; A. A. Popov, A. Sandu, E. D. Nino-Ruiz, G. Evensen, arXiv:2003.00354 (shrinkage dans un filtre d'ensemble).
- B. Ait-El-Fquih, I. Hoteit, _A Structurally Localized Ensemble Kalman Filtering Approach_, arXiv:2603.03926, 2026.

**Fonctions de matrices, Krylov, quasi-Newton**

- N. J. Higham, _Functions of Matrices: Theory and Computation_, SIAM, 2008.
- C.-H. Guo, N. J. Higham, _A Schur–Newton Method for the Matrix pth Root and Its Inverse_, SIMAX 28(3):788–804, 2006.
- Y. Nakatsukasa, R. W. Freund, _Computing Fundamental Matrix Decompositions Accurately via the Matrix Sign Function in Two Iterations: The Power of Zolotarev's Functions_, SIAM Review 58(3):461–493, 2016.
- E. S. Gawlik, _Zolotarev Iterations for the Matrix Square Root_, arXiv:1804.11000, SIMAX.
- R. H. Byrd, J. Nocedal, R. B. Schnabel, _Representations of quasi-Newton matrices and their use in limited memory methods_, Math. Prog. 63:129–156, 1994.
- J. B. Erway, R. F. Marcia, _On solving large-scale limited-memory quasi-Newton equations_, arXiv:1510.06378.
- J. Brust, J. B. Erway, R. F. Marcia, _On Solving L-SR1 Trust-Region Subproblems_, arXiv:1506.07222, COAP 66(2), 2017.
- R. H. Byrd, S. L. Hansen, J. Nocedal, Y. Singer, _A Stochastic Quasi-Newton Method for Large-Scale Optimization_, arXiv:1401.7020, SIOPT 26(2), 2016.
- M. L. Parks, E. de Sturler, G. Mackey, D. D. Johnson, S. Maiti, _Recycling Krylov Subspaces for Sequences of Linear Systems_ (GCRO-DR), SISC 28(5):1651–1674, 2006.
- K. Carlberg, V. Forstall, R. Tuminaro, _Krylov-subspace recycling via the POD-augmented conjugate-gradient method_, arXiv:1512.05820, 2015.
- L. Burke, S. Güttel, _Krylov Subspace Recycling With Randomized Sketching For Matrix Functions_, arXiv:2308.02290, SIMAX.
- O. Vinyals, D. Povey, _Krylov Subspace Descent for Deep Learning_, arXiv:1111.4259, AISTATS 2012.
- J. A. Loe, R. B. Morgan, _Toward efficient polynomial preconditioning for GMRES_, arXiv:1911.07065, NLAA 29, 2022.
- D. C.-L. Fong, M. A. Saunders, _LSMR_, arXiv:1006.0758, SISC 33(5), 2011.
- Y. Liu, F. Roosta, _MINRES: From Negative Curvature Detection to Monotonicity Properties_, arXiv:2206.05732, SIOPT.
- Y. Nesterov, B. T. Polyak, _Cubic regularization of Newton method_, Math. Prog. 108(1), 2006 ; C. Cartis, N. I. M. Gould, P. L. Toint, ARC Parts I & II, Math. Prog. 127.
- C. Bekas, E. Kokiopoulou, Y. Saad, _An estimator for the diagonal of a matrix_, Appl. Numer. Math. 57(11-12), 2007.
- R. A. Baston, Y. Nakatsukasa, _Stochastic diagonal estimation: probabilistic bounds and an improved algorithm_, arXiv:2201.10684, 2022.

**Muon, orthogonalisation, systèmes et précision réduite**

- K. Jordan et al., _Muon: MomentUm Orthogonalized by Newton-Schulz_, 2024 (kellerjordan.github.io/posts/muon/, github.com/KellerJordan/Muon).
- J. Bernstein, L. Newhouse, _Modular Duality in Deep Learning_, arXiv:2410.21265, 2024.
- Kimi Team, _Muon is Scalable for LLM Training_ (Moonlight), arXiv:2502.16982, 2025.
- N. Amsel, D. Persson, C. Musco, R. Gower, _Polar Express_, arXiv:2505.16932, ICLR 2026 (Oral).
- E. Grishina, M. Smirnov, M. Rakhuba, _Accelerating Newton-Schulz Iteration… via Chebyshev-type Polynomials_ (CANS), arXiv:2506.10935, 2025.
- H. Huang, _How Much Orthogonalization Does Muon Need?_, arXiv:2606.00371, 2026.
- I.-V. Modoranu, P. Zmushko, E. Schultheis, M. Safaryan, D. Alistarh, _DASH: Faster Shampoo via Batched Block Preconditioning and Efficient Inverse-Root Solvers_, arXiv:2602.02016, 2026.
- K. Ahn, B. Xu, N. Abreu, Y. Fan, G. Magakyan, P. Sharma, Z. Zhan, J. Langford, _Dion: Distributed Orthonormalized Updates_, arXiv:2504.05295, 2025.
- N. Amsel, Z. Zhang, K. Ahn, N. Naeimi, Y. Feng, B. Chen, T. Dao, J. Langford, _Dion3_, arXiv:2608.11612, 2026.
- K. Ahn, N. Amsel, J. Langford, _Dion2: A Simple Method to Shrink Matrix in Muon_, arXiv:2512.16928, 2025.
- T. Nakamori et al., _Orth-Dion: Eliminating Geometric Mismatch in Distributed Low-Rank Spectral Optimization_, arXiv:2605.16341, 2026.
- A. Khaled, K. Ozkara, T. Yu, M. Hong, Y. Park, _MuonBP: Faster Muon via Block-Periodic Orthogonalization_, arXiv:2510.16981, 2025.
- A. Dev, A. Bohara, M. Takáč, S. Horváth, _CacheMuon_, arXiv:2606.16371, 2026.
- J. C. Pauloski et al., _KAISA: An Adaptive Second-Order Optimizer Framework_, arXiv:2107.01739, SC'21.
- L. Zhang, S. Shi, W. Wang, B. Li, _Scalable K-FAC Training… With Distributed Preconditioning_ (DP-KFAC), arXiv:2206.15143, 2022.
- S. Shi, L. Zhang, B. Li, _Accelerating Distributed K-FAC with Smart Parallelism_ (SPD-KFAC), arXiv:2107.06533, 2021.
- S. Wang, P. Zhou, J. Li, H. Huang, _4-bit Shampoo for Memory-Efficient Network Training_, arXiv:2405.18144, NeurIPS 2024.
- J. Li, K. Ding, K.-C. Toh, P. Zhou, _Memory-Efficient 4-bit Preconditioned Stochastic Optimization_, arXiv:2412.10663, 2024.
- Y. Su, X. Zhang, X. Liu, W. Zhao, T. Zhang, _MuonQ_, arXiv:2605.11396, 2026.
- H. Xi, C. Cai, L. Zhu, Y. Lu, K. Keutzer, J. Chen, S. Han, _COAT: Compressing Optimizer States and Activation for Memory-Efficient FP8 Training_, arXiv:2410.19313, ICLR 2025.
- T. Dettmers, M. Lewis, S. Shleifer, L. Zettlemoyer, _8-bit Optimizers via Block-wise Quantization_, arXiv:2110.02861.
- Z. Lu, Y. Zhang, Q. Yang, S. Armour, _Asteria_, arXiv:2605.16184, 2026.
- Y. Wang et al., _Pipelined Dense Symmetric EVD on Multi-GPU_, arXiv:2511.16174, 2025 ; R. Ringoot, R. Alomairy, A. Edelman, _Memory-Aware Bulge-Chasing on GPUs_, arXiv:2510.12705, 2025 ; _Efficient GPU-Centered SVD_, arXiv:2508.11467, 2025.
- M. Sheibanian, P. Shaeri, A. Beigi, R. T. Woo, A. Keluskar, _Tri-Accel_, arXiv:2508.16905, 2025.
- Y. Shao, I. N. Athanasiadis, G. van Voorn, T. Kapoor, _Curvature-aware dynamic precision approach for PINNs_, arXiv:2606.04736, 2026.
- Documentation PyTorch (`torch.compile`, `foreach_map`, `torch.func`), blog PyTorch _HadaCore_, blog NVIDIA Megatron / `emerging_optimizers`, blog Dao AI Lab _Gram Newton-Schulz_.

**PEFT et courbure (2025–2026)**

- J. Zhang, A. Proutière, _Curvature-Guided LoRA: Matching Full Fine-Tuning in Function Space_, arXiv:2603.29824, 2026.
- Y. Liu, A. Ghezelbash, _Between Gradient and Natural Gradient: A Continuum of LoRA Initializations_, arXiv:2607.26247, 2026.
- Y. Feng, C. Lin, C.-M. Kao, _Learning in the Fisher Subspace: A Guided Initialization for LoRA Fine-Tuning_, arXiv:2605.01046, 2026.
- S. Sathyavageeswaran, _FIM-LoRA_, arXiv:2605.16800, 2026.
- F. Zhang, M. Pilanci, _Riemannian Preconditioned LoRA for Fine-Tuning Foundation Models_, arXiv:2402.02347, 2024.

**Autres**

- A. Buchholz, F. Wenzel, S. Mandt, _Quasi-Monte Carlo Variational Inference_, arXiv:1807.01604, ICML 2018.
- J. C. Spall, _Monte Carlo Computation of the Fisher Information Matrix in Nonstandard Settings_, J. Comput. Graph. Statist. 14(4), 2005.
- D. Rothchild et al., _FetchSGD_, arXiv:2007.07682, 2020.
- Z. Huang, L. Van Gool, _A Riemannian Network for SPD Matrix Learning_, arXiv:1608.04233 ; Z. Lin, arXiv:1908.09326 ; S. Chewi, T. Maunu, P. Rigollet, A. J. Stromme, arXiv:2001.01700 (géométrie SPD / Bures-Wasserstein).

### 7.2 État de vérification des identifiants

**Vérifiés sur source primaire** (titre, auteurs et date de soumission v1 confirmés sur arxiv.org ou sur la page de l'éditeur) — 200 identifiants, soit la totalité de ceux cités dans ce rapport à l'exception des cinq listés ci-dessous :

0803.1553 · 0901.3079 · 0907.4698 · 0909.4061 · 0912.3599 · 1006.0758 · 1006.4046 · 1108.0158 · 1111.4259 · 1203.4049 · 1206.0594 · 1207.5322 · 1209.2736 · 1302.7121 · 1311.0851 · 1401.7020 · 1402.5568 · 1403.6015 · 1410.2118 · 1412.1193 · 1501.01711 · 1502.00301 · 1502.01379 · 1503.05671 · 1503.07338 · 1505.02250 · 1506.07222 · 1510.01799 · 1510.06378 · 1511.07263 · 1512.05820 · 1601.04737 · 1601.04738 · 1602.01407 · 1602.02020 · 1602.02690 · 1608.04233 · 1609.00048 · 1610.08104 · 1703.00209 · 1705.08286 · 1706.04454 · 1706.08343 · 1712.09473 · 1804.11000 · 1805.03765 · 1806.03884 · 1807.01604 · 1810.12281 · 1811.05919 · 1901.00696 · 1901.10159 · 1905.12558 · 1906.02353 · 1908.09326 · 1909.02108 · 1909.02487 · 1909.13384 · 1910.12184 · 1911.07065 · 1912.07145 · 1912.08828 · 1912.10985 · 2001.01700 · 2002.01387 · 2003.00354 · 2003.02737 · 2003.10173 · 2004.14455 · 2006.05924 · 2006.08877 · 2007.07682 · 2008.11865 · 2009.11392 · 2010.09649 · 2011.08181 · 2011.13609 · 2102.03646 · 2102.06737 · 2105.01170 · 2106.02925 · 2106.03947 · 2106.14806 · 2107.01739 · 2107.06533 · 2109.05451 · 2109.10659 · 2110.02820 · 2110.02861 · 2112.10526 · 2201.10684 · 2201.12250 · 2203.06105 · 2203.08120 · 2204.00595 · 2206.05732 · 2206.15143 · 2206.15397 · 2207.03186 · 2207.06503 · 2208.06290 · 2208.06452 · 2209.04876 · 2210.08494 · 2211.08597 · 2301.03644 · 2301.07825 · 2302.01941 · 2302.03764 · 2302.11474 · 2305.02456 · 2305.04684 · 2305.19535 · 2306.07774 · 2307.07884 · 2308.02290 · 2308.03296 · 2308.04259 · 2309.02014 · 2309.06497 · 2310.05715 · 2311.00636 · 2312.05705 · 2401.10190 · 2401.17550 · 2402.01956 · 2402.02347 · 2402.05379 · 2404.10844 · 2404.10914 · 2405.14402 · 2405.15013 · 2405.15603 · 2405.16397 · 2405.18144 · 2406.06248 · 2406.06420 · 2409.11321 · 2410.02117 · 2410.03969 · 2410.11551 · 2410.19313 · 2410.21265 · 2411.07979 · 2412.10663 · 2501.04566 · 2501.19183 · 2502.07752 · 2502.13583 · 2502.16982 · 2502.18830 · 2502.19576 · 2504.05295 · 2504.13302 · 2504.20096 · 2505.12149 · 2505.16932 · 2505.21404 · 2506.03595 · 2506.10935 · 2506.16759 · 2507.18807 · 2508.11467 · 2508.16905 · 2508.21022 · 2509.03378 · 2509.11210 · 2509.23587 · 2510.04606 · 2510.09378 · 2510.12705 · 2510.16981 · 2510.25292 · 2511.06477 · 2511.16174 · 2512.05749 · 2512.16928 · 2602.02016 · 2603.02148 · 2603.03926 · 2603.29108 · 2603.29824 · 2604.18357 · 2605.01046 · 2605.06316 · 2605.11396 · 2605.16184 · 2605.16341 · 2605.16800 · 2605.26327 · 2606.00371 · 2606.02365 · 2606.04736 · 2606.10196 · 2606.16371 · 2607.26247 · 2608.11612 · 2608.18065 · 2609.02451 · cond-mat/0009149

**Cités uniquement en §6.2 comme réserves, et non utilisés comme appui d'un argument** : 1612.06013 (Gower & Richtárik — entrée non confirmée sur arxiv.org) ; 2008.01440, 2208.07095, 2601.01575, 2608.22129 (existence et titre confirmés, **listes d'auteurs non vérifiées individuellement**).

Les références sans identifiant arXiv (Van Loan & Pitsianis 1992, K-FOC, Pearlmutter 1994, Schraudolph 2002, Martens ICML 2010, Sorella 2001 hors préprint, Gaspari & Cohn 1999, Vorobyov et al. 2003, Carlson 1988, Shan–Wax–Kailath 1985, Bunch–Nielsen–Sorensen 1978, Byrd–Nocedal–Schnabel 1994, Coleman & Moré 1984, Ledoit & Wolf 2004 et 2020, Guo & Higham 2006, Nakatsukasa & Freund 2016, Higham 2008, Oseledets 2011, Yang 1995, Anderson 2007/2009, Evensen 1994, Bishop et al. 2001, Bernacchia et al. 2018, Gupta–Koren–Singer 2018, Tang et al. CVPR 2021, Furrer & Bengtsson 2007, Nesterov & Polyak 2006, Cartis–Gould–Toint, Bekas–Kokiopoulou–Saad 2007, Gebremedhin et al. 2005, Spall 2005, Amari 1998) ont été confirmées via le dépôt institutionnel, la page de l'éditeur ou les actes de la conférence ; les réserves de pagination sont listées en §6.2.