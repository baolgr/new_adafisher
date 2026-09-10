# Plan — Banc de référence Fisher : écart des approximations à la Fisher vraie et à la Fisher empirique exacte, par type de couche

*Plan d'implémentation pour `fisher_drift_analysis` — septembre 2026. À adapter au format `docs/reports/plan.md` du repo.*

Conventions de statut (mêmes que `rapport_fisher_gpu_2026.md`) : `[ÉTABLI]` = vérifié sur la source primaire ; `[DÉRIVÉ]` = arithmétique faite ici à partir de chiffres publiés ou d'architectures connues ; `[ESTIMATION]` = ordre de grandeur non mesuré, à confirmer au Lot 1.

---

## 0. Réponse courte : sur quels modèles tourner

**Matériel visé.** Le « 1/4 de H100 » des grappes de l'Alliance correspond à l'instance MIG `nvidia_h100_80gb_hbm3_2g.20gb` : **20 Go** de mémoire, environ 2/8 du GPU `[ÉTABLI, doc Alliance/WestDRI]`. L'A100 de Narval a 40 Go. Tous les chiffres ci-dessous sont donnés pour **20 Go**, avec l'équivalent 40 Go entre parenthèses.

**Le critère de faisabilité n'est pas « le modèle est petit ».** La Fisher vraie sur un ensemble de $N$ entrées s'écrit $F=U^\top U$, avec $U\in\mathbb R^{m\times P}$, $m=N\,r$, où $r$ est le rang du facteur de sortie ($r=C-1$ en softmax à $C$ classes, $r=1$ en binaire, $r=T(V-1)$ en modèle de langue). Trois régimes de calcul en découlent (§2) :

| Régime | Condition | Ce qu'on obtient |
|---|---|---|
| **A — dense** | $P\lesssim 2{,}9\cdot10^4$ en fp64 (≈ $4{,}1\cdot10^4$ sur 40 Go) | $F$, $\hat E$ et toutes les approximations formées explicitement ; spectres complets, blocs inter-couches, tout est exact |
| **B — facteur / dual** | $m\cdot P_\ell\cdot 4\,\text{o}$ tient couche par couche ($\lesssim 10$ Go) | $F$ exacte sur $N$ entrées, représentée par ses facteurs ; toutes les métriques par identités de Woodbury, sans jamais former $P\times P$ |
| **C — sans matrice** | un forward + backward tient en mémoire | produits $Fv$ exacts (Schraudolph) ; métriques par Lanczos / Hutch++ / gradient conjugué, donc avec une erreur stochastique ou d'itération contrôlée |

**Modèles retenus** (chiffres de paramètres `[DÉRIVÉ]`, script en annexe B) :

| Id | Modèle | $P$ | Régime | Types de couches couverts | Pourquoi lui |
|---|---|---|---|---|---|
| A1 | MLP 784–32–32–10 + LayerNorm (MNIST) | 26 634 | A | Linear sans partage, LN | cas où la théorie K-FAC est la plus propre ; témoin |
| A2 | CNN 3 conv (16-32-64) + GroupNorm / BN-eval (CIFAR-10) | 24 458 | A | Conv (partage spatial), normalisation | reproduit le banc « jouet » d'AdaFisher (App. B.2) mais avec la vraie Fisher exacte |
| A3 | ViT-micro, $d=32$, 2 blocs, pooling moyen (CIFAR-10) | 21 162 | A | patch-embed, QKV, sortie d'attention, MLP, LN, tête | cadre *reduce* propre (Eschenhagen et al.) |
| A4 | GPT-micro caractères, $d=32$, 2 blocs, $T=64$, $V=65$ | 23 360 | A | embeddings, attention causale, tête LM | cadre *expand* ; $m$ énorme mais $P$ petit, donc dense faisable |
| B1 | **CCT-2/3×2** (CIFAR-10) | 283 723 | B, $N\approx 900$ (1 800) | conv tokenizer, transformeur, LN, sequence pooling | **modèle d'AdaFisher lui-même** (Tableau 2 du papier) |
| B2 | ResNet-20 BN (CIFAR-10) | 269 722 | B, $N\approx 900$ (1 800) | Conv, BN | petit ResNet de la famille CNN d'AdaFisher |
| B3 | RoBERTa-base + LoRA $r=8$ (q, v) + tête, SST-2 / MNLI | 294 912 adaptateurs + 592 130 tête | B, $N\approx 2\,800$ ($U$ entier) à 4 000 (en flux) | LoRA-A, LoRA-B, tête dense | régime PEFT, celui d'iEF et de FisherAdapTune |
| B4 | ViT-Tiny/4 (CIFAR-10), fine-tuning complet | 5 362 762 | B par couche, $N\approx 1\,000$ | transformeur à largeur réaliste ($d=192$) | vérifie que les conclusions de A3/B1 tiennent à largeur réelle |
| B5 *(option)* | ResNet-18 CIFAR | 11 173 962 | B mixte, $N=256$ | Conv profondes | modèle principal d'AdaFisher ; demande l'astuce « ghost » (§2.3) |
| C1 *(option)* | GPT-1 réduit à 4 blocs (AdaFisher, WikiText-2) | 28 351 488 appris | C | LM à grand vocabulaire | seule façon de tester le cadre LM d'AdaFisher |
| C2 *(option)* | SegFormer-B0 basse résolution, BCE seule | ≈ 3,8 M `[à vérifier]` | B par couche ($N\approx 8$ images) ou C | pont vers FisherAdapTune | la segmentation de FisherAdapTune (SegFormer-B3, SAM2) n'est accessible qu'en C |

Le cœur du plan est **A1–A4 + B1–B4**. A donne la vérité complète sur des modèles jouets ; B la donne sur des modèles que les papiers utilisent réellement (B1) ou sur le régime qui nous intéresse (B3), au prix d'un rang limité à $m$. C ne sert qu'à relier aux configurations des papiers.

**Écartés** : ResNet-50/101, DenseNet-121, Tiny Swin, FocalNet en fine-tuning complet (régime C uniquement, rien de plus que C1 à un coût bien supérieur) ; SAM2-Large (idem).

---

## 1. Objet, questions, hypothèses falsifiables

### 1.1 Ce qu'on mesure

Deux références exactes sur un même ensemble de sondes $\mathcal D_N=\{x_n\}_{n\le N}$ (fixé, sans augmentation, versionné) et au même $\theta$ :

- **Fisher vraie** $F=\frac1N\sum_n J_n^\top\Lambda_n J_n$, où $J_n=\partial z_n/\partial\theta$ et $\Lambda_n=\mathrm{Cov}_{y\sim r(\cdot\mid z_n)}[\nabla_z\log r(y\mid z_n)]$ est calculée **analytiquement** (pas d'échantillonnage). Pour softmax, $\Lambda_n=\mathrm{diag}(p_n)-p_np_n^\top$.
- **Fisher empirique exacte** $\hat E=\frac1N\sum_n \nabla_\theta\ell_n\nabla_\theta\ell_n^\top$, avec les vrais labels, sans aucune structure imposée.

Et un zoo d'approximations (§4), chacune définie par une **structure** $S$ (pleine, bloc-diagonale, K-FAC expand/reduce, EKFAC, TKFAC, diagonale-Kronecker d'AdaFisher, diagonale, Hadamard pour les normalisations…) appliquée à une **source** $\sigma$ (vecteurs rétropropagés « type-2 » exacts, Monte-Carlo à $K$ tirages, empiriques).

Remarque de définition, importante pour la rigueur. $F=J^\top\Lambda J$ est exacte dès que la loi prédictive a une Fisher en $z$ de forme close (catégorielle, Bernoulli, gaussienne), **indépendamment** de l'égalité $F=\text{GGN}$. Cette égalité, elle, exige un lien canonique (Martens, arXiv:1412.1193, §9) : vraie pour softmax-CE sur logits, BCE sur logits, MSE. Elle est **fausse pour la perte Dice** de FisherAdapTune (errata H5 : BCE + Dice), qui n'est pas une log-vraisemblance : il n'existe pas de « Fisher vraie » de la Dice. Pour C2, la référence est donc la Fisher de la seule vraisemblance BCE, et on le dit.

### 1.2 Questions

- **Q1 (source).** Quelle part de l'écart à $F$ vient du remplacement des labels du modèle par les labels des données ($\hat E$ contre $F$), à structure fixée ?
- **Q2 (structure).** Quelle part vient de la structure imposée, mesurée à source fixée : $S(\text{type-2})$ contre $F$, et $S(\text{emp})$ contre $\hat E$ ?
- **Q3 (interaction).** Les deux erreurs s'additionnent-elles ? Le chemin emprunté par AdaFisher, $S_{\rm AF}(\text{emp})$ comparé à $F$, est-il pire que la somme de ses parties ?
- **Q4 (type de couche).** Comment Q1–Q3 varient-elles entre Linear sans partage, Linear à poids partagés (attention, MLP de séquence), Conv, normalisation, embedding, LoRA, tête ?
- **Q5 (opérationnel).** Que devient le classement quand on compare non pas les estimateurs « idéaux » (mêmes données, même $\theta$) mais **l'état réel de l'optimiseur** (EMA $\gamma=0{,}8$, min-max, amortissement, mini-batch de 256, BN en mode train) ?

### 1.3 Hypothèses falsifiables (à écrire dans le repo avant de lancer les runs)

- **HF1** — Sur des sondes tirées du *train*, l'écart $\hat E$ contre $F$, mesuré par l'efficacité quadratique $\rho$ (§5), se dégrade au fil de l'entraînement ; sur des sondes tirées du *val*, beaucoup moins. *Réfutée si* les deux courbes coïncident à l'intervalle de bruit près (§3.4).
- **HF2** — L'erreur de structure de K-FAC (source type-2) est plus grande sur les couches à poids partagés (QKV, conv) que sur les Linear sans partage, et **K-FAC-reduce bat K-FAC-expand** en classification (cadre *reduce*), l'inverse en LM. *Réfutée si* l'écart expand/reduce est sous le bruit.
- **HF3** — Sur la diagonale seule, l'hypothèse d'indépendance d'AdaFisher, $\mathbb E[a_j^2g_i^2]\approx\mathbb E[a_j^2]\,\mathbb E[g_i^2]$, a une erreur relative non négligeable (≥ 10 %) sur les normalisations et l'attention. *Réfutée si* $\mathrm{diag}(A)\otimes\mathrm{diag}(G)$ et $\mathrm{diag}(B_\ell)$ coïncident à ±5 %.
- **HF4** — Dans les normalisations, le couplage $\gamma$–$\beta$ que la Prop. 3.1 d'AdaFisher exclut porte une fraction d'énergie non négligeable du bloc exact $2C\times2C$.
- **HF5** — L'information inter-couches rapporte peu (Abreu et al., arXiv:2510.09378) : $\rho(\mathrm{BD}(F))\ge0{,}9$ sur CNN **et** transformeur ($\rho$ vaut 1 pour $K=F$). *Réfutée si* l'écart est grand sur le transformeur (couplage Q–K).
- **HF6** — Avec $K=1$ et un petit batch, K-FAC-MC est moins bon que K-FAC-emp en $\rho$ (cohérent avec le constat d'iEF, arXiv:2406.06420).
- **HF7** — Le classement des approximations en protocole opérationnel P2 n'est pas celui du protocole structurel P1 : l'EMA et la normalisation min-max, pas la structure, dominent l'erreur réelle d'AdaFisher.
- **HF8** — À petit $N$, une partie des différences entre approximations est **sous le plancher de bruit** de $F$ elle-même. C'est un résultat en soi : il borne ce qu'on peut affirmer.

Chaque hypothèse est reliée à une décision de conception (§11) : c'est ce qui rattache le banc à l'objectif « faiblesses → nouvelle architecture ».

---

## 2. Faisabilité : les trois régimes de calcul

### 2.1 Notation et facteur de sortie

- $U\in\mathbb R^{m\times P}$ empile, pour chaque sonde $n$ et chaque colonne $c$ d'une racine $\Lambda_n^{1/2}$, la ligne $N^{-1/2}\,(\Lambda_n^{1/2}e_c)^\top J_n$. Alors $F=U^\top U$ exactement.
- Racine fermée pour softmax, sans décomposition : $S_n[:,c]=\sqrt{p_c}\,(e_c-p_n)$ vérifie $S_nS_n^\top=\mathrm{diag}(p_n)-p_np_n^\top$ (même choix que Dangel et al., *KFAC from scratch*, arXiv:2507.05127, fiche récapitulative §6). Elle donne $C$ colonnes pour un rang $C-1$ : +11 % de lignes à $C=10$. Pour le binaire, une seule colonne $\sqrt{p(1-p)}$.
- Chaque ligne de $U$ coûte **une** rétropropagation d'un vecteur de sortie, et les $N$ sondes d'un même batch partagent la même passe arrière (échantillons indépendants : BN en mode eval, dropout coupé). Construire $U$ coûte donc $C$ rétropropagations par batch de sondes.
- $\hat E$ a le même format avec $m=N$ lignes $N^{-1/2}\nabla_\theta\ell_n$.

### 2.2 Régime A — dense

$F$ formée par SYRK $U^\top U$ en fp64. Mémoire : $8P^2$ octets, et environ $3\times$ avec `eigh` (matrice, vecteurs propres, espace de travail). D'où $P_{\max}\approx\sqrt{B/24}$ : **28 867 sur 20 Go**, 40 824 sur 40 Go, 57 735 sur 80 Go `[DÉRIVÉ]`. Le fp64 n'est pas un luxe : les spectres de Fisher s'étalent sur plus de $10^8$, et en fp32 les valeurs propres sous $\sim10^{-7}\lambda_{\max}$ ne sont pas fiables.

Choisir $N$ tel que $N(C-1)\ge P$ ($N=4\,000$ pour A1–A3, soit $m=36\,000$) : $F$ peut alors être de rang plein. Ce point est décisif, car en dessous toutes les métriques qui passent par une inverse sont dominées par l'amortissement dans le noyau de $F$ (voir §3.3). Coût de la SYRK : $mP^2\approx 2\cdot10^{13}$ flop fp64, soit quelques secondes `[DÉRIVÉ]`.

A4 (LM) est le cas instructif : $m=N\,T(V-1)=4\,096$ lignes par séquence, donc le dual est inutilisable, mais $P=23\,360$ rend le primal dense trivial. Deux options, à comparer entre elles comme test : SYRK sur $256\times4\,096$ lignes (≈ $5{,}7\cdot10^{14}$ flop fp64, moins d'une minute `[ESTIMATION]`), ou $P$ produits $Fe_j$ par `GGNLinearOperator`.

### 2.3 Régime B — facteur / dual, en flux par couche

On ne forme jamais $U$ en entier. Pour chaque couche $\ell$ on garde deux choses :

1. le **Gram par couche** $K_\ell=U_\ell U_\ell^\top\in\mathbb R^{m\times m}$ (le NTK de la couche, pondéré par $\Lambda$). On a alors $\|F\|_F^2=\|\sum_\ell K_\ell\|_F^2$, $\|F_{\ell\ell'}\|_F^2=\langle K_\ell,K_{\ell'}\rangle_F$, et le spectre non nul de $F$ est celui de $\sum_\ell K_\ell$ (principe de ViViT, Dangel, Tatzel & Hennig, arXiv:2106.02624 `[ÉTABLI]`) ;
2. les **statistiques de couche** $(a_{n,t},\,g_{n,c,t})$ capturées par hooks (entrées de couche et gradients de sortie rétropropagés, pour chaque colonne $c$). Toutes les approximations structurées (K-FAC, EKFAC, TKFAC, AdaFisher, diagonale) se construisent à partir d'elles, et on sait appliquer une approximation $K_\ell$ aux lignes de $U_\ell$ sans matérialiser $U_\ell$ quand c'est nécessaire.

Deux façons de calculer $K_\ell$ pour une couche linéaire à poids partagés (ligne $(n,c)$ : $\sum_t g_{n,c,t}a_{n,t}^\top$) :

- **matérialiser** $U_\ell$ (mémoire $4mP_\ell$) puis faire la SYRK ;
- **ghost** (sans gradients par échantillon) : $K_\ell[(n,c),(n',c')]=\sum_{t,s}(a_{n,t}\!\cdot\! a_{n',s})(g_{n,c,t}\!\cdot\! g_{n',c',s})$, de coût $\approx 2(mT)^2(d_{\rm in}+d_{\rm out})$. C'est la « ghost norm » de la confidentialité différentielle (Li et al., arXiv:2110.05679), et on choisit couche par couche entre les deux voies comme dans le *mixed ghost clipping* (Bu, Mao & Xu, arXiv:2205.10683) `[ÉTABLI]`.

Chiffres `[DÉRIVÉ]` à $m=N(C-1)$ :

| Modèle | $N$ (20 Go) | $m$ | $U$ entier fp32 | $U_\ell$ max | Gram fp64 | $\hat E$ entier |
|---|---|---|---|---|---|---|
| CCT-2/3×2 | 1 000 | 9 000 | 10,2 Go | 2,7 Go | 0,65 Go | 1,1 Go |
| ResNet-20 | 1 000 | 9 000 | 9,7 Go | 1,3 Go | 0,65 Go | 1,1 Go |
| ViT-Tiny/4 | 1 000 | 9 000 | 193 Go (**en flux**) | 5,3 Go | 0,65 Go | 21 Go (en flux) |
| RoBERTa LoRA + tête ($C=2$) | 4 096 | 4 096 | 14,5 Go | 9,7 Go (tête) | 0,13 Go | 14,5 Go |
| ResNet-18 (option) | 256 | 2 304 | 103 Go | 21,7 Go → **ghost** | 0,04 Go | 11,4 Go |

Ces chiffres supposent une racine de $\Lambda$ tronquée au rang $C-1$. Avec la racine fermée à $C$ colonnes, compter +11 % de lignes, ou prendre $N=900$ au lieu de 1 000 (valeurs du §0).

Pour ResNet-18 à $N=256$ : matérialiser `layer1`–`layer3` (0,3 / 1,4 / 5,4 Go) et passer `layer4` en ghost ($T=16$ positions, ≈ $1{,}4\cdot10^{13}$ flop) `[DÉRIVÉ]`. En ghost, `layer1` coûterait $7\cdot10^{15}$ flop : c'est bien un choix couche par couche.

**Prix du régime B : le rang.** $F$ a un rang au plus $m\ll P$. Les métriques de Frobenius et de direction restent exactes ; les métriques qui passent par une inverse amortie (KL, conditionnement) mélangent deux effets : l'erreur dans l'image de $F$, et la courbure **inventée** par une approximation de rang plein (K-FAC) dans le noyau de $F$, où $F_\lambda=\lambda I$. Ce second effet est réel (c'est le « Kronecker invente du rang » du rapport, §3.3.1), mais il doit être lu avec un balayage en $\lambda$ et confronté au régime A, où $N(C-1)\ge P$ le supprime.

### 2.4 Régime C — sans matrice

`GGNLinearOperator` / `EFLinearOperator` de `curvlinops` sur les sondes : $Fv$ exact en 1 JVP + 1 VJP, **quel que soit $V$** (le $\Lambda$ softmax s'applique analytiquement). Métriques accessibles : norme spectrale de $F-cK$ (Lanczos), traces (Hutch++/XTrace, déjà dans `curvlinops`), $\rho(K)$ via gradient conjugué pour $F_\lambda^{-1}g$, spectres (SLQ). Non accessible proprement : le KL (log-déterminant par SLQ possible, mais stochastique ; optionnel). Pour C1, le K-FAC type-2 est impossible ($V=40\,478$ rétropropagations) : seul K-FAC-MC est comparé, et c'est une limite à écrire.

### 2.5 Pièges matériels et numériques (à régler au Lot 0)

- **TF32 désactivé** : `torch.backends.cuda.matmul.allow_tf32 = False` **et** `torch.backends.cudnn.allow_tf32 = False`. Le second est `True` par défaut : sur A100/H100 les convolutions tournent alors en TF32, avec une erreur relative de l'ordre de $10^{-3}$ sur les gradients par échantillon. Cette erreur suffit à fausser les tests d'exactitude.
- **MIG** : un seul processus, pas de NCCL. Prévoir de la RAM hôte (≥ 64 Go dans la requête Slurm) pour déporter $U$ ou les statistiques de couche du régime B.
- **BN** : en mode *train*, la sortie d'un échantillon dépend du batch, donc gradient par échantillon, $\hat E$ et $F$ (somme sur les échantillons) ne sont pas définis. La référence se calcule en mode **eval** (statistiques courantes). L'écart entre facteurs calculés en mode train (ce que voit AdaFisher) et en mode eval est mesuré à part (§3.2, P2). Variante GroupNorm pour A2 afin d'avoir un objet propre sans cette question.
- **Dropout / stochastic depth** coupés pour toute référence. AdaFisher estime ses facteurs avec dropout actif : autre source d'écart en P2.
- **LoRA à l'initialisation** : $B=0$ entraîne $\nabla_A\ell=0$, donc $F_{AA}=0$. Point de contrôle $t=0$ dégénéré, à exclure ou à signaler.
- **Convention LM** : Fisher par token, **conditionnée aux préfixes des données** (teacher forcing) : $F_{\rm LM}=\frac1{NT}\sum_{n,t}J_{n,t}^\top\Lambda_{n,t}J_{n,t}$. Ce n'est pas la Fisher de la loi jointe autorégressive, qui échantillonnerait aussi les préfixes. De même, deux $\hat E$ existent (par séquence ou par token) : les deux sont calculés et étiquetés.
- **Définition des blocs** : `timm` fusionne Q, K, V en un seul `Linear` (un bloc K-FAC à entrée partagée), HuggingFace les sépare (trois blocs). A3 est implémenté dans les deux variantes, parce que « l'erreur de K-FAC sur l'attention » dépend de ce choix.
- **Vectorisation** : fixer `rvec` (convention PyTorch) partout, et vérifier que $A\otimes B$ contre $B\otimes A$ est cohérent avec `curvlinops` par le Test 1 de *KFAC from scratch*.

---

## 3. Plan d'expérience

### 3.1 Design factoriel : structure × source × référence

Pour chaque couche $\ell$ et chaque point de contrôle, on calcule l'erreur $d(S(\sigma),R)$ pour :

- $S\in\{$pleine, bloc-diag., K-FAC-expand, K-FAC-reduce, EKFAC, TKFAC, AF-diag-Kronecker, diagonale exacte, (normalisations : Hadamard, Prop. 3.1 telle qu'implémentée)$\}$ ;
- $\sigma\in\{$type-2, MC$_{K}$ avec $K\in\{1,4,16\}$, emp$\}$ ;
- $R\in\{F,\hat E\}$.

Les cellules qui répondent directement aux questions :

| Cellule | Isole | Question |
|---|---|---|
| $d(\hat E,F)$ | erreur de **source** seule (aucune structure) | Q1 |
| $d(S(\text{type-2}),F)$ | erreur de **structure** seule, sur la vraie Fisher | Q2 |
| $d(S(\text{emp}),\hat E)$ | erreur de **structure** seule, sur la Fisher empirique — le second objet de la demande | Q2 |
| $d(S(\text{emp}),F)$ | chemin réel d'AdaFisher / FisherAdapTune | Q3 |
| $d(S(\text{MC}_K),F)$ | structure + bruit Monte-Carlo | HF6 |
| $d(S(\sigma),F)$ par type de couche | tout ce qui précède, désagrégé | Q4 |

**Décomposition de l'erreur (Q3).** On rapporte, pour chaque $S$ :
$$\underbrace{d(S(\text{emp}),F)}_{\text{total}}\quad\text{contre}\quad\underbrace{d(\hat E,F)}_{\text{source}}+\underbrace{d(S(\text{emp}),\hat E)}_{\text{structure sur }\hat E}\quad\text{et}\quad d(\hat E,F)+\underbrace{d(S(\text{type-2}),F)}_{\text{structure sur }F}.$$
Pour une vraie distance (Frobenius, spectrale), l'inégalité triangulaire borne le total par la première somme. L'écart entre le total et cette borne mesure si les deux erreurs **se compensent** (total ≪ somme) ou **s'alignent** (total ≈ somme). Pour le KL, qui n'est pas une distance, on rapporte seulement les trois termes.

**Deux chemins pour la diagonale d'AdaFisher (HF3).** $\mathrm{diag}(A)\otimes\mathrm{diag}(G)=\mathrm{diag}(A\otimes G)$ : l'estimateur brut d'AdaFisher (et de FisherAdapTune, Éq. 9) **est la diagonale de K-FAC**. Il y a donc deux chemins de $B_\ell$ vers lui : $B_\ell\to A\otimes G\to\mathrm{diag}$ et $B_\ell\to\mathrm{diag}(B_\ell)\to$ factorisation. L'écart $\mathrm{diag}(B_\ell)$ contre $\mathrm{diag}(A)\otimes\mathrm{diag}(G)$ vaut entrée par entrée $\mathbb E[a_j^2g_i^2]-\mathbb E[a_j^2]\mathbb E[g_i^2]=\mathrm{Cov}(a_j^2,g_i^2)$ (sans partage de poids) : c'est le biais d'indépendance **sur la diagonale seule**, distinct de la perte des termes hors diagonale.

**Décomposition propre aux couches à poids partagés** (régime A, et régime B quand $mT$ reste raisonnable) :
$$B_\ell\;\xrightarrow{\text{termes croisés }t\neq t'\text{ supprimés}}\;B_\ell^{\rm exp}=\tfrac1N\textstyle\sum_{n,c,t}(a_{n,t}a_{n,t}^\top)\otimes(g_{n,c,t}g_{n,c,t}^\top)\;\xrightarrow{\text{indépendance}}\;\text{K-FAC-expand}.$$
La première flèche est l'erreur de **partage de poids**, la seconde l'erreur d'**indépendance**. K-FAC-reduce prend un autre chemin, comparé directement à $B_\ell$.

### 3.2 Deux protocoles

- **P1 — structurel.** Même $\theta$, mêmes sondes $\mathcal D_N$, pas d'EMA, pas de min-max, amortissement balayé. Répond à « quelle est la meilleure approximation *possible* dans chaque famille ».
- **P2 — opérationnel.** On lance l'optimiseur réel (AdaFisher du dépôt amont, **code inchangé**) et, à chaque point de contrôle, on extrait son état interne tel qu'il sert au pas : $H_D$, $S_D$ après EMA ($\gamma=0{,}8$), min-max, $\lambda=10^{-3}$, BN en mode train, dropout actif, mini-batch de 256. On le compare aux références exactes au **même** $\theta$. Mêmes mesures pour K-FAC (ASDL) si on le fait tourner. La min-max rend le préconditionneur affine et non linéaire (objection déjà documentée), donc en P2 on privilégie les métriques invariantes d'échelle ($\cos_F$, $\rho$, conditionnement à échelle optimale).

HF7 se teste en comparant les classements P1 et P2 (corrélation de rang de Kendall, par type de couche).

### 3.3 Amortissement et échelle

Toute métrique qui passe par une inverse dépend de $\lambda$. On ne fixe pas $\lambda$ : on balaie $\lambda=\alpha\,\bar\lambda$, avec $\bar\lambda=\mathrm{tr}(R)/P$ et $\alpha\in\{10^{-4},10^{-3},10^{-2},10^{-1},1\}$, et on trace les courbes. On ajoute le point $\lambda=10^{-3}$ d'AdaFisher, mais seulement en P2, puisqu'il s'applique à des facteurs normalisés en $[0,1]$. Pour les approximations qui ne prétendent pas respecter l'échelle (AdaFisher, diagonale « gratuite » d'Adam), on rapporte aussi la version à **échelle optimale** $c^\star=\langle R,K\rangle_F/\|K\|_F^2$.

### 3.4 Plancher de bruit et partition des sondes

- **Bruit de $F$ elle-même.** $\mathcal D_N$ est coupé en deux moitiés $\mathcal D^{(1)},\mathcal D^{(2)}$ (en régime B, ce ne sont que deux sous-ensembles de lignes de $U$ : coût nul). On calcule $d(F^{(1)},F^{(2)})$ sur 20 partitions aléatoires, ce qui donne un intervalle à 95 %. **Toute différence entre approximations plus petite que ce plancher n'est pas interprétable** (HF8). On trace aussi $d(F_{N'},F_N)$ en fonction de $N'<N$.
- **Sondes train contre val.** Deux ensembles de sondes, tirés du train et du val. $\hat E$ et $F$ se comportent différemment sur des points déjà appris (gradients empiriques qui s'annulent) et sur des points non vus (HF1).
- **Graines** : 3 graines d'entraînement par modèle (régime A : 5), la même partition de sondes pour toutes.

### 3.5 Points de contrôle

- Trajectoires : AdamW (référence neutre) et AdaFisher (trajectoire propre), mêmes hyperparamètres que le dépôt amont quand ils existent.
- $t\in\{0,\;1\%,\;10\%,\;50\%,\;100\%\}$ des pas. Pour LoRA, $t=0$ est dégénéré (§2.5) : le premier point est pris après 50 pas.
- Pour B3 (fine-tuning), $t=0$ = modèle pré-entraîné + tête aléatoire, point de départ de FisherAdapTune.

---

## 4. Le zoo d'approximations, par type de couche

Notation commune. $a_{n,t}$ : entrée de la couche à la position partagée $t$ (token, position spatiale), avec un 1 ajouté pour le biais. $g_{n,c,t}$ : gradient rétropropagé en sortie de couche (pré-activation) pour la colonne $c$ de la source $\sigma$ :

- type-2 : $c=1..C$, vecteur de sortie $S_n[:,c]$ ;
- MC$_K$ : $c=1..K$, vecteur $K^{-1/2}(p_n-e_{\tilde y_{n,c}})$ avec $\tilde y\sim p_n$. Le facteur $K^{-1/2}$ est celui qui manquait dans *KFAC from scratch* (errata, erreur 3) ; sans lui, K-FAC-MC ne converge pas vers la GGN ;
- emp : $c=1$, vecteur $p_n-e_{y_n}$.

Gradient par échantillon (et par colonne) de la couche : $\mathcal G_{n,c}=\sum_t g_{n,c,t}a_{n,t}^\top$. Bloc exact : $B^\sigma_\ell=\frac1N\sum_{n,c}\mathrm{vec}(\mathcal G_{n,c})\mathrm{vec}(\mathcal G_{n,c})^\top$ (donc $B^{\text{type-2}}_\ell=F_{\ell\ell}$ et $B^{\text{emp}}_\ell=\hat E_{\ell\ell}$).

| Structure | Définition | Couches | Remarque |
|---|---|---|---|
| Pleine | $F$ ou $\hat E$ | tout | référence |
| Bloc-diagonale exacte | $\mathrm{blkdiag}_\ell(B_\ell)$ | tout | isole l'information inter-couches (HF5) ; deux découpages : par `nn.Module` et par tenseur (W / b, Q / K / V) |
| K-FAC-expand | $A=\frac1{NT}\sum_{n,t}aa^\top$, $G=\frac1N\sum_{n,c,t}gg^\top$, $K=A\otimes G$ (`rvec` : $G\otimes A$) | Linear, Conv, embedding | exact en cadre *expand* pour un réseau linéaire profond (Eschenhagen et al., arXiv:2311.00636, Prop. 1) `[ÉTABLI]` |
| K-FAC-reduce | facteurs formés sur $\sum_t a_{n,t}$ et $\sum_t g_{n,c,t}$ (normalisation de l'article) | Linear partagé, Conv | exact en cadre *reduce* à agrégation par somme pondérée, par exemple le pooling moyen (Prop. 2) `[ÉTABLI]` ; **constantes de normalisation reprises de `KFACLinearOperator`, pas re-dérivées** |
| EKFAC | base propre $Q_A\otimes Q_G$ de K-FAC, valeurs propres $s_{ij}=\frac1N\sum_{n,c}[(Q_G^\top\mathcal G_{n,c}Q_A)_{ij}]^2$ | Linear, Conv | conserve $\mathrm{tr}(B_\ell)$ par construction (test V9) |
| TKFAC | $K=\delta\,\Phi\otimes\Psi$, $\Phi=\frac{\mathbb E[\mathrm{tr}(\Gamma)\,\Lambda]}{\mathbb E[\mathrm{tr}\Lambda\,\mathrm{tr}\Gamma]}$, $\Psi=\frac{\mathbb E[\mathrm{tr}(\Lambda)\,\Gamma]}{\mathbb E[\mathrm{tr}\Lambda\,\mathrm{tr}\Gamma]}$, $\delta=\frac{\mathbb E[\mathrm{tr}\Lambda\,\mathrm{tr}\Gamma]}{\mathrm{tr}\Phi\,\mathrm{tr}\Psi}$, avec $\Lambda=aa^\top$, $\Gamma=gg^\top$ | Linear, Conv | Gao et al., arXiv:2011.10741, Éq. 4.3–4.9 `[ÉTABLI]` ; la trace est conservée **sans partage** ; pour les couches partagées, adaptation *expand* à signaler comme choix |
| AF-brut | $\mathrm{diag}(A)\otimes\mathrm{diag}(G)$, normalisation $/|T|$ pour les conv (App. A.3) | Linear, Conv | = diagonale de K-FAC ; = $\tilde F_D$ de FisherAdapTune (Éq. 9) |
| AF-op | $H'_D\otimes S'_D+\lambda I$ (Éq. 4), avec min-max et EMA | tout | **uniquement via l'adaptateur du code amont** (§8.1) |
| Diagonale exacte | $\mathrm{diag}(B_\ell)$ | tout | témoin : ce qu'une méthode diagonale peut faire de mieux |
| Diagonale « gratuite » | second moment $v$ d'Adam, mis à l'échelle | tout (P2) | Li, Dangel, Tam & Raffel, arXiv:2507.18807 : le témoin à exiger de toute méthode diagonale |

**Normalisations** (LayerNorm, GroupNorm, BN en eval) : paramètres $\gamma,\beta\in\mathbb R^C$, entrée normalisée $\hat x_{n,t}$. Gradients par échantillon exacts : $\nabla_\gamma=\sum_t g_t\odot\hat x_t$ et $\nabla_\beta=\sum_t g_t$. Le bloc exact est $2C\times2C$ : il est **toujours** calculable exactement, même en régime C.

| Structure | Définition |
|---|---|
| Exacte jointe | bloc $2C\times2C$ sur $(\gamma,\beta)$ |
| Exacte séparée | blocs $\gamma$ et $\beta$ sans termes croisés (ce que fait AdaFisher : « cross-terms … excluded ») → mesure HF4 |
| Hadamard (Prop. 3.1 corrigée, errata H1) | $\gamma$ : $\big(\frac1{|T|}\sum_t\hat x\hat x^\top\big)\odot\big(\frac1{|T|}\sum_t gg^\top\big)$ à la normalisation près |
| Prop. 3.1 telle qu'implémentée | diagonale $[H_D]_c[S_D]_c$ lue dans le code amont |

**Point à vérifier dans le code amont avant toute conclusion.** La Prop. 3.1 parle d'activations « pre-normalized », alors que sa preuve écrit que $h_{i-1}$ « contains normalized activations ». Or le gradient de $\gamma$ fait intervenir $\hat x$ (normalisé). Un hook `forward` sur `nn.BatchNorm2d` / `nn.LayerNorm` capture l'entrée **avant** normalisation. Si le code utilise cette entrée telle quelle, l'estimateur des normalisations porte sur la mauvaise variable. Le banc le révèle directement : Prop. 3.1 implémentée contre diagonale exacte. **Ne rien affirmer avant d'avoir lu le code.**

**Embeddings** : entrée one-hot, donc $A$ diagonal (fréquences des tokens) ; le bloc exact n'a de lignes non nulles que pour les tokens présents dans les sondes.

**LoRA** ($W_0+BA$) : adaptateur $A$ (entrée $a$, gradient de sortie $B^\top g$), adaptateur $B$ (entrée $Aa$, gradient de sortie $g$). K-FAC par adaptateur ; blocs exacts ; **couplage A–B** mesuré par M8 (§5), puisque le produit $BA$ est bilinéaire et que la bloc-diagonalité le coupe exactement là où il y a le plus de couplage.

**Tête** : Linear sans partage, mais $\Lambda(p)$ dépend de $a$ : l'indépendance y est fausse par construction. C'est un point de mesure utile.

---

## 5. Métriques

$R\in\{F,\hat E\}$ est la référence, $K$ l'approximation, $X_\lambda=X+\lambda I$. Chaque métrique est donnée avec ce qu'elle mesure et la façon de la calculer dans chaque régime.

| Id | Métrique | Ce qu'elle mesure | A | B | C |
|---|---|---|---|---|---|
| M1 | $e_F=\frac{\Vert R-K\Vert _F}{\Vert R\Vert _F}$ ; $\cos_F=\frac{\langle R,K\rangle_F}{\Vert R\Vert _F\Vert K\Vert _F}$ ; $e_F^\star=\sqrt{1-\cos_F^2}$ (échelle optimale) | fidélité entrée par entrée, dominée par les grandes valeurs propres | dense | $\Vert R\Vert _F^2=\Vert UU^\top\Vert _F^2$, $\langle R,K\rangle=\mathrm{tr}(UKU^\top)$, $\Vert K\Vert _F^2$ fermé (Kronecker : $\Vert A\Vert _F^2\Vert G\Vert _F^2$) | Hutchinson sur $\Vert R-K\Vert _F^2$ (stochastique) |
| M2 | $e_2=\Vert R-c^\star K\Vert _2/\Vert R\Vert _2$ | pire direction | `eigvalsh` | Lanczos sur $v\mapsto U^\top Uv-c^\star Kv$ | Lanczos |
| M3 | $D_\lambda(K\Vert R)=\tfrac12[\mathrm{tr}(K_\lambda R_\lambda^{-1})-P-\log\det(K_\lambda R_\lambda^{-1})]$ = KL$\big(\mathcal N(0,R_\lambda^{-1})\Vert\mathcal N(0,K_\lambda^{-1})\big)$, et le sens inverse | écart **invariant affine** entre préconditionneurs (perte de Stein) ; le critère naturel pour un préconditionneur | Cholesky | Woodbury exact (encadré ci-dessous) | SLQ (option) |
| M4 | $\kappa_\lambda=\lambda_{\max}/\lambda_{\min}$ de $K_\lambda^{-1/2}R_\lambda K_\lambda^{-1/2}$ | nombre d'itérations d'un gradient conjugué préconditionné, $\propto\sqrt{\kappa}$ | `eigh` généralisé | Lanczos sur $K_\lambda^{-1}R_\lambda$ | Lanczos |
| M5 | $\rho(K)=\frac{(g^\top d)^2}{(d^\top R_\lambda d)(g^\top R_\lambda^{-1}g)}$, $d=K_\lambda^{-1}g$ ; plus $\cos(d,R_\lambda^{-1}g)$ et $\gamma$ d'iEF | fraction de la baisse quadratique optimale obtenue en suivant $d$ ; $\rho\in[0,1]$ par Cauchy-Schwarz dans le produit scalaire $R_\lambda$ | exact | Woodbury | CG |
| M6 | recouvrement des sous-espaces propres dominants, $\Vert V_R^\top V_K\Vert _F^2/k$, $k\in\{C,10C\}$ | les $O(C)$ directions aberrantes (Papyan) sont-elles captées ? | exact | $V_R=U^\top W\mathrm{diag}(\mu)^{-1/2}$ via le Gram | Lanczos |
| M7 | $\sigma_2/\sigma_1$ et $\sum_{i>1}\sigma_i^2/\sum\sigma_i^2$ du réarrangement $\mathcal R(B_\ell^{\rm exp})$ ; $\Vert B_\ell-B_\ell^{\rm exp}\Vert _F/\Vert B_\ell\Vert _F$ ; $\Vert \mathrm{diag}B_\ell-\mathrm{diag}A\otimes\mathrm{diag}G\Vert /\Vert \mathrm{diag}B_\ell\Vert $ | biais d'indépendance, part du partage de poids, biais sur la diagonale | exact | SVD randomisée à produits implicites (§2.B.1 du rapport) | — |
| M8 | $c_{\ell\ell'}=\frac{\langle K_\ell,K_{\ell'}\rangle_F}{\Vert K_\ell\Vert _F\Vert K_{\ell'}\Vert _F}=\frac{\Vert F_{\ell\ell'}\Vert _F^2}{\Vert F_{\ell\ell}\Vert _F\Vert F_{\ell'\ell'}\Vert _F}$ | couplage inter-blocs ; c'est une CKA non centrée entre noyaux tangents de couches | exact | Grams par couche | — |

Pour M7, le diagnostic du réarrangement a un précédent à citer : Koroko et al. (arXiv:2201.10285) approximent déjà les blocs de Fisher par SVD du produit de Kronecker (KP-SVD) sur des auto-encodeurs profonds `[ÉTABLI, résumé]`. La nouveauté du banc n'est donc pas l'outil, mais la carte par type de couche sur des architectures modernes, contre une référence exacte et au fil de l'entraînement. Le rapport `rapport_fisher_gpu_2026.md` (§0-e, §2.B.1) devrait citer ce travail.

### 5.1 Encadré — M3 et M5 exacts en régime B

Avec $R=U^\top U$ ($U\in\mathbb R^{m\times P}$), $\mathcal G=UU^\top$ et $M=\lambda I_m+\mathcal G$ :

- $\log\det R_\lambda=P\log\lambda+\log\det M-m\log\lambda$ (identité de Sylvester) ;
- $R_\lambda^{-1}=\lambda^{-1}\big(I-U^\top M^{-1}U\big)$ ;
- $\mathrm{tr}(K_\lambda R_\lambda^{-1})=\lambda^{-1}\big[\mathrm{tr}K+\lambda P-\mathrm{tr}\big(M^{-1}(UKU^\top+\lambda\mathcal G)\big)\big]$ ;
- $\mathrm{tr}(K_\lambda^{-1}R_\lambda)=\lambda\,\mathrm{tr}(K_\lambda^{-1})+\mathrm{tr}(UK_\lambda^{-1}U^\top)$ ;
- $\log\det K_\lambda$ : Kronecker, $\sum_{i,j}\log(\alpha_i\gamma_j+\lambda)$ ; diagonale, immédiat ; bloc-diagonale exacte, $\sum_\ell[P_\ell\log\lambda+\log\det(I+K_\ell/\lambda)]$.

Test d'autocohérence : pour $K=R$, $UKU^\top=\mathcal G^2$ et $\mathrm{tr}(M^{-1}(\mathcal G^2+\lambda\mathcal G))=\mathrm{tr}\,\mathcal G$, donc $\mathrm{tr}(K_\lambda R_\lambda^{-1})=P$ et $D_\lambda=0$.

```python
import math, torch

def stein_kl_lowrank(U: torch.Tensor, K, lam: float) -> torch.Tensor:
    """KL( N(0, R_lam^{-1}) || N(0, K_lam^{-1}) ) with R = U^T U, U: (m, P), fp64.
    K must provide: apply_rows(U) -> U @ K (row-wise K u_i), trace(), logdet(lam)."""
    m, P = U.shape
    G = U @ U.T                                              # m x m Gram
    M = G + lam * torch.eye(m, dtype=U.dtype, device=U.device)
    L = torch.linalg.cholesky(M)
    UKUt = U @ K.apply_rows(U).T                             # m x m  = U K U^T
    X = torch.cholesky_solve(UKUt + lam * G, L)
    tr_KR = (K.trace() + lam * P - X.diagonal().sum()) / lam # tr(K_lam R_lam^{-1})
    logdet_R = P * math.log(lam) + 2 * torch.log(L.diagonal()).sum() - m * math.log(lam)
    return 0.5 * (tr_KR - P - K.logdet(lam) + logdet_R)

def rho_efficiency(U, K, g, lam):
    """Fraction of the optimal quadratic decrease under R_lam reached along d = K_lam^{-1} g."""
    m = U.shape[0]
    M = U @ U.T + lam * torch.eye(m, dtype=U.dtype, device=U.device)
    Rinv_g = (g - U.T @ torch.linalg.solve(M, U @ g)) / lam  # Woodbury
    d = K.solve(g, lam)                                      # K_lam^{-1} g
    dRd = (U @ d).pow(2).sum() + lam * d.pow(2).sum()
    return (g @ d) ** 2 / (dRd * (g @ Rinv_g))
```

---

## 6. Positionnement par rapport à l'existant

- **AdaFisher, App. B.2** `[ÉTABLI, lu dans le papier]` : la seule validation de l'approximation se fait sur un modèle jouet (2 conv + 2 linéaires, sous-ensemble de MNIST). La « vraie Fisher » y est obtenue avec NNGeometry **par échantillonnage Monte-Carlo** de $p(y\mid x)$, et la seule mesure est l'erreur absolue moyenne (MAE) entre **diagonales** (Fig. 12). Le banc remplace chaque élément : $\Lambda$ analytique au lieu du Monte-Carlo, blocs et spectres au lieu de la diagonale seule, métriques invariantes d'échelle au lieu d'une MAE qui dépend de l'échelle, 8 types de couches, les modèles de l'article lui-même (CCT-2/3×2, famille ResNet), et un plancher de bruit.
- **Kunstner, Balles & Hennig** (arXiv:1905.12558) : écart $\hat E$ contre $F$ montré sur de petits problèmes, sans carte par couche.
- **Benzing** (arXiv:2201.12250) : mises à jour exactes par Woodbury, jugées au niveau de l'optimisation, pas de la fidélité par bloc.
- **Eschenhagen et al.** (arXiv:2311.00636) : exactitude d'expand/reduce sur réseaux linéaires profonds ; pas de mesure de fidélité contre la Fisher exacte de réseaux non linéaires.
- **Koroko et al.** (arXiv:2201.10285) : approximation de Kronecker optimale par KP-SVD, sur auto-encodeurs.
- **ViViT** (arXiv:2106.02624) : exploite la structure de rang faible de la GGN (outil repris en régime B).
- **Zhang et al., *Why Transformers Need Adam*** (arXiv:2402.16788) : hétérogénéité **du Hessien** entre blocs de paramètres, forte dans les transformeurs et faible dans les CNN. Cela motive la ventilation par type de couche, mais porte sur le Hessien et non sur les approximations de Fisher.
- **Ormaniec, Dangel & Singh** (arXiv:2410.10986) : Hessien d'une couche d'auto-attention, à dépendances très différentes entre Q/K et V. Cela justifie de séparer Q, K et V en blocs distincts (variante A3-séparée).

Lacune revendiquée, **à confirmer par une recherche ciblée avant rédaction** : aucune carte systématique, par type de couche, des erreurs de K-FAC / EKFAC / TKFAC / diagonale d'AdaFisher contre **à la fois** la Fisher vraie analytique et la Fisher empirique exacte, avec décomposition source/structure et plancher de bruit.

---

## 7. Architecture logicielle (à brancher dans le repo)

### 7.1 Arborescence proposée

```text
fisher_ref/
  conventions.py        # dtype policy, TF32 off, vec = rvec, 1/N scaling, metrics_version
  probes.py             # probe sets (train/val), hashed + versioned, no augmentation
  models/               # A1–A4 definitions; wrappers B1–B5 with a layer-type registry
  registry.py           # module -> {linear, linear_shared, conv, norm, embed, lora_A, lora_B, head}
  capture.py            # hooks: inputs a, normalized x_hat (norms), output grads g per column c
  sources.py            # backprop vectors: type-2 (closed-form sqrt), MC_K (with K^{-1/2}), empirical
  reference/
    dense.py            # regime A: F, E_hat, B_exp as dense fp64
    factor.py           # regime B: per-layer U_l (materialized) or ghost Grams K_l; streaming
    matfree.py          # regime C: curvlinops GGN / EF operators
  approx/
    base.py             # CurvatureBlock protocol (below)
    kfac.py             # expand / reduce, any source
    ekfac.py  tkfac.py  diag.py  blockdiag.py
    adafisher.py        # adapter: reads the upstream optimizer's state, no reimplementation
    norm_layers.py      # exact joint / separate blocks, Hadamard, Prop. 3.1 as implemented
    lora.py
  metrics/
    frobenius.py  spectral.py  stein_kl.py  ngd.py  subspace.py  kron_diag.py  coupling.py  noise_floor.py
  runners/
    p1_structural.py    # fixed theta, fixed probes, lambda sweep
    p2_operational.py   # snapshots of real optimizer state during training
  configs/*.yaml
tests/                  # T1–T12 (Section 10)
```

### 7.2 Interface commune des approximations

Toutes les approximations exposent la même interface, de sorte que chaque métrique s'écrit une seule fois, pour tous les régimes :

```python
from typing import Protocol
import torch

class CurvatureBlock(Protocol):
    P: int
    def matvec(self, v: torch.Tensor) -> torch.Tensor: ...          # K v
    def apply_rows(self, U: torch.Tensor) -> torch.Tensor: ...      # rows u_i -> K u_i
    def solve(self, v: torch.Tensor, lam: float) -> torch.Tensor: ...  # (K + lam I)^{-1} v
    def trace(self) -> torch.Tensor: ...
    def fro2(self) -> torch.Tensor: ...
    def logdet(self, lam: float) -> torch.Tensor: ...               # log det (K + lam I)
    def diag(self) -> torch.Tensor: ...
    def to_dense(self) -> torch.Tensor: ...                         # regime A only
```

Implémentations : `Dense`, `LowRank(U)` (références en régime B), `Kron(A, G)` (vecteurs propres des facteurs en cache, fp64), `EKFAC(QA, QG, s)`, `ScaledKron` (TKFAC), `Diag`, `HadamardNorm`, `BlockDiag([...])`.

### 7.3 Capture des statistiques (source type-2)

```python
import torch

def softmax_sqrt_factor(logits: torch.Tensor) -> torch.Tensor:
    """S of shape (N, C, C) with S[n] @ S[n].T = diag(p_n) - p_n p_n^T.
    Column c of S[n] is sqrt(p_c) * (e_c - p_n)."""
    p = logits.softmax(-1)
    eye = torch.eye(p.shape[-1], dtype=p.dtype, device=p.device)
    return (eye.unsqueeze(0) - p.unsqueeze(-1)) * p.sqrt().unsqueeze(1)

def capture_type2(model, x, layers: dict):
    """For each layer k: inputs a[k] (N, ..., d_in) and backprop vectors g[k] (N, C, ..., d_out).
    Model must be in eval mode (BN running stats, no dropout) so that samples are independent."""
    inputs, outputs = {}, {}
    def make_hook(name):
        def hook(mod, inp, out):
            inputs[name] = inp[0].detach()
            outputs[name] = out
        return hook
    handles = [m.register_forward_hook(make_hook(k)) for k, m in layers.items()]
    logits = model(x)
    for h in handles:
        h.remove()
    S = softmax_sqrt_factor(logits.detach())
    names = list(outputs)
    per_c = [torch.autograd.grad(logits, [outputs[k] for k in names],
                                 grad_outputs=S[:, :, c], retain_graph=True)
             for c in range(logits.shape[-1])]
    g = {k: torch.stack([gc[i] for gc in per_c], dim=1) for i, k in enumerate(names)}
    return inputs, g
```

Précisions : (i) pour les conv, `a` = patchs dépliés (`F.unfold`), `g` remis en forme $(N,C,T,C_{\rm out})$ ; (ii) pour les normalisations, `a` doit être $\hat x$, recalculé dans le hook à partir de l'entrée (LN : statistiques de l'échantillon ; BN-eval : statistiques courantes) ; (iii) un module appelé plusieurs fois (poids liés, par exemple la tête LM liée aux embeddings) écrase le hook : délier en A4, ou accumuler explicitement.

Lignes de $U_\ell$ (convention `rvec`, identique à `W.flatten()`) et Gram « ghost » :

```python
def layer_rows(a, g):
    """a: (N, T, d_in), g: (N, C, T, d_out) -> rows of U_l (N*C, d_out*d_in), before the N^{-1/2} factor."""
    return torch.einsum('nti,ncto->ncoi', a, g).flatten(2).flatten(0, 1)

def layer_gram_ghost(a, g, chunk=64):
    """K_l[(n,c),(n',c')] = sum_{t,s} <a_nt, a_n's> <g_nct, g_n'c's>, no per-sample gradients.
    Cost ~ 2 (N C T)^2 (d_in + d_out); choose per layer vs layer_rows (mixed ghost)."""
    N, C = g.shape[:2]
    out = a.new_zeros(N * C, N * C)
    for i0 in range(0, N, chunk):
        ai, gi = a[i0:i0 + chunk], g[i0:i0 + chunk]
        for j0 in range(i0, N, chunk):
            aj, gj = a[j0:j0 + chunk], g[j0:j0 + chunk]
            Ka = torch.einsum('nti,msi->nmts', ai, aj)
            Kg = torch.einsum('ncto,mdso->ncmdts', gi, gj)
            blk = torch.einsum('nmts,ncmdts->ncmd', Ka, Kg).reshape(len(ai) * C, len(aj) * C)
            out[i0 * C:(i0 + len(ai)) * C, j0 * C:(j0 + len(aj)) * C] = blk
            out[j0 * C:(j0 + len(aj)) * C, i0 * C:(i0 + len(ai)) * C] = blk.T
    return out
```

---

## 8. Dépendances amont et invariants

### 8.1 Dépendances

| Dépendance | Usage | Statut |
|---|---|---|
| `AtlasAnalyticsLab/AdaFisher` — `optimizers/`, `Image_Classification/`, `Language_Model/` | calcul de $H_D,S_D$ (y compris normalisations), état de l'optimiseur en P2, hyperparamètres, GPT-1 4 blocs | **lecture seule**, via adaptateur |
| Lot 1 du repo (extraction de l'infrastructure K-FAC d'AdaFisher) | réutilisé tel quel | la question ouverte « `diag(minmax=False)` bit-exact contre `_get_F_tilde`, `diag(minmax=True)` contre l'Éq. (4) » doit être **tranchée avant P2** |
| `curvlinops` (f-dangel) | oracle : `GGNLinearOperator`, `EFLinearOperator`, `KFACLinearOperator(fisher_type ∈ {TYPE2, MC, EMPIRICAL}, kfac_approx ∈ {EXPAND, REDUCE}, mc_samples)`, `EKFACLinearOperator`, `hutchpp_trace`, `xtrace`, Lanczos | noms d'API vérifiés dans la documentation le 2026-09-10 ; **épingler la version** (`pip freeze`). Couches K-FAC prises en charge : `Linear`, `Conv2d` seulement, donc normalisations et LoRA sont implémentées chez nous |
| `Thrandis/EKFAC-pytorch` | seconde implémentation d'EKFAC (recoupement) | lecture seule |
| `SHI-Labs/Compact-Transformers` | `cct_2_3x2_32` : 2 couches, 2 têtes, `mlp_ratio=1`, $d=128$, tokenizer 2 conv 3×3 sans biais, position apprise | vérifier que la définition utilisée dans `Image_Classification/` d'AdaFisher est la même |
| `transformers` + `peft` | RoBERTa-base, LoRA $r=8$ sur `query`/`value`, tête dans `modules_to_save` | — |
| `AtlasAnalyticsLab/FisherAdapTune` | critère de dérive (option C2) | lecture seule |

Détail à vérifier dans `Language_Model/` : AdaFisher annonce 28 351 488 paramètres appris pour son GPT-1 à 4 couches. C'est **exactement** $4\times7\,087\,872$, le nombre de paramètres de 4 blocs de transformeur à $d=768$ avec biais `[DÉRIVÉ]`. Les embeddings (≈ 31 M pour un vocabulaire de 40 478) ne seraient donc pas comptés comme appris : pré-entraînés et gelés, ou liés. À confirmer, parce que cela change $P$ pour C1.

### 8.2 Ce qui ne change pas contre ce qui est expérimental

**Invariants** (toute modification invalide les comparaisons antérieures) :

- le code amont (adaptateurs seulement) ;
- `curvlinops` comme oracle (jamais forké) ;
- les ensembles de sondes (hachés) ;
- la politique de précision (fp64 pour les références en A et pour les Grams en B, TF32 coupé) ;
- la convention `rvec` et l'échelle $1/N$ ;
- la définition des blocs (par `nn.Module`, variante par tenseur déclarée) ;
- une chaîne `metrics_version` écrite dans chaque fichier de résultats.

**Expérimental** (versionné, peut évoluer) :

- estimateur de normalisation corrigé (Hadamard) ;
- adaptation *expand* de TKFAC aux couches partagées ;
- iEF appliqué au critère de dérive ;
- futur Kronecker de rang $r$ ;
- choix des $\alpha$ d'amortissement.

---

## 9. Lots d'implémentation (une itération à la fois)

| Lot | Contenu | Livrable | Critère de sortie |
|---|---|---|---|
| **0** | `conventions.py`, drapeaux TF32, sondes, `registry.py` ; tests T1–T6 | CI verte | T1–T6 passent en fp64 |
| **1** | A1 (MLP-LN) de bout en bout : $F,\hat E,\hat F_K$ dense, zoo Linear + LN, M1–M8, plancher de bruit, 5 points de contrôle × 5 graines | premier jeu de figures (§11) + résultats `parquet` | T7–T9 passent ; plancher de bruit tracé |
| **2** | A2 (conv, GN et BN-eval), A3 (QKV fusionné et séparé, pooling moyen), A4 (LM, deux conventions de $\hat E$) ; décomposition partage / indépendance | carte complète en régime A | HF2, HF3, HF4 tranchées sur A |
| **3** | Régime B : `factor.py`, Grams par couche, ghost, Woodbury (M3, M5) ; **validé d'abord sur A1–A3 contre le dense**, puis B1 (CCT), B2 (ResNet-20) | carte B1/B2 | T10–T11 : écart B/A ≤ tolérances |
| **4** | B3 (RoBERTa-LoRA, SST-2 puis MNLI), B4 (ViT-Tiny, en flux) | carte PEFT + largeur réaliste | HF5, HF6 tranchées |
| **5** | Adaptateur AdaFisher (bit-exact) + protocole P2 sur A2, B1, B2 | P1 contre P2 | T12 ; HF7 tranchée |
| **6** *(options)* | B5 ResNet-18 (ghost mixte) ; C1 GPT-1 4 blocs (sans matrice, K-FAC-MC seulement) ; C2 pont FisherAdapTune : décisions de gel comparées au plancher de bruit et au critère recalculé avec $F$ exacte / iEF | extensions | à décider après le Lot 5 |

Ordre justifié : le régime A est l'oracle du régime B ; P2 dépend de la question ouverte du Lot 1 du repo ; les options n'ont de sens qu'une fois la carte principale stable.

---

## 10. Critères de validation empirique

### 10.1 Tests d'exactitude (bloquants)

| Test | Énoncé | Tolérance |
|---|---|---|
| T1 | $Fv$ dense (A) contre `GGNLinearOperator` sur 20 vecteurs aléatoires | rel. ≤ $10^{-10}$ (fp64) |
| T2 | $F$ type-2 contre Fisher MC à $K=10^4$ | $e_F\approx O(K^{-1/2})$, cohérent sur 5 graines |
| T3 | *KFAC from scratch*, Test 1 ($N=1$, sans partage) : K-FAC-type-2 = bloc GGN, K-FAC-emp = bloc EF | rel. ≤ $10^{-12}$ |
| T4 | Test 2 (réseau linéaire profond, MSE) : K-FAC-type-2 = bloc GGN, **K-FAC-emp ≠ EF** (le test doit échouer à l'égalité) | ≤ $10^{-12}$ / écart > $10^{-3}$ |
| T5 | Eschenhagen et al., Props. 1–2 : expand exact en cadre expand, reduce exact en cadre reduce (pooling moyen) | ≤ $10^{-10}$ |
| T6 | Blocs de normalisation et BN-eval : gradients par échantillon contre différences finies centrées | rel. ≤ $10^{-6}$ |
| T7 | KL Woodbury (§5.1) contre KL dense sur A1 ; $D_\lambda(R\Vert R)=0$ | ≤ $10^{-8}$ |
| T8 | TKFAC : $\mathrm{tr}(K)=\mathrm{tr}(B_\ell)$ sur couches sans partage | ≤ $10^{-10}$ |
| T9 | EKFAC : $\mathrm{tr}(K)=\mathrm{tr}(B_\ell)$ ; recoupement avec EKFAC-pytorch | ≤ $10^{-10}$ / ≤ $10^{-6}$ |
| T10 | Régime B contre régime A, sur A1–A3, toutes métriques | rel. ≤ $10^{-8}$ |
| T11 | Gram ghost contre Gram matérialisé, sur une couche où les deux tiennent | ≤ $10^{-10}$ (fp64), ≤ $10^{-5}$ (fp32) |
| T12 | Adaptateur AdaFisher contre état interne amont, même batch et même graine | écart nul, ou ≤ $10^{-7}$ si l'ordre des réductions diffère |

### 10.2 Critères de lecture

- Une différence entre deux approximations n'est rapportée que si elle dépasse le plancher de bruit de $F$ à 95 % (§3.4).
- Toute conclusion « par type de couche » doit valoir sur au moins un modèle de régime A **et** un de régime B.
- Toute conclusion sur l'inverse (M3, M4) est montrée sur tout le balayage en $\alpha$, jamais pour un seul $\lambda$.

---

## 11. Règles de décision : du résultat à la direction de recherche

| Observation | Conséquence pour la suite |
|---|---|
| Erreur de structure (type-2) ≫ erreur de source, concentrée sur les couches partagées | priorité au Kronecker de rang $r$ et au choix expand/reduce par couche (entrées #4 et #6 du classement du rapport) |
| Erreur de source dominante | priorité à iEF et à la Fisher exacte par dualité en PEFT (entrées #2, #3) |
| P2 ≫ P1 (HF7) | la faiblesse est l'EMA, la min-max ou l'amortissement, pas la structure : pistes EMA géodésique, fixation de jauge par le déterminant, oubli directionnel |
| Biais d'indépendance diagonal fort sur les normalisations (HF3–HF4) | estimateur de normalisation corrigé (Hadamard + terme croisé $\gamma$–$\beta$) : contribution peu coûteuse et directement opposable à la Prop. 3.1 |
| $\rho(\mathrm{BD}(F))\approx1$ (HF5 confirmée) | abandonner les pistes inter-couches (entrées 16, 21, 34, 35 du rapport) |
| Différences sous le plancher de bruit à la taille $N$ d'AdaFisher B.2 | critique méthodologique : la validation « closely aligns » d'AdaFisher n'est pas falsifiable à cette taille d'échantillon |

---

## 12. Risques et parades

- **Rang déficient en régime B** : métriques d'inverse dominées par $\lambda$. Parades : balayage, régime A comme juge, décomposition image/noyau de $F$ pour M3.
- **Précision** : accumuler $U$ en fp32 mais les Grams et les produits $UKU^\top$ en fp64 ; TF32 coupé (T1 le détecte).
- **Poids liés et modules réutilisés** : hooks écrasés ; délier, ou accumuler explicitement (A4).
- **BN train contre eval** : la référence est en eval, l'écart avec le mode train est mesuré en P2.
- **Dérive d'API** (`curvlinops`, code AdaFisher) : versions épinglées, T12 rejoué à chaque mise à jour.
- **Coût du ghost sur les premières conv** ($T$ grand) : choix par couche selon $\min(4mP_\ell,\;2(mT)^2(d_{\rm in}+d_{\rm out}))$.
- **Stockage** : ne pas écrire les Grams ($m^2$ par couche et par point de contrôle) ; écrire uniquement les métriques, sauf pour B1 où l'on garde les Grams d'un point de contrôle pour les analyses a posteriori.

---

## 13. Sorties attendues et budget

**Format** : un fichier `parquet` par (modèle, point de contrôle), avec les colonnes `layer, layer_type, structure, source, reference, lambda_alpha, metric, value, ci_low, ci_high, N, seed, protocol, metrics_version`.

**Figures** :

- F1 : carte $\rho$ (structure × type de couche), une par référence ;
- F2 : barres de décomposition source / structure / interaction ;
- F3 : courbes au fil de l'entraînement, sondes train contre val ;
- F4 : $\sigma_2/\sigma_1$ par couche ;
- F5 : matrices de couplage $c_{\ell\ell'}$ ;
- F6 : nuage de rangs P1 contre P2 ;
- F7 : plancher de bruit en fonction de $N$ ;
- F8 : panneau spécial normalisations.

**Budget** `[ESTIMATION, à mesurer au Lot 1]` sur instance 2g.20gb :

- régime A : quelques minutes par point de contrôle (SYRK ≈ $2\cdot10^{13}$ flop fp64, `eigh` de $2{,}5\cdot10^4$) ;
- B1/B2 : < 30 min par point de contrôle ;
- B4 en flux : 1–2 h ;
- B5 ghost : ≈ 1 h.

Campagne principale (A1–A4, B1–B4, 5 points de contrôle, 3–5 graines, P1 et P2) : de l'ordre de 100–200 h-GPU MIG.

---

## Références (vérifiées sur arXiv le 2026-09-10, sauf mention)

- Martens, *New insights and perspectives on the natural gradient method*, arXiv:1412.1193 (JMLR 2020), §9 : $F=G$ sous lien canonique.
- Martens & Grosse, K-FAC, arXiv:1503.05671 ; Grosse & Martens, K-FAC conv, arXiv:1602.01407.
- George et al., EKFAC, arXiv:1806.03884.
- Gao, Liu, Huang, Wang, Wang, Xu & Yu, *A Trace-restricted Kronecker-Factored Approximation to Natural Gradient* (TKFAC), arXiv:2011.10741.
- Eschenhagen, Immer, Turner, Schneider & Hennig, K-FAC expand/reduce, arXiv:2311.00636.
- Dangel, Mucsányi, Weber & Eschenhagen, *KFAC From Scratch*, arXiv:2507.05127.
- Dangel, Eschenhagen, Ormaniec, Fernandez, Tatzel & Kristiadi, `curvlinops`, arXiv:2501.19183 ; documentation `curvlinops.readthedocs.io` (API vérifiée).
- Dangel, Tatzel & Hennig, *ViViT*, arXiv:2106.02624.
- Kunstner, Balles & Hennig, arXiv:1905.12558 ; Wu, Yu, Zhang & Woodland, iEF, arXiv:2406.06420.
- Li, Dangel, Tam & Raffel, *Fishers for Free?*, arXiv:2507.18807.
- Benzing, arXiv:2201.12250 ; Abreu, Vyas, Kakade & Morwani, arXiv:2510.09378.
- Koroko et al., KP-SVD, arXiv:2201.10285.
- Novak, Sohl-Dickstein & Schoenholz, *Fast Finite Width NTK*, arXiv:2206.08720.
- Li, Tramèr, Liang & Hashimoto (ghost clipping), arXiv:2110.05679 ; Bu, Mao & Xu (mixed ghost clipping), arXiv:2205.10683.
- Zhang et al., *Why Transformers Need Adam: A Hessian Perspective*, arXiv:2402.16788 ; Ormaniec, Dangel & Singh, arXiv:2410.10986.
- Hassani et al., CCT, arXiv:2104.05704 (0,28 M paramètres pour la plus petite variante) ; Xie et al., SegFormer, arXiv:2105.15203 (taille de B0 à revérifier).
- Martins Gomes et al., AdaFisher, arXiv:2405.16397 (Prop. 3.1, Prop. 3.2 / Éq. 4, App. A.3, App. B.2, Tableau 2, App. D.3) ; Rostami, Chen & Hosseini, FisherAdapTune, arXiv:2606.10196.
- Matériel : documentation Alliance / WestDRI sur les instances MIG H100 (`2g.20gb` ≈ 2/8 de GPU, 20 Go).

---

## Annexe A — Récapitulatif des types de couches par modèle

| Type | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|---|---|---|
| Linear sans partage | ✓ | tête | tête | — | tête, pool | tête | tête | tête |
| Linear partagé (QKV, sortie, MLP) | — | — | ✓ (reduce) | ✓ (expand) | ✓ | — | — | ✓ |
| Conv | — | ✓ | patch | — | tokenizer | ✓ | — | patch |
| Normalisation | LN | GN / BN-eval | LN | LN | LN | BN-eval | (gelée) | LN |
| Embedding | — | — | position | token + position | position | — | (gelée) | position |
| LoRA A / B | — | — | — | — | — | — | ✓ | — |

## Annexe B — Script de comptage (reproductible)

Le script `feasibility.py` (livré à côté de ce plan) recalcule les nombres de paramètres par type de couche et les tailles mémoire ci-dessus à partir des architectures, sans télécharger de poids. À relancer après toute modification d'architecture, puis à recouper avec `sum(p.numel() for p in model.parameters() if p.requires_grad)` sur le cluster.