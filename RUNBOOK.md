# Runbook : alerte de dérive sur le modèle de crédit

À ouvrir quand `monitor.py` a levé une alerte : au moins une variable d'un lot a un PSI supérieur à 0,25.

Ce document sert à la personne d'astreinte comme à l'agent. Il dit dans quel ordre enquêter, comment reconnaître chacune des cinq causes possibles, et quelle action recommander.

## Principe

Un PSI élevé dit seulement **qu'une variable a changé**. Il ne dit ni pourquoi, ni si c'est grave. L'enquête répond à ces deux questions, dans cet ordre :

1. **Est-ce un problème technique ?** Les données sont cassées entre la source et le modèle. On le vérifie en premier, car les signes sont nets et la correction est obligatoire, **quelle que soit l'importance de la variable**.
2. **Sinon, ce changement compte-t-il pour le modèle ?** Les clients ont vraiment changé. Il reste à savoir si le modèle s'en sert.

## Procédure d'enquête

Suivre les étapes dans l'ordre. Une variable expliquée à une étape n'a pas besoin d'être examinée aux étapes suivantes.

| Étape | Outil | Ce qu'on regarde |
|---|---|---|
| 1 | `compute_psi(day)` | Liste des variables dont le PSI dépasse 0,25. Ce sont les variables à expliquer. En dessous de 0,15, c'est en général du bruit (lots de 250 lignes). |
| 2 | `missing_data_rate(day)` | Une variable de la liste a-t-elle des valeurs manquantes absentes de la référence ? → **donnees_manquantes** |
| 3 | `never_seen_modality(day)` | Une variable texte de la liste contient-elle une valeur jamais vue ? → **modalite_inconnue** |
| 4 | `compare_mean_median(day, variable)` | Pour chaque variable **numérique** restante : le rapport lot / référence est-il proche de 10, 100, 1000 ou 0,1, 0,01 ? → **bug_unite** |
| 5 | `feature_importance(variable)` | Pour chaque variable restante (changement réel) : niveau `forte` ou `moyenne` → **derive_population** ; niveau `faible` → **fausse_alerte** |

## Les cinq causes

### 1. `bug_unite` : erreur d'unité

- **Signature :** `compare_mean_median` donne un rapport proche d'une puissance de 10 (environ 100, 1000, 0,01…), sur la moyenne **et** sur la médiane. Aucun manquant, aucune modalité inconnue.
- **À ne pas confondre avec :** une dérive de population, qui donne un rapport réaliste (entre 0,5 et 2 environ). Aucune vraie population ne voit tous ses montants multipliés par 100 du jour au lendemain.
- **Action : `corriger_pipeline`.** Écarter le lot, trouver l'étape qui a changé d'unité (centimes au lieu d'euros, par exemple) et la corriger. Ne pas réentraîner le modèle.

### 2. `donnees_manquantes` : valeurs manquantes

- **Signature :** `missing_data_rate` montre un taux non nul dans le lot, alors que la référence est à 0. Le PSI de la variable est élevé, mais `compare_mean_median` montre une moyenne et une médiane presque inchangées : les valeurs restantes sont normales, ce sont les cases vides qui font monter le PSI.
- **À ne pas confondre avec :** une dérive, où ce sont les valeurs elles-mêmes qui changent.
- **Action : `corriger_pipeline`.** Écarter le lot et chercher pourquoi le champ n'est plus rempli (collecte, jointure, champ renommé à la source).

### 3. `modalite_inconnue` : modalité inconnue

- **Signature :** `never_seen_modality` renvoie, pour une variable texte, une valeur absente de la référence, souvent sur une part importante du lot.
- **À ne pas confondre avec :** une dérive sur une variable texte, où les proportions des modalités **connues** changent, sans nouvelle valeur.
- **Action : `corriger_pipeline`.** Le modèle ne connaît pas cette valeur et prédit à l'aveugle pour ces clients. Identifier son origine (nouveau choix dans l'application, faute de frappe, encodage modifié) et la ramener à une modalité connue, ou décider explicitement de l'intégrer au prochain entraînement. Ne pas ignorer, même si la variable a une importance faible.

### 4. `derive_population` : vraie dérive de la population

- **Signature :** aucune trace technique aux étapes 2 à 4. Les changements sont plausibles (rapport entre 0,5 et 2 environ), touchent souvent plusieurs variables liées dans le même sens (par exemple l'âge et l'ancienneté d'emploi), et au moins une de ces variables a une importance `forte` ou `moyenne`.
- **À ne pas confondre avec :** une fausse alerte, qui est aussi un vrai changement, mais sur des variables que le modèle utilise peu.
- **Action : `reentrainer_model`.** Les clients ne ressemblent plus à ceux sur lesquels le modèle a appris. Prévenir l'équipe modèle et réentraîner sur des données récentes.

### 5. `fausse_alerte` : fausse alerte

- **Signature :** aucune trace technique aux étapes 2 à 4. Le changement est réel, mais toutes les variables concernées ont une importance `faible` dans `feature_importance`.
- **À ne pas confondre avec :** une dérive de population, qui touche au moins une variable importante.
- **Action : `ignorer`.** Le modèle n'est pas affecté. Noter l'événement, sans autre intervention.

## Plusieurs causes dans le même lot

Traiter chaque variable alertée séparément. S'il y a à la fois une cause technique et une autre cause, **l'action technique passe en premier** (`corriger_pipeline`) : on ne peut juger une dérive que sur des données saines.

## Format de la conclusion

Terminer l'enquête par :

- **cause** : un des cinq mots-clés ci-dessus, écrit à l'identique ;
- **variables** : la liste des variables en cause ;
- **action** : `corriger_pipeline`, `reentrainer_model` ou `ignorer` ;
- **justification** : les chiffres des outils qui ont permis de trancher.

## Récapitulatif

| Cause | Signe décisif | Outil | Action |
|---|---|---|---|
| `bug_unite` | rapport ≈ puissance de 10 | `compare_mean_median` | `corriger_pipeline` |
| `donnees_manquantes` | taux de manquants > 0, référence à 0 | `missing_data_rate` | `corriger_pipeline` |
| `modalite_inconnue` | valeur jamais vue | `never_seen_modality` | `corriger_pipeline` |
| `derive_population` | changement plausible sur variable importante | `feature_importance` | `reentrainer_model` |
| `fausse_alerte` | changement plausible sur variable peu importante | `feature_importance` | `ignorer` |
