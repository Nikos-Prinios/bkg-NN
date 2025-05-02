import csv
import random
from collections import deque
from pathlib import Path

def charger_phrases_unique(csv_path: str) -> dict[str, deque]: # Note: deque sans type hinting interne ok
    """
    Charge un CSV (format supposé: categorie,phrase_avec_potentielle_virgule)
    et renvoie un dictionnaire :
    { "catégorie": deque([...phrases mélangées...]), ... }
    """
    piles: dict[str, list[str]] = {}
    try:
        with Path(csv_path).open(encoding="utf-8") as f:
            # Lire l'en-tête manuellement pour le sauter
            try:
                next(f) # Saute la première ligne (en-tête)
            except StopIteration:
                print(f"Warning: Le fichier CSV '{csv_path}' est vide ou n'a pas d'en-tête.")
                return {} # Retourne un dict vide si le fichier est vide

            # Lire les lignes suivantes
            for line_num, line in enumerate(f, start=2): # Commence à compter à la ligne 2
                line = line.strip() # Enlève les espaces/retours chariot au début/fin
                if not line: # Ignore les lignes vides
                    continue

                # Découper sur la PREMIÈRE virgule seulement
                parts = line.split(',', 1)

                if len(parts) == 2:
                    category = parts[0].strip()
                    phrase = parts[1].strip()

                    # Optionnel : Enlever les guillemets si présents autour des champs
                    if category.startswith('"') and category.endswith('"'):
                        category = category[1:-1]
                    if phrase.startswith('"') and phrase.endswith('"'):
                        phrase = phrase[1:-1]

                    # Ajouter à la liste correspondante
                    piles.setdefault(category, []).append(phrase)
                else:
                    # La ligne n'a pas pu être découpée en deux parties (pas de virgule?)
                    print(f"Warning: Ligne {line_num} ignorée dans '{csv_path}' - format inattendu : {line}")

    except FileNotFoundError:
        print(f"ERREUR : Le fichier CSV '{csv_path}' est introuvable.")
        return {} # Retourne un dict vide si le fichier n'existe pas
    except Exception as e:
        print(f"ERREUR : Problème lors de la lecture du fichier CSV '{csv_path}': {e}")
        return {} # Retourne un dict vide en cas d'autre erreur

    if not piles:
        print(f"Warning: Aucune phrase valide n'a été chargée depuis '{csv_path}'. Vérifiez le format du fichier.")

    # Mélange puis conversion en deque
    piles_deque: dict[str, deque[str]] = {} # Créer un nouveau dict pour les deques
    for cat, lst in piles.items():
        random.shuffle(lst)
        piles_deque[cat] = deque(lst) # Assigner la deque au nouveau dict

    return piles_deque # Retourner le dict contenant les deques


class PhraseManager:
    """Garantie : aucune phrase répétée tant que la pile de sa catégorie n’est pas vide."""

    def __init__(self, csv_path: str):
        # Appelle la fonction corrigée pour charger les phrases
        self.piles = charger_phrases_unique(csv_path)
        if not self.piles:
             print(f"Attention : PhraseManager initialisé sans aucune phrase chargée depuis {csv_path}.")

    def get(self, category: str) -> str:
        # Vérifie si la catégorie existe dans les piles chargées
        if category not in self.piles:
             # Tentative de fallback si la catégorie exacte n'existe pas?
             # Ou simplement retourner un message par défaut.
             print(f"Debug: Catégorie '{category}' non trouvée dans PhraseManager.")
             return "L’IA réfléchit intensément..." # Message générique

        pile = self.piles[category] # Accède à la deque pour cette catégorie

        if pile: # Si la deque n'est pas vide
            return pile.popleft() # Retire et renvoie l'élément de gauche (le premier)
        else:
            # La pile pour cette catégorie est vide
            # On pourrait recharger/remélanger ici, mais pour l'instant, message fixe.
            # Attention: Si on recharge ici, il faut accéder à la liste originale avant shuffle/deque
            print(f"Debug: Pile vide pour la catégorie '{category}'.")
            return "L’IA se répète, signe de sa grande intelligence."


# ----------- Exemple d’usage -----------
if __name__ == "__main__":
    # Test simple du chargement et de la récupération
    csv_file = "sentences.csv"
    print(f"Test de chargement depuis : {csv_file}")
    pm = PhraseManager(csv_file)

    print("\nPiles chargées :")
    for cat, pile in pm.piles.items():
        print(f"- {cat}: {len(pile)} phrases")

    if pm.piles:
        print("\nExemples de récupération :")
        # Essayer de récupérer des phrases de catégories existantes
        categories_test = list(pm.piles.keys())
        if categories_test:
            cat_test_1 = categories_test[0]
            print(f"\nCatégorie '{cat_test_1}':")
            for _ in range(5): # Essayer de piocher 5 phrases
                phrase = pm.get(cat_test_1)
                print(f"  -> {phrase}")
                if "répète" in phrase or "plus rien" in phrase: # Arrêter si la pile semble vide
                    break

            if len(categories_test) > 1:
                cat_test_2 = categories_test[1]
                print(f"\nCatégorie '{cat_test_2}':")
                print(f"  -> {pm.get(cat_test_2)}")

        # Tester une catégorie inexistante
        print("\nCatégorie 'inexistante':")
        print(f"  -> {pm.get('inexistante')}")
    else:
        print("\nAucune pile chargée, impossible de tester la récupération.")