import pandas as pd

def split_dataframe(df, max_rows_per_sheet=1048576):
    """
    Divise un DataFrame en chunks respectant la limite de lignes par feuille Excel
    """
    chunks = []
    total_rows = len(df)
    current_start = 0
    
    while current_start < total_rows:
        chunk_size = min(max_rows_per_sheet, total_rows - current_start)
        chunks.append(df.iloc[current_start:current_start + chunk_size])
        current_start += chunk_size
    
    return chunks

# Charger le fichier Excel
file_path = 'src2/data/datasetBig.xlsx'  # Remplacez par le bon chemin vers votre fichier
df = pd.read_excel(file_path)

# Séparer les commentaires par un point
df['comment'] = df['comment'].str.split('.')

# Trouver la longueur maximale de la liste de commentaires dans chaque ligne
max_len = df['comment'].apply(len).max()

# Remplir les autres colonnes pour chaque commentaire séparé
df_expanded = df.explode('comment')

# Remplir les autres colonnes en dupliquant les valeurs
df_expanded = df_expanded.reset_index(drop=True)

# Diviser le DataFrame en chunks respectant la limite Excel
chunks = split_dataframe(df_expanded)

# Sauvegarder dans un nouveau fichier Excel
output_path = 'src2/data/split_comments.xlsx'
with pd.ExcelWriter(output_path) as writer:
    for idx, chunk in enumerate(chunks, start=1):
        chunk.to_excel(writer, sheet_name=f'Sheet{idx}', index=False)
        print(f"Feuille {idx} sauvegardée ({len(chunk)} lignes)")

print(f"\nFichier sauvegardé dans {output_path}")