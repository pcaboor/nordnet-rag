# nordnet-rag

```
venv\Scripts\activate
```

```
source venv/bin/activate
```


```
pip freeze > requirements.txt
```

```
pip install -r requirements.txt
```



```sql
CREATE TABLE links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword TEXT NOT NULL, -- Pour lier un mot-clé ou une requête
    url TEXT NOT NULL, -- Lien associé
    description TEXT -- Description du lien (facultatif)
);
```

```sql
INSERT INTO links (keyword, url, description)
VALUES
('fibre', 'https://www.nordnet.com/connexion-internet/internet-fibre', 'Page des offres fibre NordNet'),
('satellite', 'https://www.nordnet.com/connexion-internet/internet-satellite', 'Page des offres satellite NordNet');
```

```py
import sqlite3

def fetch_links_from_db(query, db_path="nordnet_links.db"):
    """
    Récupère les liens pertinents à partir d'une base SQLite en fonction d'une requête.
    
    Args:
        query (str): La requête utilisateur.
        db_path (str): Le chemin de la base de données SQLite.

    Returns:
        list: Une liste de tuples contenant (keyword, url, description).
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Recherche les liens correspondants au mot-clé (simple ou avancée)
    cursor.execute("""
        SELECT keyword, url, description
        FROM links
        WHERE keyword LIKE ?
    """, (f"%{query}%",))  # Recherche approximative pour inclure des correspondances partielles

    results = cursor.fetchall()
    conn.close()
    
    return results
```

```py
def generate_response_with_sources(retriever, question):
    # Récupère les documents similaires
    docs = retriever.get_relevant_documents(question)
    context, sources = format_docs(docs)

    # Récupère les liens associés à partir de la base de données
    db_links = fetch_links_from_db(question)
    
    # Formate les liens récupérés
    additional_links = [f"{desc}: {url}" for _, url, desc in db_links]

    # Construire le message
    messages = [
        {"role": "system", "content": "You are an expert of network connection and an engineer at the French enterprise NordNet."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
    ]
    
    # Génère une réponse avec le modèle
    chain_response = llm.invoke(messages)
    response = chain_response.content
    
    return response, sources + additional_links
```

```py
if sources:
    with st.expander("Sources et liens pertinents"):
        for source in sources:
            st.markdown(f"- [Consulter]({source})")

        # Ajouter des liens de la base SQLite
        if db_links := fetch_links_from_db(prompt):
            st.markdown("### Liens additionnels :")
            for _, url, desc in db_links:
                st.markdown(f"- **{desc}:** [Visiter]({url})")
```
