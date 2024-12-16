import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
import sqlite3

# Initialisation des embeddings et de la base de données Chroma
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="./db-nordnet", embedding_function=embedding_model)
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "lambda_mult": 0.7,  # Paramètre MMR pour l'équilibre pertinence/diversité
    }
)

# Configuration du modèle de langage
llm = ChatOllama(model="llama3.2", keep_alive="3h", max_tokens=512, temperature=0.3)

# Updated prompt template with more context and guidance
template = """<bos><start_of_turn>system
You are an AI assistant specialized in Nordnet's internet services, network technologies, and customer support. Your goal is to provide helpful, informative, and precise answers while maintaining the following principles:

1. Prioritize accuracy and clarity in your responses
2. Draw directly from the provided context
3. If the context doesn't fully cover the question, provide the most relevant information you can
4. Be conversational and helpful, but stay focused on Nordnet-related topics

Acceptable response types:
- Technical explanations about internet services
- Troubleshooting guidance
- Service descriptions
- General network technology insights
- Customer support information

<start_of_turn>user
Context: {context}

Question: {question}

Provide a comprehensive and helpful response based on the available information."""

prompt = ChatPromptTemplate.from_template(template)

# Fonction pour formater les sources de manière plus lisible
def format_sources_display(sources):
    """
    Format source display to be more readable and concise
    
    Args:
        sources (list): List of source dictionaries with 'source' and 'content' keys
    
    Returns:
        list: Formatted sources with truncated content and clean source names
    """
    formatted_sources = []
    for source in sources:
        # Extract clean source name (remove 'data/' prefix and file extension)
        clean_source = source['source'].replace("data/", "").rsplit('.', 1)[0]
        
        # Truncate content to a reasonable length
        truncated_content = source['content'][:200] + '...' if len(source['content']) > 200 else source['content']
        
        formatted_sources.append({
            'name': clean_source,
            'content': truncated_content,
            'full_source': source['source']  # Preserve full source for potential linking
        })
    
    return formatted_sources

# Fonction pour formater les documents et extraire les sources
def format_docs(docs):
    chunks_and_sources = []
    for doc in docs:
        content = doc.page_content  # The text chunk
        source = doc.metadata.get("source", "Source inconnue")  # The associated source
        chunks_and_sources.append({"content": content, "source": source})
    return chunks_and_sources

# Fonction pour générer la réponse avec les sources
def generate_response_with_sources(retriever, question):
    # Récupère les documents similaires
    docs = retriever.get_relevant_documents(question)
    chunks_and_sources = format_docs(docs)

    # Build context from document contents
    context = "\n\n".join([chunk["content"] for chunk in chunks_and_sources])
    sources = [chunk["source"] for chunk in chunks_and_sources]

    # Construire le message
    messages = [
        {"role": "system", "content": "You are an expert of network connection and an engineer at the French enterprise Nordnet."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
    ]
    
    # Génère une réponse avec le modèle
    chain_response = llm.invoke(messages)
    response = chain_response.content
    
    return response, chunks_and_sources

# Configuration de la page Streamlit
st.set_page_config(page_title="Nordnet", page_icon="🛰️")

# Ajout de styles personnalisés
st.markdown("""
    <style>
        /* Messages de l'utilisateur */
        .user-message {
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
            max-width: 100%;
            align-self: flex-end;
            background-color: #303030
        }

        /* Messages de l'assistant */
        .assistant-message {
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
            max-width: 100%;
            align-self: flex-start;
            background-color: #303030
        }

        .chat-message {
            font-size: 14px;
        }

        /* Messages en général */
        .message-container {
            display: flex;
            flex-direction: column;
        }

        /* Style pour les sources */
        .sources-container {
            background-color: #2C2C2C;
            padding: 10px;
            border-radius: 5px;
            margin-top: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# Initialisation des messages si pas déjà fait
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": ("Bonjour! Je suis là pour répondre à vos questions Nordnet et la fibre. Comment puis-je vous aider?", [])}
    ]

# Sidebar avec historique
with st.sidebar:
    st.markdown(f'<img src="https://cdn.nordnet.fr/components/v1/assets/components/images/logos/logo-nordnet-white.svg" class="sidebar-logo">', unsafe_allow_html=True)

    # Bouton pour effacer l'historique
    if st.button("Effacer l'historique"):
        st.session_state.messages = [
            {"role": "assistant", "content": ("Bonjour! Je suis là pour répondre à vos questions Nordnet et la fibre. Comment puis-je vous aider?", [])}
        ]
    
    # Affichage de l'historique dans la sidebar
    st.markdown("### Historique des conversations")
    for message in st.session_state.messages[1:]:  # Skip the first welcome message
        if message["role"] == "user":
            st.markdown(f"**😎 Vous:** {message['content'][:100]}...")
        else:
            response, _ = message["content"]
            st.markdown(f"**🧠 Assistant:** {response[:100]}...")

# Zone principale de chat
st.title("Bonjour !")

# Affichage des messages dans la zone principale
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="message-container user-message">😎 {message["content"]}</div>', unsafe_allow_html=True)
    elif message["role"] == "assistant":
        # Gestion des réponses avec les sources
        response, sources = message["content"]
        st.markdown(f'<div class="message-container assistant-message">🧠 {response}</div>', unsafe_allow_html=True)
        
        if sources:
            formatted_sources = format_sources_display(sources)
            with st.expander("Sources"):
                for source in formatted_sources:
                    st.markdown(f"""
<div class="sources-container">
**{source['name']}**: {source['content']}
</div>
                    """, unsafe_allow_html=True)

# Zone de chat
if prompt := st.chat_input("Posez votre question ici..."):
    # Ajout du message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="message-container user-message">😎 {prompt}</div>', unsafe_allow_html=True)

    
    
    # Génération de la réponse
    with st.spinner("Réflexion en cours..."):
        response, sources = generate_response_with_sources(retriever, prompt)
        st.session_state.messages.append({"role": "assistant", "content": (response, sources)})
        st.markdown(f'<div class="message-container assistant-message">🧠 {response}</div>', unsafe_allow_html=True)
        
        if sources:
            formatted_sources = format_sources_display(sources)
            with st.expander("Sources"):
                for source in formatted_sources:
                    st.markdown(f"""
<div class="sources-container">
**{source['name']}**: {source['content']}
</div>
                    """, unsafe_allow_html=True)

