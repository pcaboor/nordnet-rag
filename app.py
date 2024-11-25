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

# Template du prompt
template = """<bos><start_of_turn>user
Answer the question based only on the following context and provide a detailed, accurate response...

CONTEXT: {context}
QUESTION: {question}
"""

prompt = ChatPromptTemplate.from_template(template)



# Fonction pour formater les documents et extraire les sources
def format_docs(docs):
    formatted_docs = [doc.page_content for doc in docs]
    sources = [doc.metadata.get('source', 'Source inconnue') for doc in docs]
    return "\n\n".join(formatted_docs), sources

# Fonction pour générer la réponse avec les sources
def generate_response_with_sources(retriever, question):
    # Récupère les documents similaires
    docs = retriever.get_relevant_documents(question)
    context, sources = format_docs(docs)

    # Construire le message
    messages = [
        {"role": "system", "content": "You are an expert of network connection and an engineer at the French enterprise Nordnet."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
    ]
    
    # Génère une réponse avec le modèle
    chain_response = llm.invoke(messages)
    response = chain_response.content
    
    return response, sources

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
            with st.expander("Sources"):
                for source in sources:
                    st.markdown(f"- [Consulter]({source})")
                

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
            with st.expander("Sources"):
                for source in sources:
                    st.markdown(f"- [{source}]({source})")
