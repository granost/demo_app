import streamlit as st
from google import genai
from google.genai import types
from google.api_core import exceptions
import numpy as np
from numpy.linalg import norm
import pandas as pd

st.header('Demo app')
with st.expander("Wpisz klucz do API Gemini"):
    gemini_key = st.text_input("Gemini key", type="password")

if gemini_key:
    client = genai.Client(api_key=gemini_key)


# funkcje uniwersalne
def get_gemini_response(prompt):
    """
    Generate Google Gemini response
    :param prompt: a string with a user query or an API supported structure
    :return: model response
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text
    #
    except exceptions.PermissionDenied as e:
        st.error('Niewłaściwy klucz API.')
        st.error(e)
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None


# funkcje do chatbota
def cosine_sim(a: list, b: list):
    cos_sim = np.dot(a, b) / (norm(a) * norm(b))
    return cos_sim


@st.cache_resource
def get_embeddings(texts):
    """
    Generate Google embeddings.
    :param texts: list of strings
    :return: embeddings
    """
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"))
    return result.embeddings


def find_relevant_faq(query, faq_embeddings, faq_data, threshold=0.7, num_answers=3):
    """
    Znajdź pasujące odpowiedzi z FAQ
    :param query: user query
    :param faq_embeddings: embedded FAQ questions
    :param faq_data: FAQ DataFrame
    :param threshold: min cosine similarity threshold
    :param num_answers: number of relevant answers
    :return: a list of relevant FAQ answers
    """
    query_embedding = get_embeddings([query])
    # calculate cosine similarities between the query and embedded faq questions
    similarities = [cosine_sim(query_embedding[0].values, i.values) for i in faq_embeddings]
    sorted_indices = np.argsort(similarities)[::-1]
    filtered_instances = [i for i in sorted_indices if similarities[i] >= threshold][:num_answers]
    relevant_answers = [faq_data.loc[i, 'answers'] for i in filtered_instances]
    return relevant_answers


def query_gemini_rag(query: str, context: list = None):
    context_str = "\n".join([i for i in context])
    if context:
        prompt = f"Jesteś pomocnym asystentem klienta. Odpowiedz rzeczowo, po polsku. Użyj kontekstu: {context_str}. Pytanie: {query}"
    else:
        prompt = f"Jesteś pomocnym asystentem klienta. Odpowiedz rzeczowo, po polsku. Pytanie: {query}"
    response = get_gemini_response(prompt)
    return response


# Osobne zakładki dla różnych funkcjonalności
tab1, tab2 = st.tabs(['Opisywacz zdjęć', 'FAQ Chatbot'])

with tab1:
    st.title("Opisywacz zdjęć")

    # wgrywanie zdjęć
    uploaded_file = st.file_uploader("Wybierz zdjęcie", type=["jpg", "jpeg", "png", "bmp"], key="image_uploader"
                                     )

    if uploaded_file:
        # wyświetl obraz
        st.image(uploaded_file, caption="Uploaded Image")
        # przekonewertuj do bajtów
        image_bytes = uploaded_file.read()

    # Po wgraniu zdjęcia i wysłaniu klucza wygeneruj opis.
    if gemini_key is None:
        st.warning("Wprowadź klucz do API.")
    if uploaded_file and gemini_key:
        # prompt do zdjęć
        prompt = [
            types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg',
            ),
            'Opisz obraz po polsku jednym zdaniem. Stwórz od jednego do trzech tagów opisujących obraz.'
        ]

        with st.spinner("Generowanie opisu"):
            description = get_gemini_response(prompt)

        st.subheader('Opis zdjęcia:')
        st.write(description)

# chatbot
with tab2:
    #stworzone przez gemini
    faq_data = pd.DataFrame({
        "questions": [
            "Jaka jest polityka zwrotów?",
            "Jak mogę zresetować hasło?",
            "Jakie metody płatności są obsługiwane?",
            "Jak długo trwa dostawa?",
            "Czy mogę zmienić adres dostawy po złożeniu zamówienia?",
            "Co zrobić, jeśli otrzymałem uszkodzony produkt?",
            "Czy oferujecie darmową wysyłkę?",
            "Jak mogę skontaktować się z obsługą klienta?",
            "Czy mogę anulować zamówienie?",
            "Jak śledzić status mojego zamówienia?"
        ],
        "answers": [
            "Zwroty są możliwe w ciągu 14 dni od otrzymania produktu, jeśli jest nienaruszony i w oryginalnym opakowaniu.",
            "Aby zresetować hasło, kliknij 'Zapomniałem hasła' na stronie logowania i postępuj zgodnie z instrukcjami w e-mailu.",
            "Akceptujemy karty kredytowe, przelewy bankowe, Blik oraz PayPal.",
            "Dostawa trwa zwykle od 2 do 5 dni roboczych, w zależności od wybranego przewoźnika.",
            "Zmiana adresu dostawy jest możliwa przed wysłaniem zamówienia. Skontaktuj się z obsługą klienta.",
            "Prosimy zgłosić uszkodzony produkt w ciągu 48 godzin od otrzymania, aby uzyskać zwrot lub wymianę.",
            "Darmowa wysyłka jest dostępna dla zamówień powyżej 200 zł.",
            "Możesz skontaktować się z nami przez e-mail (kontakt@sklep.pl) lub telefon (123-456-789).",
            "Zamówienie można anulować przed wysyłką, kontaktując się z obsługą klienta.",
            "Status zamówienia możesz sprawdzić w sekcji 'Moje zamówienia' po zalogowaniu na swoje konto."
        ]
    })
    st.title("Czatbot do obsługi klienta")
    # sprawdź czy przekonwertowano FAQ
    if 'embeddings_generated' not in st.session_state:
        st.session_state.embeddings_generated = False
    # przycisk, aby nie uruchamiac przetwarzania FAQ przy każdym uruchomieniu aplikacji
    if st.button('Rozpocznij chat z botem.'):
        st.session_state.embeddings_generated = True
    if st.session_state.embeddings_generated:
        faq_embeddings = get_embeddings(faq_data['questions'].to_list())
        st.success('FAQ gotowe')
        # Miejsce na pytanie użytkownika
        if prompt := st.chat_input("Jak mogę pomóc?"):
            # Pytanie
            with st.chat_message("user"):
                st.markdown(prompt)
            faq_context = find_relevant_faq(prompt, faq_embeddings, faq_data)
            # Odpowiedź
            with st.chat_message("assistant"):
                response = query_gemini_rag(prompt, faq_context)
                st.markdown(response)
