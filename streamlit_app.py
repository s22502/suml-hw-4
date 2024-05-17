import streamlit as st
import pandas as pd
import time
import matplotlib as plt
import os
from transformers import pipeline

st.success('Gratulacje! Z powodzeniem uruchomiłeś aplikację')

st.title('Homework SUML - Streamlit - s22502')

st.image('./assets/translation.png')

st.header('Opis')
st.write('Aplikacja służy do tłumaczenia tekstu z języka angielskiego na język niemiecki oraz analizy wydźwięku emocjonalnego tekstu.')

st.header('Instrukcja')
st.write('Z listy wybierz jedną z dwóch opcji: "Wydźwięk emocjonalny tekstu (eng)" lub "Tłumaczenie z języka angielskiego na niemiecki". Następnie wpisz tekst w odpowiednie pole i wciśnij "CTRL + ENTER". Wynik pojawi się poniżej.')

st.header('Przetwarzanie języka naturalnego')

option = st.selectbox(
    "Opcje",
    [
        "Wydźwięk emocjonalny tekstu (eng)",
        "Tłumaczenie z języka angielskiego na niemiecki",
    ],
)

if option == "Wydźwięk emocjonalny tekstu (eng)":
    text = st.text_area(label="Wpisz tekst")
    if text:
        classifier = pipeline("sentiment-analysis")
        answer = classifier(text)
        st.write(answer)
elif option == "Tłumaczenie z języka angielskiego na niemiecki":
    text = st.text_area(label="Wpisz tekst po angielsku")
    if text:
        st.spinner()
        with st.spinner(text='Trwa tłumaczenie...'):
            classifier = pipeline("translation", model="google-t5/t5-small")
            answer = classifier(text)
            if 'translation_text' in answer[0]:
                if answer[0]['translation_text'] == text:
                    st.error('Błąd. Nie udało się przetłumaczyć tekstu. Prawdopodobnie wpisany tekst nie jest po angielsku.')
                else:
                    st.success('Udało się! Tłumaczenie poniżej!')
                    st.write(answer[0])
            else:
                st.error('Nieznany błąd')

st.subheader('Aplikacja stworzona przez: Barnaba Gańko - s22502')