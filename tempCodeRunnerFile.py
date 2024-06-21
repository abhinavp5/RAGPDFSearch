import streamlit as st
from annotated_text import annotated_text

def app():
    text = "Hello my name is Bob and I like to go on very long walks"
    matches = ["I like", " long walks", "Hello", "my name" ]
    
    # Finding start indexes of matches and sorting them
    match_indexes = [(match, text.index(match)) for match in matches if match in text]
    sorted_indexes = sorted(match_indexes, key=lambda index: index[1])

    temp = 0
    segments = []  # To store text segments to be displayed with annotations
    for match, index in sorted_indexes:
        if index > temp:
            segments.append(text[temp:index])
        segments.append((match, "", "#8cff66"))
        temp = index + len(match)

    if temp < len(text):
        segments.append(text[temp:])

    annotated_text(*segments)


if __name__ == '__main__':
    app()
