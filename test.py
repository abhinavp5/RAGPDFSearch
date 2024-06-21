import streamlit as st
from annotated_text import annotated_text

def displayResults1(text, matches):
    st.subheader("These were the results in response to your query:")
    st.write("Matched Phrases: " + ', '.join(matches))  # Display matches

    # To store text segments to be displayed with annotations
    segments = []
    last_index = 0  # To keep track of the last processed index

    # Function to find all start indices of a substring in a string
    def find_all_indices1(text, substring):
        start = 0
        while start < len(text):
            start = text.find(substring, start)
            if start == -1:
                break
            yield start
            start += len(substring)  # Move past the current match

    # Collect all matches with their start indices
    all_matches = []
    for match in matches:
        all_matches.extend([(match, idx) for idx in find_all_indices1(text, match)])

    # Sort all matches by their start indices
    sorted_matches = sorted(all_matches, key=lambda x: x[1])

    # Iterate over sorted matches and build segments
    for match, index in sorted_matches:
        if index >= last_index:  # Ensure no overlapping or repeated highlights
            if index > last_index:
                segments.append(text[last_index:index])  # Add text before the match
            segments.append((match, "", "#8cff66"))  # Add highlighted match
            last_index = index + len(match)

    # Add any remaining text after the last match
    if last_index < len(text):
        segments.append(text[last_index:])

    # Display the annotated text with highlights
    annotated_text(*segments)
