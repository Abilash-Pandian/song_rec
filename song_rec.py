import streamlit as st
import pandas as pd
import pyperclip

# Read the CSV file
df = pd.read_csv("C:\\Users\\Abilash Pandian\\OneDrive\\Desktop\\DATASETS\\data_moods.csv")

# Drop unnecessary columns
df = df.drop(columns=['name', 'album', 'artist', 'release_date'])

# Streamlit App
def main():
    st.title("Mood-Based Song Selector")
    
    # Sidebar for user input
    x = st.sidebar.selectbox("Select Mood:", ["Energetic", "Sad", "Happy", "Calm"])
    
    # Filter the DataFrame based on the selected mood
    filtered_df = df[df['mood'] == x]
    
    # Sample 5 songs from the filtered DataFrame
    sampled_songs = filtered_df.sample(n=5)
    
    # Display the sampled songs
    st.write("Selected Mood:", x)
    st.write("Sampled Songs:")
    st.dataframe(sampled_songs[['id']])
    
    for song_id in sampled_songs['id']:
        st.write(f"{song_id}  ")
        copy_button = st.button(f"Copy {song_id}", key=song_id)
        if copy_button:
            pyperclip.copy(song_id)
            st.success(f"{song_id} copied to clipboard!")

if __name__ == "__main__":
    main()
