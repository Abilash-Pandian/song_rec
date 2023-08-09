import streamlit as st
import pandas as pd

df = pd.read_csv("data_moods.csv")
df = df.drop(columns=['name', 'album', 'artist', 'release_date'])

def main():
    st.title("Mood-Based Song Selector")
    x=st.sidebar.selectbox("Select Mood:",["Energetic","Sad","Happy","Calm"])
    filtered_df = df[df['mood'] == x]
    sampled_songs = filtered_df.sample(n=5)
    st.write("Selected Mood:", x)
    st.write("Spotify id for 5",x,"songs")
    st.dataframe(sampled_songs[['id']])
    for song_id in sampled_songs['id']:
        st.write(f"{song_id}  ")
        copy_button = st.button(f"Copy {song_id}", key=song_id)
        if copy_button:
            pyperclip.copy(song_id)
            st.success(f"{song_id} copied to clipboard!")
if __name__ == "__main__":
    main()
