import streamlit as st
import pandas as pd


# Read the CSV file
df = pd.read_csv("data_moods.csv")

# Drop unnecessary columns
df = df.drop(columns=['name', 'album', 'artist', 'release_date'])

# Streamlit App
def main():
    st.title("Mood-Based Song Selector")
    
    # Sidebar for user input
    x = st.sidebar.selectbox("Select Mood:", ["Energetic", "Sad", "Happy", "Calm"])
    
    # Filter the DataFrame based on the selected mood
    filtered_df = df[df['mood'].str.lower() == x.lower()]
    
    # Sample 5 songs from the filtered DataFrame
    sampled_songs = filtered_df.sample(n=5)
    
    # Display the sampled songs as a description
    st.header("Selected Mood:")
    st.subheader(x)
    st.header("Sampled Songs:")
    
    songs_description = ""
    
    for index, row in sampled_songs.iterrows():
        songs_description += f"**{row['id']}**: [Listen on Spotify]({row['spotify_link']})\n\n"
        copy_button = st.button(f"Copy {row['id']}", key=row['id'])
        if copy_button:
            pyperclip.copy(row['id'])
            st.success(f"📋 {row['id']} copied to clipboard!")

    st.markdown(songs_description, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
