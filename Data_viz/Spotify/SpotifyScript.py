import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Load the dataset (replace with your file path)
file_path = 'C:\\Users\\vpark\\Vee\\Github_projects\\Data_viz\\Spotify\\spotify-2023.csv'
spotify_data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Spotify API credentials
CLIENT_ID = '0f7a816e1c5f4962b6b2a16a7a6c9950'
CLIENT_SECRET = '743bc941d51a427c8b48bff3bcf169ce'

# Authenticate with Spotify
auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)

# Function to get album cover URL for each track
def get_album_cover_url(track_name, artist_name):
    query = f"track:{track_name} artist:{artist_name}"
    result = sp.search(query, type='track', limit=1)
    if result['tracks']['items']:
        album_cover_url = result['tracks']['items'][0]['album']['images'][0]['url']
        return album_cover_url
    return None

# Apply the function to each row and create the 'cover_url' column
spotify_data['cover_url'] = spotify_data.apply(lambda row: get_album_cover_url(row['track_name'], row['artist(s)_name']), axis=1)

# Save the updated DataFrame with cover URLs
spotify_data.to_csv('C:\\Users\\vpark\\Vee\\Github_projects\\Data_viz\\Spotify\\spotify-2023.csv', index=False)
print("Updated dataset saved with cover URLs.")
