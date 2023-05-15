import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from spotipy import SpotifyClientCredentials, Spotify
from spotipy.util import prompt_for_user_token
import os
import requests
from re import sub, IGNORECASE
import librosa
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from pandas import DataFrame
#import pandas as pd
from python_tsp.heuristics import solve_tsp_simulated_annealing
from python_tsp.distances import euclidean_distance_matrix

from sklearn.preprocessing import MaxAbsScaler

outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'out')

def get_playlist_samples_no_limit(sp: Spotify, playlist_id: str, verbose: bool = False):
    print('Getting playlist data...')
    playlist_name = sp.playlist(playlist_id, fields='name')['name']
    r = sp.playlist_tracks(playlist_id)
    t = r['items']
    songs = []
    while r['next']:
        r = sp.next(r)
        t.extend(r['items'])
    for song in t:
        try:
            name = song['track']['name']
            artist = song['track']['album']['artists'][0]['name']
            id = song['track']['id']
            preview_url = song['track']['preview_url']
            filename = f'{sub("[^A-Z0-9]", "", song["track"]["name"], 0, IGNORECASE)}.mp3'

            songs.append(
                {
                    'name': name,
                    'artist': artist,
                    'id': id,
                    'preview_url': preview_url,
                    'filename': filename,
                })
        except:
            if verbose:
                print('Error finding Spotify data for track.')

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    print('Getting Spotify previews...', end='\r', flush=True)
    for index, song in enumerate(songs):
        print(f'Getting Spotify previews... [{index + 1} / {len(songs)}]', end='\r', flush=True)
        try: 
            r = requests.get(song['preview_url'])
            if verbose:
                print(f'\nFound preview for track {song["name"]}')
            if not os.path.exists(f'{outdir}/{song["filename"]}'):
                if verbose:
                    print(f'Downloading preview for track {song["name"]}')
                open(f'{outdir}/{song["filename"]}', 'wb').write(r.content)
            else:
                if verbose:
                    print(f'Found existing preview for track {song["name"]}')

        except:
            if verbose:
                print(f'\nNo preview available for track {song["name"]}')
 
    print() 
    songs = [song for song in songs if song['preview_url'] != None]
    return songs, playlist_name


def generate_embedding(songs: list[dict], n_dims: int = 3, verbose: bool = False):
    import logging
    logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
    print('Calculating MFCCs...', end='\r', flush=True)
    if n_dims < 2 or n_dims > 3:
        raise Exception("Number of dimensions for T-SNE must be 2 or 3.")

    matrix = []
    for index, song in enumerate(songs):
        print(f'Calculating MFCCs... [{index + 1} / {len(songs)}]', end='\r', flush=True)
        try:
            if(verbose):
                print(f'Loading song {song["name"]}')   
            if os.path.exists(f'{outdir}/{song["filename"]}.mfcc.npy'):
                mfcc = np.load(f'{outdir}/{song["filename"]}.mfcc.npy')       
            else:  
                mp3 = librosa.load(os.path.join(outdir, song['filename']))
                mfcc = np.array(librosa.amplitude_to_db(
                    librosa.feature.melspectrogram(
                        y=mp3[0],
                        sr=mp3[1],
                        n_mels=96
                    ),
                    ref=1.0
                )).flatten()
                np.save(f'{outdir}/{song["filename"]}.mfcc', mfcc)

            #TODO: FIGURE OUT WHY THIS HAPPENS
            if(len(mfcc) > 122880):
                mfcc = mfcc[:122880]
            logging.debug(len(mfcc))
            matrix.append(mfcc)
        except:
            if verbose:
                print('Error loading song and converting to MFCC')

    print()
    print('Generating T-SNE embedding...')
    perplexity = 15 #len(matrix) /  #if len(matrix) / 2 < 50 else 50
    embedding = TSNE(n_components=n_dims, learning_rate='auto', init='pca', perplexity=perplexity, n_iter=5000, n_iter_without_progress=500).fit_transform(np.array(matrix))

    for i, e in enumerate(embedding):
        songs[i]['x_pos'] = e[0]
        songs[i]['y_pos'] = e[1]
        if(n_dims == 3):
            songs[i]['z_pos'] = e[2]

    return DataFrame(songs)

def plot_embedding(df: DataFrame, method: str = 'show'):

    if method != 'show' and method != 'save':
        raise Exception('Invalid display method. Use either save or show.')
    
    fig = plt.figure()

    try:
        if 'z_pos' in df.columns:
            ax = fig.add_subplot(projection='3d')
            for artist, group_index in df.groupby('artist').groups.items():
                x = df.loc[group_index,'x_pos']
                y = df.loc[group_index,'y_pos']
                z = df.loc[group_index,'z_pos']
                names = df.loc[group_index, 'name']

                ax.scatter3D(x,y,z, label=artist)
                for i, txt in enumerate(names):
                    ax.text(x.iloc[i], y.iloc[i], z.iloc[i], txt)

            if 'playlist_position' in df.columns:
                print('Drawing TSP lines')
                x = df.sort_values('playlist_position')['x_pos'].tolist()
                y = df.sort_values('playlist_position')['y_pos'].tolist()
                z = df.sort_values('playlist_position')['z_pos'].tolist()
                ax.plot3D(x, y, z, 'blue')

            ax.legend()
        else:
            for artist, group_index in df.groupby('artist').groups.items():
                x = df.loc[group_index,'x_pos']
                y = df.loc[group_index,'y_pos']
                names = df.loc[group_index, 'name']

                plt.scatter(x, y, label=artist)
                for i, txt in enumerate(names):
                    plt.annotate(txt, (x.iloc[i], y.iloc[i]))

            if 'playlist_position' in df.columns:
                print('Drawing TSP lines')
                x = df.sort_values('playlist_position')['x_pos'].tolist()
                y = df.sort_values('playlist_position')['y_pos'].tolist()
                plt.plot(x, y, ',', linestyle='--')

            plt.legend()

        if method == 'show':
            plt.show()
        elif method == 'save':
            plt.savefig(f'embedding.png')

    except:
        raise Exception("Invalid number of dimensions.")
    
def solve_tsp(df: DataFrame, open_loop: bool = False, verbose: bool = False):
    print('Calculating optimal loop...')
    positions = []
    for _, song in df.iterrows():
        try:
            positions.append([song['x_pos'], song['y_pos'], song['z_pos']])
        except:
            positions.append([song['x_pos'], song['y_pos']])
    
    if verbose:
        print('Calculating distance matrix...')
    distance_matrix = euclidean_distance_matrix(np.array(positions))
    if open_loop:
        distance_matrix[:, 0] = 0
    if verbose:
        print('Solving TSP...')
    permutation, _ = solve_tsp_simulated_annealing(distance_matrix)
    df['playlist_position'] = ''

    for index, order in enumerate(permutation):
        df.at[order, 'playlist_position'] = index

    return df

def create_new_playlist(sp: Spotify, username: str, playlist_name: str, df: DataFrame):
    new_playlist = sp.user_playlist_create(user=username, name=playlist_name)
    tracks = df.sort_values('playlist_position')['id'].tolist()
    while tracks:
        sp.user_playlist_add_tracks(username, new_playlist['id'], tracks[:100], position=None)
        tracks = tracks[100:]

def main(verbose: bool = False):
    sp = Spotify(client_credentials_manager=SpotifyClientCredentials(client_id='28b5ee3660094d4283a71522a9a671e7',client_secret='240ff8ddeed34835950652bfc2b3ad49'))
    username = ''
    while username == '':
        username = input("Enter your Spotify username: ")
        try:
            sp.user(username)
        except:
            username = ''

    try:
        token = prompt_for_user_token(username, scope='playlist-read-private playlist-modify-private playlist-modify-public',client_id='28b5ee3660094d4283a71522a9a671e7',client_secret='240ff8ddeed34835950652bfc2b3ad49', redirect_uri='http://localhost:8888/callback')
        sp = Spotify(auth=token)
    except:
        raise Exception('Couldn\'t log in to Spotify.')
    
    playlists = []
    for index, p in enumerate(sp.user_playlists(username)['items']):
        print(f'{index + 1}) {p["name"]}')
        playlists.append(p['id'])
    
    try:
        selected_playlist = playlists[int(input('Enter the number of the playlist you would like to SmartLoop: ')) - 1]
    except:
        raise Exception('Invalid playlist selection.')
    
    try:
        n_dims = int(input('Enter the number of dimensions you want the T-SNE reduction to output (2 or 3): '))
        if n_dims < 2 or n_dims > 3:
            raise Exception()
    except:
        raise Exception('Invalid number of dimensions.')
    
    songs, playlist_name = get_playlist_samples_no_limit(sp=sp, playlist_id=selected_playlist, verbose=verbose)
    songs_embed = generate_embedding(songs=songs, n_dims=n_dims, verbose=verbose)
    df = solve_tsp(songs_embed, verbose=verbose)
    if verbose:
        print(df.sort_values('playlist_position').to_string())
    plot_embedding(df, method='show')
    create_new_playlist(sp=sp, username=username, playlist_name=f'{playlist_name} SmartLoop', df=df)


if __name__ == '__main__':
    main(verbose=False)