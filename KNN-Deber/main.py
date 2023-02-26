# Control de Cambios: 11 Febrero 2023
# Developer Cambios: GMEJIA

# para Ejecutar en el Command Prompt: streamlit run main.py

# Libreria Framework para Presentacion en Web - APP Server
import streamlit as st

# Libreria para el Calculo del tiempo Transcurrido en la Ejecucion
import time

# Libreria para Uso de Imagenes
from PIL import Image

# Librería para el manejo y análisis de estructuras de datos
import pandas as pd

# Librerias para hacer análisis predictivo, clasificadores, algoritmos de clusterización
from sklearn.neighbors import NearestNeighbors

# Libreria para visualizar gráficos de forma inmediata
import plotly.express as px
import streamlit.components.v1 as components

# ========================================================================================================
# Funciones del Programa
## Funcion para la carga de los datos en la Variable de Salida
# @st.cache(allow_output_mutation=True)
@st.cache_data
def load_data():
    #df = pd.read_csv("DataSet/filtered_track_df.csv")
    #df = pd.read_csv("DataSet/filtered_track_df_10K.csv")
    #df = pd.read_csv("DataSet/filtered_track_df_20K.csv")
    #df = pd.read_csv("DataSet/filtered_track_df_30K.csv")
    #df = pd.read_csv("DataSet/filtered_track_df_40K.csv")
    df = pd.read_csv("DataSet/filtered_track_df_50K.csv")
    # dataset de entrenamiento.
    df['genres'] = df.genres.apply(lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
    exploded_track_df = df.explode("genres")
    return exploded_track_df

## Funcion para el Entrenamiento - observación - predicción
def n_neighbors_uri_audio(genre, start_year, end_year, test_feat):
    genre = genre.lower()
    # Llenado de las variables para nueva observación
    genre_data = exploded_track_df[
        (exploded_track_df["genres"] == genre) & (exploded_track_df["release_year"] >= start_year) & (
                exploded_track_df["release_year"] <= end_year)]
    genre_data = genre_data.sort_values(by='popularity', ascending=False)[:500]
    neigh = NearestNeighbors()
    # entrenamiento del modelo
    neigh.fit(genre_data[audio_feats].to_numpy())
    # predicción de la clase
    n_neighbors = neigh.kneighbors([test_feat], n_neighbors=len(genre_data), return_distance=False)[0]
    uris = genre_data.iloc[n_neighbors]["uri"].tolist()
    audios = genre_data.iloc[n_neighbors][audio_feats].to_numpy()
    return uris, audios

def n_neighbors_uri_audio_Name(genre, start_year, end_year, test_feat):
    genre = genre.lower()
    # Llenado de las variables para nueva observación
    genre_data = exploded_track_df[
        (exploded_track_df["genres"] == genre) & (exploded_track_df["release_year"] >= start_year) & (
                exploded_track_df["release_year"] <= end_year)]
    genre_data = genre_data.sort_values(by='popularity', ascending=False)[:500]
    neigh = NearestNeighbors()
    # entrenamiento del modelo
    neigh.fit(genre_data[audio_feats].to_numpy())
    # predicción de la clase
    n_neighbors = neigh.kneighbors([test_feat], n_neighbors=len(genre_data), return_distance=False)[0]
    uris = genre_data.iloc[n_neighbors]["uri"].tolist()
    audios = genre_data.iloc[n_neighbors][audio_feats].to_numpy()
    names = genre_data.iloc[n_neighbors]["name"].tolist()
    linkWeb = genre_data.iloc[n_neighbors]["preview_url"].tolist()
    return uris, audios, names, linkWeb

# ========================================================================================================
# Creacion de la Interface Visual
st.set_page_config(page_title="Sistema para Recomendación de Musica", layout="wide")

# Carga de Imagen
image = Image.open('Imagenes/logo.JPG')
st.image(image, caption='Logo UEES')

title = "Algoritmo para Recomendación de Musica, Basado en: K-Nearest neighbors"
st.title(title)

st.write("Materia: ANÁLISIS DE ALGORITMOS")
st.write("FACULTAD DE INGENIERÍA ESCUELA DE SISTEMAS Y TELECOMUNICACIONES")
st.write("GRUPO 3:")
st.write("GONZALO EUGENIO MEJIA ALCIVAR, gmejia@uees.edu.ec")
st.write("CHRISTIAN FRANCISCO GOMEZ, christian.gomez@uees.edu.ec")
st.write("EDWIN ALEJANDRO BAJANA, edwin.bajana@uees.edu.ec")
st.write("JAVIER ISAIAS GRACIA MOREIRA, javier.gracia@uees.edu.ec")
st.write("WASHINGTON ANDRADE MUÑOZ, washndrade@uees.edu.ec")
st.write("----------------------------------------------------------------------------------------")
st.write("Proyecto: Implementacion de Algoritmo <vecinos más cercanos K-NN>, K-Nearest neighbors")

genre_names = ['Dance Pop', 'Electronic', 'Electropop', 'Hip Hop', 'Jazz', 'K-pop', 'Latin', 'Pop', 'Pop Rap', 'R&B',
               'Rock']
audio_feats = ["acousticness", "danceability", "energy", "instrumentalness", "valence", "tempo"]

exploded_track_df = load_data()

#exploded_track_df.head(10)

st.write("!El Usuario puede personalizar su gusto musical según el género y caracteristicas de audio.")
st.write("El Algotimo K-NN, genera recomendaciones con los diferentes parametros seleccionados.")
st.markdown("##")
st.write(" 1.- DataFrame <DataSet>, Registros ", exploded_track_df.shape[0], " Cargados para el Aprendizaje!")
TotalDataSet = " - Registros Procesados en DataSet => " + str(exploded_track_df.shape[0])
st.dataframe(exploded_track_df)

with st.container():
    col1, col2, col3, col4 = st.columns((2, 0.5, 0.5, 0.5))
    with col3:
        st.markdown("##")
        st.markdown("Elija el género de música preferido:")
        genre = st.radio(
            "",
            genre_names, index=genre_names.index("Pop"))
    with col1:
        st.write("----------------------------------------------------------------------------------------")
        st.markdown("2.- Defina los parametros para Observacion del Modelo y Recomendacion K-NN ")
        start_year, end_year = st.slider(
            'A.- Rango de años:',
            1990, 2020, (2001, 2005)
        )
        acousticness = st.slider(
            'B.- Acústica',
            0.0, 1.0, 0.5)
        danceability = st.slider(
            'C.- Bailabilidad',
            0.0, 1.0, 0.5)
        energy = st.slider(
            'D.- Energía',
            0.0, 1.0, 0.5)
        instrumentalness = st.slider(
            'E.- Instrumentalidad',
            0.0, 1.0, 0.0)
        valence = st.slider(
            'F.- Valencia',
            0.0, 1.0, 0.45)
        tempo = st.slider(
            'G.- Tempo',
            0.0, 244.0, 118.0)

num = st.number_input("Cantidad de Recomendaciones a Visualizar", min_value=4, max_value=50)

st.write("----------------------------------------------------------------------------------------")
if st.button("3.- Ejecuta Proceso de Recomendaciones  K-NN!"):
    start_time = time.time()
    #tracks_per_page = 50
    st.write("----------------------------------------------------------------------------------------")
    tracks_per_page = num
    test_feat = [acousticness, danceability, energy, instrumentalness, valence, tempo]
    #uris, audios = n_neighbors_uri_audio(genre, start_year, end_year, test_feat)
    uris, audios, names, linkWeb = n_neighbors_uri_audio_Name(genre, start_year, end_year, test_feat)
    tracks = []
    elapsed_time = time.time() - start_time
    print("-- Tiempo Transcurrido: %0.10f segundos." % elapsed_time)
    T1 = ("%0.05f" % elapsed_time)
    st.success("-- Tiempo Transcurrido: %0.10f segundos." + str(elapsed_time) + TotalDataSet, icon="✅")
    st.success("-- Tiempo Transcurrido: %0.05f segundos." + str(T1) + TotalDataSet, icon="✅")

    # Declaracion de la Variables Arreglo DATASET para ver el BIG(o)
    n_values = [10000, 20000, 30000, 40000, 50000]
    t_values = [0.00698, 0.00798, 0.00898, 0.00997, 0.01097]

# Presentacion del Resultado
    with st.container():
        col1, col2, col3 = st.columns((1, 0.5, 2))
        with col1:
            st.write("4.- Resultado Recomendaciones Link:", len(linkWeb))
            st.dataframe(linkWeb)
        with col2:
            st.write("==== Nombre de Cancion:", len(names))
            st.dataframe(names)
        with col3:
            st.write("==== DataSet Caracteristica del Audio:", len(audios))
            st.dataframe(audios)

# llena Arreglo de Links, para visualizacion
    tracks = []
    for uri in uris:
        track = """<iframe src="https://open.spotify.com/embed/track/{}" width="260" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>""".format(
            uri)
        tracks.append(track)

    if 'previous_inputs' not in st.session_state:
        st.session_state['previous_inputs'] = [genre, start_year, end_year] + test_feat

    current_inputs = [genre, start_year, end_year] + test_feat
    if current_inputs != st.session_state['previous_inputs']:
        if 'start_track_i' in st.session_state:
            st.session_state['start_track_i'] = 0
        st.session_state['previous_inputs'] = current_inputs

    if 'start_track_i' not in st.session_state:
        st.session_state['start_track_i'] = 0

# Presentar los resultados en el Contenedor
    with st.container():
        col1, col2, col3 = st.columns([2, 1, 2])
        current_tracks = tracks[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
        current_audios = audios[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
        if st.session_state['start_track_i'] < len(tracks):
            num = 0
            for i, (track, audio) in enumerate(zip(current_tracks, current_audios)):
                if i % 2 == 0:
                    with col1:
                        with st.expander("Gráfico de radar =>" + names[num]):
                            st.write('Cancion:', names[num])
                            st.markdown(linkWeb[num], unsafe_allow_html=True)
                            #st.markdown("5.- Recomendacion PK:", current_tracks)
                            df = pd.DataFrame(dict(
                                r=audio[:5],
                                theta=audio_feats[:5]))
                            # https://python-charts.com/es/ranking/grafico-radar-plotly/
                            # fig = px.line_polar(df, r='r', theta='theta', line_close=True, color_discrete_sequence = ['yellow'])
                            fig = px.line_polar(df, r='r', theta='theta', line_close=True,
                                                color_discrete_sequence=['red'])
                            fig.update_traces(fill='toself')
                            fig.update_layout(height=400, width=340)
                            st.plotly_chart(fig)
                            # Visualizacion del Link en HTML
                            #components.html(track, height=400,)
                else:
                    with col3:
                        with st.expander("Gráfico de radar =>" + names[num]):
                            st.write('Cancion:', names[num])
                            st.markdown(linkWeb[num], unsafe_allow_html=True)
                            #st.write(st.session_state['start_track_i'])
                            df = pd.DataFrame(dict(
                                r=audio[:5],
                                theta=audio_feats[:5]))
                            fig = px.line_polar(df, r='r', theta='theta', line_close=True)
                            fig.update_traces(fill='toself')
                            fig.update_layout(height=400, width=340)
                            st.plotly_chart(fig)
                            # Visualizacion del Link en HTML
                            #components.html(track, height=400, )
                num = num + 1
        else:
            st.write("====== No quedan MAS canciones para recomendar ======")
