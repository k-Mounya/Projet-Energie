import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import xgboost
import joblib

st.sidebar.title("Sommaire")
pages = ["Introduction et problématique", "Exploration des données", "Analyse des données", "Modélisation et prédictions"]
page = st.sidebar.radio("Aller vers la page", pages)
pd.set_option('display.max_columns', None)
dfsmp = pd.read_csv(r'C:\Users\tonym\OneDrive\MOSCAT OneDrive\PROFESSIONNEL\DATASCIENTEST\Projet ENERGIE\dfsmp.csv', sep=',', header=0)

if page == pages[0]:
    st.write("### Contexte du projet")
    st.write("La croissance démographique, l’accès d’une part grandissante de la population mondiale à l’énergie, le développement rapide de certaines économies, synonyme d’industrialisation, les pays développés habitués à une énergie abondante et relativement bon marché, sont autant de facteurs contribuant à une hausse continue de la consommation d’énergie.")
    st.write("Le secteur économique de l'énergie en France comprend la production locale et l'importation d'énergie primaire, Pour couvrir les besoins énergétiques de la France, la branche énergétique française utilise de l'énergie primaire, produite en France ou importée, puis la transforme et la distribue aux utilisateurs.")
    st.write("Nous nous intéressons à la production locale, ainsi la France compte dans son bouquet énergétique des énergies fossiles et d’autres renouvelables tels que : le nucléaire, le pétrole, le gaz naturel, des d'énergies renouvelables et déchets.")
    st.write("Le gestionnaire du réseau de transport d'électricité français RTE représente chaque jour, et en temps réel, les données liées à la consommation et production d’électricité sur sa plateforme Eco2Mix.")
    st.write("L'objectif de notre projet consiste à explorer et visualiser les données à partir des données mise à notre disposition à partir de cette plateforme afin de constater le phasage entre la consommation et d'autres paramètres tels que la production énergétique au niveau national et au niveau régional (risque de black-out notamment), les conditions météorologiques ou la densité de population. Dans ce sens nous allons nous focaliser sur :")
    st.write("- L’analyse au niveau régional pour en déduire une prévision de consommation")
    st.write("- L’analyse par filière de production : énergie nucléaire / renouvelable")
    st.write("- Un focus sur les énergies renouvelables et leurs lieux d’implantation.")
    st.write("- Pour y parvenir, nous allons utiliser un ensemble de données d’approximativement 2 millions d’enregistrements. Les données contiennent les informations sur la consommation d’électricité et sa production à partir de plusieurs de plusieurs sources d’énergie : nucléaire, solaire, éolienne, bioénergie, fioul, …  par région métropolitaine (hors corse) enregistrées par demi-heure.")
    st.image(r"C:\Users\tonym\OneDrive\MOSCAT OneDrive\PROFESSIONNEL\DATASCIENTEST\Projet ENERGIE\Images\Image énergies.jpg")

elif page == pages[1]:
    st.write("Notre DataFrame sur les consommations d'énergie par région et par tranche de 3 h")
    pd.set_option('display.max_columns', None)
    dfsmp = pd.read_csv(r'C:\Users\tonym\OneDrive\MOSCAT OneDrive\PROFESSIONNEL\DATASCIENTEST\Projet ENERGIE\dfsmp.csv', sep=',', header=0)
    dfsmp.columns = [col.replace(" ", "\n") for col in dfsmp.columns]
    st.dataframe(dfsmp.head(10), height=300)

    # Checkbox pour afficher les valeurs manquantes
    if st.checkbox('Afficher les régions'):
        st.write(dfsmp['region'].unique())

        # Checkbox pour afficher les valeurs manquantes
    if st.checkbox('Afficher les années'):
        st.write(dfsmp['annee'].unique())
    
    # Checkbox pour afficher le shape du DataFrame
    if st.checkbox('Afficher les colonnes et types'):
        st.write('Colonne et types du DataFrame:')
        st.write(dfsmp.dtypes)

    # Checkbox pour afficher le describe du DataFrame
    if st.checkbox('Afficher le describe'):
        st.write('Description du DataFrame :')
        st.write(dfsmp.describe())

    # Checkbox pour afficher le shape du DataFrame
    if st.checkbox('Afficher le shape'):
        st.write('Forme du DataFrame :')
        st.write(dfsmp.shape)

elif page == pages[2]:
    st.write("### Analyse des Données")

    # Prétraitement des données pour assurer le bon format des colonnes
    dfsmp['annee_mois'] = pd.to_datetime(dfsmp['annee_mois'], format='%Y-%m').dt.to_period('M').astype(str)

    # Création du pivot table pour la heatmap avec la température
    heatmap_data = dfsmp.pivot_table(index='annee_mois', columns='region', values='temperature (C°)', aggfunc='mean')
    st.write("Données de la Heatmap :")
    st.write(heatmap_data)

    # Création de la heatmap avec Plotly
    fig = px.imshow(heatmap_data, 
                    labels=dict(x="Région", y="Année-Mois", color="Température (°C)"),
                    title="Heatmap de la Température par Région et par Année-Mois",
                    aspect="auto",
                    color_continuous_scale='Viridis')  # Utilisation d'une échelle de couleurs contrastée
    
    st.plotly_chart(fig)    
    st.write ("On remarque que la moyenne de température en Provence-Alpes Côtes d'Azur est la plus forte de toutes les régions de France")
    st.write ("Ce sont les régions Hauts de France, Normandie et Bretagne, plus proches du littoral, qui ont les températures les plus faibles en moyenne sur toute la période observée")
    st.write ("Sans surprise, c'est bien pendant les mois d'hiver (Décembre, Janvier, Février) que l'on observe les températures les plus froides, en particulier pour le Grand-Est, plus éloigné du littoral")
    st.write ("******************************************************************************************************************")
    #############################################################################
    df = dfsmp.groupby('annee_mois').agg({
    'temperature (C°)': 'mean',
    'nucl': 'mean'
    }).reset_index()

    # Création du graphique avec Plotly
    fig = go.Figure()

    # Ajouter la courbe pour la température
    fig.add_trace(go.Scatter(x=df['annee_mois'], y=df['temperature (C°)'], mode='lines+markers', name='Température (°C)', yaxis='y1'))

    # Ajouter la courbe pour la production d'électricité
    fig.add_trace(go.Scatter(x=df['annee_mois'], y=df['nucl'], mode='lines+markers', name='Energie nucléaire', yaxis='y2'))

    # Mise à jour de la mise en page du graphique
    fig.update_layout(
    title = 'Graphique en courbe de la température et de la production d énergie nucléaire',
    xaxis_title='Année-Mois',
    yaxis_title='Température (°C)',
    yaxis2=dict(
        title='Production d électricité nucléaire (MWh)',
        overlaying='y',
        side='right'
    ),
    legend=dict(x=0, y=1.0, traceorder='normal')
    )

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)
    st.write ("On remarque pendant toute la période que la température et la production d'énergie nucléaire font les vases communiquant. Quand une variable a des valeurs faibles, l'autre a des valeurs fortes et vice et versa")
    st.write ("******************************************************************************************************************")
    ################################################################

    heatmap_conso = dfsmp.pivot_table(index='annee_mois', columns='region', values='conso', aggfunc='mean')

    # Création de la heatmap avec Plotly
    fig2 = px.imshow(heatmap_conso, 
                    labels=dict(x="Région", y="Année-Mois", color="conso"),
                    title="Heatmap de la Consommation par Région et par Année-Mois",
                    aspect="auto",
                    color_continuous_scale='Inferno')  # Utilisation d'une échelle de couleurs contrastée
    
    st.plotly_chart(fig2)
    st.write ("On remarque que la consommation la plus importante est sur l'Ile de France, ce qui est corrélé avec la population tandis que la plus faible est pour Centre Val de Loire, Bretagne et Bourgogne Franche-Comté")
    st.write ("******************************************************************************************************************")
    ################################################################

    df = dfsmp.groupby('jour_sem').agg({
    'conso': 'mean',
    }).reset_index()

    # Création du graphique avec Plotly
    fig = go.Figure()

    # Ajouter la courbe pour la température
    fig.add_trace(go.Scatter(x=df['jour_sem'], y=df['conso'], mode='lines+markers', name='Consommation', yaxis='y1'))

    # Mise à jour de la mise en page du graphique
    fig.update_layout(
    title = 'Graphique de la consommation en fonction des jours de la semaine',
    xaxis_title='Jour Semaine',
    yaxis_title='Consommation',
    yaxis2=dict(
        title='Consommation en fonction des jours de la semaine',
        overlaying='y',
        side='right'
    ),
    legend=dict(x=0, y=1.0, traceorder='normal')
    )

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)
    st.write ("La consommation tout au long de la semaine, pendant les périodes de travail du Lundi au Vendredi : machines de production, électricité... tandis qu'elle est plus faible le weeke end en particulier le dimanche")
    st.write ("******************************************************************************************************************")
    ################################################################

    df = dfsmp.groupby('heure').agg({'conso': 'mean'}).reset_index()

    # Création du graphique avec Plotly
    fig = go.Figure()

    # Ajouter la courbe pour la consommation avec la couleur rouge écarlate
    fig.add_trace(go.Scatter(x=df['heure'], y=df['conso'], mode='lines+markers', name='Consommation',
                         line=dict(color='red')))  # Définir la couleur de la courbe

    # Mise à jour de la mise en page du graphique
    fig.update_layout(
    title='Graphique de la consommation en fonction des heures',
    xaxis_title='Heure',
    yaxis_title='Consommation',
    yaxis2=dict(
        title='Consommation en fonction des jours de la semaine',
        overlaying='y',
        side='right'
    ),
    legend=dict(x=0, y=1.0, traceorder='normal')
    )

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)
    st.write ("La consommation est à son pic à 12h30 et est très faibles entre minuit et 6h du matin : les heures creuses")
    st.write ("******************************************************************************************************************")
    #############################################################

    df_agg = dfsmp.groupby('region').agg({
    'conso': 'mean',
    'population': 'mean'
    }).reset_index()

    # Création du graphique avec Plotly
    fig = go.Figure()

    # Ajouter la courbe pour la consommation moyenne par région
    fig.add_trace(go.Bar(
    x=df_agg['region'],
    y=df_agg['conso'],
    name='Consommation Moyenne',
    marker_color='blue'
    ))

    # Ajouter la courbe pour la population
    fig.add_trace(go.Scatter(
    x=df_agg['region'],
    y=df_agg['population'],
    mode='lines+markers',
    name='Population',
    yaxis='y2',
    line=dict(color='red', width=2)
    ))

    # Mise à jour de la mise en page du graphique
    fig.update_layout(
    title='Consommation Moyenne et Population par Région',
    xaxis_title='Région',
    yaxis_title='Consommation Moyenne (MWh)',
    yaxis2=dict(
        title='Population',
        overlaying='y',
        side='right'
    ),
    xaxis_tickangle=-45,  # Angle des étiquettes de l'axe des x
    legend=dict(x=0, y=1.0, traceorder='normal')
    )

    st.plotly_chart(fig)
    st.write (" On remarque que la consommation moyenne par population est la plus forte en Ile De France : une personne consomme plus en moyenne en Ile de France que dans les autres régions. Cela peut s'expliquer par la quantité d'infrastructures et d'outils de production à alimenter dans cette région")
    st.write ("Par contre, les régions Centre Val de Loire et Bourgogne Franche Comté ont une population qui consomme moins : encore une fois du fait des instructures et des outils de production gourmands en énergie")
    st.write ("******************************************************************************************************************")
    #############################################################

    df_agg = dfsmp.groupby(['region', 'annee']).agg({'conso': 'sum'}).reset_index()

    # Création du graphique avec Plotly
    fig = go.Figure()

    # Ajouter des barres pour chaque année
    for annee in df_agg['annee'].unique():
        df_year = df_agg[df_agg['annee'] == annee]
        fig.add_trace(go.Bar(
        x=df_year['region'],
        y=df_year['conso'],
        name=f'Année {annee}'
        ))

    # Mise à jour de la mise en page du graphique
    fig.update_layout(
    title='Consommation par Région et par Année',
    xaxis_title='Région',
    yaxis_title='Consommation (MWh)',
    barmode='group',  # Groupement des barres
    xaxis_tickangle=-45,  # Angle des étiquettes de l'axe des x
    legend=dict(x=0, y=1.0, traceorder='normal', orientation = 'h')
    )

    st.plotly_chart(fig)
    st.write ("La consommation par année et région reste assez linéaire pour toutes les régions. On observe même une légère tendance à la baisse. On remarque que l'année 2022 est plus faible car les données s'arrêtent au premier trimestre")
    st.write ("******************************************************************************************************************")
    #############################################################

    df_agg = dfsmp.groupby('region').agg({
    'nucl': 'mean',
    'eol': 'mean',
    'bioen': 'mean',
    'therm': 'mean',
    'sol': 'mean',
    'pomp': 'mean',
    'hydr': 'mean'
    }).reset_index()

    # Création du graphique avec Plotly
    fig = go.Figure()

    # Palette de couleurs (Plasma)
    colors = px.colors.sequential.Plasma  # Utilisation de la palette Plasma

    # Ajouter des barres empilées pour chaque type d'énergie
    for i, energy_type in enumerate(['nucl', 'eol', 'bioen', 'therm', 'sol', 'pomp', 'hydr']):
        fig.add_trace(go.Bar(
            x=df_agg['region'],
            y=df_agg[energy_type],
            name=energy_type,
            marker_color=colors[i % len(colors)]  # Application de la couleur de la palette
            ))

    # Mise à jour de la mise en page du graphique
    fig.update_layout(
    title='Consommation Moyenne par Région pour Chaque Type d\'Énergie',
    xaxis_title='Région',
    yaxis_title='Consommation Moyenne (MWh)',
    barmode='stack',  # Barres empilées
    xaxis_tickangle=-45,  # Angle des étiquettes de l'axe des x
    legend=dict(x=0, y=1.0, traceorder='normal', orientation='h')  # Légende horizontale
    )

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)
    st.write ("C'est très net, la production d'énergie nucléaire est dominante dans toutes les régions de france qui peuvent en produire : Auvergne Rhône Alpes, Centre Val de Loire, Grand Est, Hauts de France, Nouvelle Aquitaine, Normandie et Occitanie")
    st.write ("Les productions d'énergie thermique et hydraulique sont a peu près égales en France et se répartissent sur les régions qui en produisent")
    st.write ("Hydraulique : Auvergne-Rhône Alpes, Grand Est, Nouvelle Aquitaine, Occitanie et Provence Alpes Côtes d'Azur")
    st.write ("Thermique : Grand Est, Hauts de France, Ile deFrance, Normandie, Provence Alpes Côtes d'Azur et Pays de la Loire")
    st.write (" Les régions qui produisent le moins d'énergie : Bourgogne Franche-Comté, Ile de France, Pays de la Loire et Bretagne")
    st.write (" Les régiosn qui produisent le plus d'énergie : Auvergne Rhône Alpes, Centre Val de Loire, Grand Est, Hauts de France, Normandie et Nouvelle Aquitaine")
    st.write ("******************************************************************************************************************")
    #############################################################


elif page == pages[3]:
    st.write("### Modélisation et Prédictions")
    
    # Afficher les résultats des modèles
    pd.set_option('display.max_columns', None)
    result_models = pd.read_csv(r'C:\Users\tonym\OneDrive\MOSCAT OneDrive\PROFESSIONNEL\DATASCIENTEST\Projet ENERGIE\result_models.csv', sep=';', header=0)
    st.write(result_models)

    st.image(r"C:\Users\tonym\OneDrive\MOSCAT OneDrive\PROFESSIONNEL\DATASCIENTEST\Projet ENERGIE\Images\post-gridsearch.png")

    # Charger les résultats sauvegardés
    results_path = r'C:\Users\tonym\OneDrive\MOSCAT OneDrive\PROFESSIONNEL\DATASCIENTEST\Projet ENERGIE\Modèles et résultats JOBLIB\LRresults.pkl'
    results = joblib.load(results_path)
    
    #Afficher les features importances
    st.title("Feature importance Random Forest")
    st.image(r"C:\Users\tonym\OneDrive\MOSCAT OneDrive\PROFESSIONNEL\DATASCIENTEST\Projet ENERGIE\Images\feature importance RandomForest.png")
  
    st.title("Feature importance Decision Tree")
    st.image(r"C:\Users\tonym\OneDrive\MOSCAT OneDrive\PROFESSIONNEL\DATASCIENTEST\Projet ENERGIE\Images\feature importance DecisionTree.png")

    st.title("Feature importance XGB")
    st.image(r"C:\Users\tonym\OneDrive\MOSCAT OneDrive\PROFESSIONNEL\DATASCIENTEST\Projet ENERGIE\Images\feature importance XGB.png")

    # Afficher les images et les explications
    st.title("Shape de Random Forest Regressor")
    st.image(r"C:\Users\tonym\OneDrive\MOSCAT OneDrive\PROFESSIONNEL\DATASCIENTEST\Projet ENERGIE\Images\Shape Random Forest Regressor.png")
    st.write("On remarque que les variables ayant le plus d'impact dans le modèle Random Forest Regressor sont : population, bioen, therm, Température (C°)")
    
    st.title("Shape de Decision Tree Regressor")
    st.image(r"C:\Users\tonym\OneDrive\MOSCAT OneDrive\PROFESSIONNEL\DATASCIENTEST\Projet ENERGIE\Images\Shape Decision Tree Regressor.png")
    st.write("On remarque que les variables ayant le plus d'impact dans le modèle Decision Tree Regressor sont : population, therm, ech_phy")

    st.title("Shape de XGB Regressor")
    st.image(r"C:\Users\tonym\OneDrive\MOSCAT OneDrive\PROFESSIONNEL\DATASCIENTEST\Projet ENERGIE\Images\Shape XGB Regressor.png")
    st.write("On remarque que les variables ayant le plus d'impact dans le modèle XGB Regressor sont : population, bioen, therm, Température (C°)")
    
    st.image(r"C:\Users\tonym\OneDrive\MOSCAT OneDrive\PROFESSIONNEL\DATASCIENTEST\Projet ENERGIE\Images\predic vs reel IDF.png")
    st.write("Île-de-France (IDF) - région avec une forte densité de population et une demande énergétique importante.")
    st.image(r"C:\Users\tonym\OneDrive\MOSCAT OneDrive\PROFESSIONNEL\DATASCIENTEST\Projet ENERGIE\Images\predic vs reel PAC.png")
    st.write("Provence-Alpes-Côte d'Azur (PACA) - région plus ensoleillée et avec des variations de consommation différentes.")
    st.image(r"C:\Users\tonym\OneDrive\MOSCAT OneDrive\PROFESSIONNEL\DATASCIENTEST\Projet ENERGIE\Images\predic vs reel NAQ.png")
    st.write("Nouvelle-Aquitaine (NAQ) - une région avec une répartition plus rurale et des besoins énergétiques différents.")

    st.write("On remarque que les modèles suivent bien les tendances du réel. Tous sous-estiment les valeurs en semaine et surestiment les valeurs du week-end")

    model_path = r"C:\Users\tonym\OneDrive\MOSCAT OneDrive\PROFESSIONNEL\DATASCIENTEST\Projet ENERGIE\Modèles et résultats JOBLIB\Random_Forest_Regressor_model.pkl"
    model = joblib.load(model_path)
    st.session_state.new_data = pd.DataFrame()
    # Conversion de la colonne 'date_heure' en datetime sans format spécifié
    dfsmp['date_heure'] = pd.to_datetime(dfsmp['date_heure'], errors='coerce')
    
    # Obtenez les régions uniques
    regions = dfsmp['region'].unique()

    # Sélection de la région
    selected_region = st.selectbox("Choisis une région", regions)
    date_input = st.date_input("Date", min_value=datetime(2023, 1, 1), max_value=datetime(2100, 12, 31))
    month = date_input.month  # Définir le mois après la sélection de la date
     # Liste des tranches d'heures (chaque tranche de 3 heures)
    time_slots = ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"]
      # Sélection d'une heure avec des tranches de 3 heures
    selected_time = st.selectbox("Choisis une heure (tranches de 3h)", time_slots)
    # Convertir l'heure sélectionnée en nombre d'heures
    hour = int(selected_time.split(":")[0])
    population = st.number_input("Population", min_value=0)
    year=date_input.year
    day=date_input.day
    
    # Filtrage des données
    filtered_data = dfsmp[(dfsmp['region'] == selected_region) & (dfsmp['date_heure'].dt.month == month) & (dfsmp['date_heure'].dt.hour == hour)]
    
    new_data = pd.DataFrame({
            'annee': [year],
            'therm': dfsmp[(dfsmp['region'] == selected_region) & (dfsmp['date_heure'].dt.month == month) & (dfsmp['date_heure'].dt.hour == hour)]['therm'].mean(),
            'nucl': dfsmp[(dfsmp['region'] == selected_region) & (dfsmp['date_heure'].dt.month == month) & (dfsmp['date_heure'].dt.hour == hour)]['nucl'].mean(),
            'eol': dfsmp[(dfsmp['region'] == selected_region) & (dfsmp['date_heure'].dt.month == month) & (dfsmp['date_heure'].dt.hour == hour)]['eol'].mean(),
            'sol': dfsmp[(dfsmp['region'] == selected_region) & (dfsmp['date_heure'].dt.month == month) & (dfsmp['date_heure'].dt.hour == hour)]['sol'].mean(),
            'hydr': dfsmp[(dfsmp['region'] == selected_region) & (dfsmp['date_heure'].dt.month == month) & (dfsmp['date_heure'].dt.hour == hour)]['hydr'].mean(),
            'pomp': dfsmp[(dfsmp['region'] == selected_region) & (dfsmp['date_heure'].dt.month == month) & (dfsmp['date_heure'].dt.hour == hour)]['pomp'].mean(),
            'bioen': dfsmp[(dfsmp['region'] == selected_region) & (dfsmp['date_heure'].dt.month == month) & (dfsmp['date_heure'].dt.hour == hour)]['bioen'].mean(),
            'ech_phy': dfsmp[(dfsmp['region'] == selected_region) & (dfsmp['date_heure'].dt.month == month) & (dfsmp['date_heure'].dt.hour == hour)]['ech_phy'].mean(),
            'stock_bat': dfsmp[(dfsmp['region'] == selected_region) & (dfsmp['date_heure'].dt.month == month) & (dfsmp['date_heure'].dt.hour == hour)]['stock_bat'].mean(),
            'destock_bat': dfsmp[(dfsmp['region'] == selected_region) & (dfsmp['date_heure'].dt.month == month) & (dfsmp['date_heure'].dt.hour == hour)]['destock_bat'].mean(),
            'eol_terr': dfsmp[(dfsmp['region'] == selected_region) & (dfsmp['date_heure'].dt.month == month) & (dfsmp['date_heure'].dt.hour == hour)]['eol_terr'].mean(),
            'eol_off': dfsmp[(dfsmp['region'] == selected_region) & (dfsmp['date_heure'].dt.month == month) & (dfsmp['date_heure'].dt.hour == hour)]['eol_off'].mean(),
            'pression_niv_mer (Pa)': dfsmp[(dfsmp['region'] == selected_region) & (dfsmp['date_heure'].dt.month == month) & (dfsmp['date_heure'].dt.hour == hour)]['pression_niv_mer (Pa)'].mean(),
            'vitesse du vent moyen 10 mn (m/s)': dfsmp[(dfsmp['region'] == selected_region) & (dfsmp['date_heure'].dt.month == month) & (dfsmp['date_heure'].dt.hour == hour)]['vitesse du vent moyen 10 mn (m/s)'].mean(),
            'temperature (C°)': dfsmp[(dfsmp['region'] == selected_region) & (dfsmp['date_heure'].dt.month == month) & (dfsmp['date_heure'].dt.hour == hour)]['temperature (C°)'].mean(),
            'humidite (%)': dfsmp[(dfsmp['region'] == selected_region) & (dfsmp['date_heure'].dt.month == month) & (dfsmp['date_heure'].dt.hour == hour)]['humidite (%)'].mean(),
            'pression station (Pa)': dfsmp[(dfsmp['region'] == selected_region) & (dfsmp['date_heure'].dt.month == month) & (dfsmp['date_heure'].dt.hour == hour)]['pression station (Pa)'].mean(),
            'precipitations dans les 3 dernieres heures (mm)': dfsmp[(dfsmp['region'] == selected_region) & (dfsmp['date_heure'].dt.month == month) & (dfsmp['date_heure'].dt.hour == hour)]['precipitations dans les 3 dernieres heures (mm)'].mean(),
            'population': [population],
            'region_FR-ARA': [1 if selected_region == 'FR-ARA' else 0],
            'region_FR-BFC': [1 if selected_region == 'FR-BFC' else 0],
            'region_FR-BRE': [1 if selected_region == 'FR-BRE' else 0],
            'region_FR-CVL': [1 if selected_region == 'FR-CVL' else 0],
            'region_FR-GES': [1 if selected_region == 'FR-GES' else 0],
            'region_FR-HDF': [1 if selected_region == 'FR-HDF' else 0],
            'region_FR-IDF': [1 if selected_region == 'FR-IDF' else 0],
            'region_FR-NAQ': [1 if selected_region == 'FR-NAQ' else 0],
            'region_FR-NOR': [1 if selected_region == 'FR-NOR' else 0],
            'region_FR-OCC': [1 if selected_region == 'FR-OCC' else 0],
            'region_FR-PAC': [1 if selected_region == 'FR-PAC' else 0],
            'region_FR-PDL': [1 if selected_region == 'FR-PDL' else 0],
            'cos_heure': [np.cos(2 * np.pi * hour / 24)],
            'sin_heure': [np.sin(2 * np.pi * hour / 24)],
            'jour_sin': [np.sin(2 * np.pi * day / 365)],
            'jour_cos': [np.cos(2 * np.pi * day / 365)],
            'jour_sem_sin': [np.sin(2 * np.pi * (date_input.weekday() + 1) / 7)],
            'jour_sem_cos': [np.cos(2 * np.pi * (date_input.weekday() + 1) / 7)],
            'mois_sin': [np.sin(2 * np.pi * month / 12)],
            'mois_cos': [np.cos(2 * np.pi * month / 12)]
        })
    
        # Bouton pour ajouter ou écraser les données
    if st.button("Prédire la consommation"):
        predicted_conso = model.predict(new_data)
        # Affichage du résultat
        st.write(f"Consommation énergétique prédite: {predicted_conso[0]:.2f} MW")

    # Affichage des données ajoutées
    st.write("Données ajoutées:")
    st.write(new_data)
