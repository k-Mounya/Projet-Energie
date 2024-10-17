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
pages = ["Introduction et problématique", "Exploration des données", "Analyse des données", "Modélisation et Prédictions","Conclusion et Perspectives"]
page = st.sidebar.radio("Aller vers la page", pages)
pd.set_option('display.max_columns', None)
dfsmp = pd.read_csv('dfsmp.csv', sep=',', header=0)

if page == pages[0]:
    #st.write("### Contexte du projet")
    st.markdown("<h1 style='text-align: center;'>Contexte du projet</h1>", unsafe_allow_html=True)
    st.write("---")

    st.write("La croissance démographique, l’accès d’une part grandissante de la population mondiale à l’énergie, le développement rapide de certaines économies, synonyme d’industrialisation, les pays développés habitués à une énergie abondante et relativement bon marché, sont autant de facteurs contribuant à une hausse continue de la consommation d’énergie.")
    st.write("Le secteur économique de l'énergie en France comprend la production locale et l'importation d'énergie primaire, Pour couvrir les besoins énergétiques de la France, la branche énergétique française utilise de l'énergie primaire, produite en France ou importée, puis la transforme et la distribue aux utilisateurs.")
    st.write("Nous nous intéressons à la production locale, ainsi la France compte dans son bouquet énergétique des énergies fossiles et d’autres renouvelables tels que : le nucléaire, le pétrole, le gaz naturel, des d'énergies renouvelables et déchets.")
    st.write("Le gestionnaire du réseau de transport d'électricité français RTE représente chaque jour, et en temps réel, les données liées à la consommation et production d’électricité sur sa plateforme Eco2Mix.")
    st.write("L'objectif de notre projet consiste à explorer et visualiser les données à partir des données mise à notre disposition à partir de cette plateforme afin de constater le phasage entre la consommation et d'autres paramètres tels que la production énergétique au niveau national et au niveau régional (risque de black-out notamment), les conditions météorologiques ou la densité de population. Dans ce sens nous allons nous focaliser sur :")
    st.write("- L’analyse au niveau régional pour en déduire une prévision de consommation")
    st.write("- L’analyse par filière de production : énergie nucléaire / renouvelable")
    st.write("- Un focus sur les énergies renouvelables et leurs lieux d’implantation.")
    st.write("- Pour y parvenir, nous allons utiliser un ensemble de données d’approximativement 2 millions d’enregistrements. Les données contiennent les informations sur la consommation d’électricité et sa production à partir de plusieurs de plusieurs sources d’énergie : nucléaire, solaire, éolienne, bioénergie, fioul, …  par région métropolitaine (hors corse) enregistrées par demi-heure.")
    st.image("Images/Image énergies.jpg")

elif page == pages[1]:
    #st.write("Notre DataFrame sur les consommations d'énergie par région et par tranche de 3 h")
    st.markdown("<h1 style='text-align: center;'>Exploration des données</h1>", unsafe_allow_html=True)
    st.write("---")
    st.markdown("""  
Cette section a pour objectif d'explorer notre jeu de données afin d'identifier les différentes valeurs et modalités de nos variables. Le **dataframe** affiché est le résultat de notre processus de **nettoyage des données**, du **traitement des valeurs manquantes**, ainsi que de l'ajout de **variables pertinentes**. Nous avons également croisé les données avec des axes d'analyse supplémentaires que nous jugeons utiles, tels que la population et les conditions météorologiques.

Nous y examinons les principales caractéristiques des données, telles que les régions, les périodes horaires, ainsi que les années, pour mieux comprendre la structure et la répartition des informations. Nous affichons également les statistiques descriptives clés, telles que les moyennes, les minimums, et les maximums. Nous allons explorer plus en détails ces axes et leurs impacts dans la section suivante dédiée à l'**analyse et visualisation**.
""")
    st.write("")
    st.write("Notre DataFrame sur les consommations d'énergie par région et par tranche de 3 h")
    st.write("")
    pd.set_option('display.max_columns', None)
    dfsmp = pd.read_csv('dfsmp.csv', sep=',', header=0)
    dfsmp.columns = [col.replace(" ", "\n") for col in dfsmp.columns]
    st.dataframe(dfsmp.head(10), height=300)

    # Checkbox pour afficher les valeurs manquantes
    if st.checkbox('Afficher les régions'):
        st.write(dfsmp['region'].unique())

        # Checkbox pour afficher les valeurs manquantes
    if st.checkbox('Afficher les années'):
        #st.write(dfsmp['annee'].unique())
        annees = dfsmp['annee'].unique()

        # Convertir les années en chaîne de caractères pour éviter les virgules
        annees_str = [str(annee) for annee in annees]

        # Créer un DataFrame avec les années en format texte
        annees_df = pd.DataFrame(annees_str, columns=["Année"])

        # Afficher le DataFrame sous forme de tableau avec st.write()
        st.write(annees_df)

    
    
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
    #st.write("### Analyse des Données")
    st.markdown("<h1 style='text-align: center;'>Analyse et visualisation des Données</h1>", unsafe_allow_html=True)
    st.write("---")

    st.markdown("""
Dans cette étude, nous allons explorer les **données de consommation énergétique** et les **données de production d’énergie** en France à travers différents angles d’analyse. L'objectif est de comprendre les variations de la demande énergétique par **jour**, **heure**, et **région**. Ces axes nous permettront de visualiser les fluctuations régulières et les éventuelles disparités entre les différentes zones géographiques.

En complément, nous avons décidé de **croiser ces données avec d'autres facteurs** comme les **conditions météorologiques** et la **population**. Ce croisement vise à évaluer si ces éléments extérieurs influencent significativement la consommation énergétique. En effet, la météo peut avoir un impact direct sur l'utilisation du chauffage ou de la climatisation, tandis que la densité de population peut expliquer certaines différences entre les régions.

L'enjeu ici est de déterminer si ces **nouvelles dimensions** apportent un éclairage supplémentaire pertinent à notre étude. Ces croisements pourraient nous offrir une vision plus complète des dynamiques énergétiques en France et nous permettre d’anticiper les évolutions futures, que ce soit en termes de **gestion des infrastructures** ou d’**optimisation de l’énergie**.
""")
    st.write("")
    # Prétraitement des données pour assurer le bon format des colonnes
    dfsmp['annee_mois'] = pd.to_datetime(dfsmp['annee_mois'], format='%Y-%m').dt.to_period('M').astype(str)

    # Création du pivot table pour la heatmap avec la température
    heatmap_data = dfsmp.pivot_table(index='annee_mois', columns='region', values='temperature (C°)', aggfunc='mean')
    st.write("**Données de la Heatmap de la température par région**")
    heatmap_data = heatmap_data.round(2)
    st.write(heatmap_data)

    
    # Création de la heatmap avec Plotly
    fig = px.imshow(heatmap_data, 
                    labels=dict(x="Région", y="Année-Mois", color="Température (°C)"),
                    title="Heatmap de la Température par Région et par Année-Mois",
                    aspect="auto",
                    color_continuous_scale='Viridis')  # Utilisation d'une échelle de couleurs contrastée
    
    st.plotly_chart(fig)    
    st.markdown("""
    **Analyse :**

    Comme attendu, la région **Provence-Alpes-Côte d'Azur (PACA)** se distingue par des **températures moyennes plus élevées**, ce qui peut influencer la **consommation énergétique**, notamment en été avec une probable demande accrue en **climatisation**.

    À l'inverse, les régions proches du littoral (**Hauts-de-France, Normandie, Bretagne**) présentent des **températures plus basses en moyenne**, ce qui suggère une demande énergétique plus importante pour le **chauffage** durant les mois froids.

    Enfin, l'analyse saisonnière montre une **chute des températures pendant les mois d'hiver**, particulièrement marquée dans des régions plus continentales comme le **Grand Est**, où la **consommation énergétique pour le chauffage** est probablement au plus haut.

    Cette analyse met en lumière les **variations régionales** et **saisonnières** qui influencent directement les besoins énergétiques en France.
    """)
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
        title='Production d électricité nucléaire (MW)',
        overlaying='y',
        side='right'
    ),
    legend=dict(x=0, y=1.0, traceorder='normal')
    )

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)
    st.markdown("""
**Analyse :**

L'analyse montre clairement que la **météo**, et plus spécifiquement la **température**, a un impact direct sur la **production d'énergie nucléaire**. Pendant les périodes de fortes chaleurs, notamment en été, la production d'énergie nucléaire tend à diminuer. Cela peut s'expliquer par le fait que les centrales nucléaires doivent réduire leur production lorsque les températures extérieures sont trop élevées, en raison des contraintes environnementales liées à l'eau utilisée pour le refroidissement. À l'inverse, pendant les mois les plus froids, la production augmente pour répondre à une demande énergétique plus forte, en particulier pour le chauffage.

Cette corrélation inverse souligne l'importance d'étudier les **données météorologiques** dans le cadre de la gestion énergétique. Le croisement de notre jeu de données sur la consommation et la production d'énergie avec des **données météorologiques** permet de mieux comprendre les dynamiques sous-jacentes et d'anticiper les pics de demande.

En intégrant les données météorologiques, nous sommes en mesure d'affiner nos **prévisions en matière de consommation énergétique** et de mieux comprendre comment la météo influe directement sur la production d'énergie.
""")
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
    st.markdown("""
    **Analyse :**

    La **région Ile-de-France** affiche une consommation d'énergie bien supérieure aux autres régions, ce qui est en grande partie lié à sa **forte densité de population** et à son statut de **centre économique majeur**. Outre la population, la présence de nombreuses **infrastructures énergivores**, telles que les bureaux, centres commerciaux et centres de données, contribue également à cette forte demande en énergie.

    À l'inverse, les régions comme le **Centre-Val de Loire**, la **Bretagne** et la **Bourgogne-Franche-Comté**, qui sont moins peuplées et ont un profil économique plus rural, affichent logiquement une **consommation plus faible**. Ces régions sont davantage orientées vers des usages résidentiels de l'énergie, avec une moindre présence de grandes infrastructures industrielles ou tertiaires.

    De plus, l'analyse révèle une **forte variation saisonnière**, avec des pics de consommation clairement visibles en hiver, probablement en lien avec l'augmentation de la demande pour le chauffage. Le croisement des **données météorologiques** pourrait apporter une meilleure compréhension des facteurs qui influencent ces variations, en confirmant que les périodes de plus forte consommation coïncident avec les mois les plus froids.
    """)
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
    st.markdown("""
    **Analyse :**

La **consommation énergétique** suit un cycle hebdomadaire marqué, avec une demande plus élevée pendant les **jours ouvrables (Lundi à Vendredi)** et une baisse significative à partir du **week-end**. Cette tendance s'explique par l'activité accrue des secteurs **industriels**, **commerciaux** et **tertiaires** durant la semaine, où les infrastructures consomment davantage d'électricité pour faire fonctionner les machines de production, les systèmes de chauffage ou de climatisation, et l'éclairage des bureaux.

La chute marquée le **week-end**, en particulier le **dimanche**, reflète la réduction de l'activité économique, car de nombreuses entreprises ferment ou fonctionnent au ralenti. Ce phénomène est un indicateur clé pour la gestion des **infrastructures énergétiques**, permettant d'optimiser la production et la distribution selon les **cycles d'activité**.
""")
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
    st.markdown("""
**Analyse :**

La consommation énergétique suit un **cycle journalier marqué**, avec des **heures creuses** très visibles entre **minuit et 6h du matin**, période où les activités résidentielles et industrielles sont réduites au minimum. Pendant cette plage horaire, les prix de l'électricité sont souvent plus bas, incitant les ménages et les entreprises à reporter certains usages énergivores (comme la recharge de véhicules électriques ou l’utilisation de gros appareils électroménagers).

Le **pic de consommation** observé à **midi-12h30** coïncide avec la pause déjeuner, où la demande en électricité augmente fortement en raison de la cuisson des repas, de l’utilisation accrue des appareils électroménagers et du retour temporaire des travailleurs à domicile.

Ce graphique illustre bien le **cycle quotidien d'activité énergétique**, avec un pic au milieu de la journée, une baisse progressive dans l'après-midi et un creux marqué durant la nuit. Il serait intéressant d’étudier comment les nouvelles habitudes de travail, comme le **télétravail**, influencent ces cycles de consommation.
""")
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
    st.markdown("""
**Analyse :**

En croisant les **données de consommation énergétique** avec la **population régionale**, nous pouvons mieux comprendre les dynamiques sous-jacentes qui influencent la demande énergétique. Ce croisement révèle que la **consommation moyenne par habitant en Ile-de-France** est nettement plus élevée que dans les autres régions, ce qui s'explique en partie par la **densité d'infrastructures** et la concentration d'activités économiques dans cette région. Il ne s'agit pas seulement de la consommation des ménages, mais aussi de celle des entreprises, des bureaux, des centres commerciaux, et des réseaux de transport, qui augmentent considérablement la demande énergétique.

En revanche, les régions comme le **Centre-Val de Loire** et la **Bourgogne-Franche-Comté**, qui ont une **population plus faible** et un **profil économique plus rural**, présentent une **consommation d'énergie par habitant plus basse**. Ces régions sont moins urbanisées et disposent de moins d'infrastructures énergivores comme les grandes entreprises industrielles ou les centres commerciaux.

Le croisement des **données de consommation avec la population** permet de mettre en lumière non seulement les régions à forte demande énergétique, mais aussi celles où chaque habitant contribue de manière significative à la consommation d’énergie. Cette approche offre une nouvelle perspective pour l’**optimisation des infrastructures énergétiques**, en ajustant la production et la distribution selon les besoins spécifiques des différentes régions. En anticipant ces différences, nous pouvons mieux comprendre où les efforts de réduction ou d'amélioration de l'efficacité énergétique doivent être concentrés.
""")
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
    st.markdown("""
**Analyse :**

La **consommation énergétique par région** reste relativement **stable au fil des années**, avec de légères fluctuations selon les régions. La tendance générale semble indiquer une **légère baisse** de la consommation, notamment à partir de 2019. Cela pourrait refléter des **efforts continus en matière d'efficacité énergétique** ou encore une **réduction des activités industrielles** dans certaines régions.

Il est important de noter que la **consommation en 2022** apparaît plus faible, mais cela s’explique par le fait que les **données ne couvrent que le premier trimestre**, ce qui fausse la comparaison avec les années précédentes. Cette baisse apparente doit donc être interprétée avec prudence.
""")
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
    title='Production Moyenne par Région pour Chaque Type d\'Énergie',
    xaxis_title='Région',
    yaxis_title='Production Moyenne (MWh)',
    barmode='stack',  # Barres empilées
    xaxis_tickangle=-45,  # Angle des étiquettes de l'axe des x
    legend=dict(x=0, y=1.0, traceorder='normal', orientation='h')  # Légende horizontale
    )

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)
    st.markdown("""
**Analyse :**

L'**énergie nucléaire** est clairement la source dominante de production d'énergie dans les régions de France où elle est exploitée. Les régions d'**Auvergne-Rhône-Alpes**, du **Centre-Val de Loire**, du **Grand Est**, des **Hauts-de-France**, de la **Nouvelle-Aquitaine**, de la **Normandie**, et de l'**Occitanie** se distinguent particulièrement par une part importante de cette source dans leur mix énergétique. Cela reflète la forte dépendance de la France à l'énergie nucléaire pour répondre à ses besoins énergétiques.

En ce qui concerne l'**énergie hydraulique** et l'**énergie thermique**, elles sont réparties de manière relativement homogène dans les régions qui en produisent. L'**énergie hydraulique** est principalement produite dans les régions montagneuses ou à fort potentiel hydroélectrique telles que l'**Auvergne-Rhône-Alpes**, le **Grand Est**, la **Nouvelle-Aquitaine**, l'**Occitanie**, et la **Provence-Alpes-Côte d'Azur**. Ces régions bénéficient de reliefs et de ressources hydriques favorables à l'exploitation hydroélectrique.

L'**énergie thermique**, quant à elle, est plus présente dans des régions industrielles comme le **Grand Est**, les **Hauts-de-France**, l'**Île-de-France**, la **Normandie**, la **Provence-Alpes-Côte d'Azur**, et les **Pays de la Loire**. Elle est souvent utilisée comme source d'appoint pour répondre aux pics de demande ou en complément des autres sources d'énergie.

En termes de **production globale**, les régions qui produisent le moins d'énergie sont la **Bourgogne-Franche-Comté**, l'**Île-de-France**, les **Pays de la Loire**, et la **Bretagne**. Ces régions se distinguent par une faible capacité de production énergétique, en raison soit de leur profil géographique, soit de l'absence d'infrastructures nucléaires ou hydroélectriques.

À l'inverse, les régions qui produisent le plus d'énergie sont l'**Auvergne-Rhône-Alpes**, le **Centre-Val de Loire**, le **Grand Est**, les **Hauts-de-France**, la **Normandie**, et la **Nouvelle-Aquitaine**. Cette répartition est directement liée à la présence de centrales nucléaires dans ces régions, couplée à d'autres sources d'énergie comme l'hydraulique et le thermique.
""")
    st.write("---")
    st.markdown("#### Conclusion")
    st.markdown("""
L’analyse détaillée des **données de consommation et de production d’énergie** a permis de dégager des **tendances claires** et des **insights clés** sur le comportement énergétique en France. À travers des visualisations par **jour**, **heure**, et **région**, nous avons identifié des cycles réguliers ainsi que des disparités régionales marquées, notamment dues aux infrastructures et aux spécificités économiques locales.

Le choix d’explorer de nouveaux axes d’analyse, comme les **conditions météorologiques** et la **densité de population**, s’est avéré particulièrement enrichissant. En croisant ces variables avec nos données de consommation, nous avons pu mieux comprendre les facteurs qui influencent la demande énergétique de manière plus précise. Les résultats montrent que ces facteurs externes jouent un rôle non négligeable dans la variation de la consommation, renforçant ainsi l'intérêt de croiser plusieurs dimensions dans l’étude énergétique.

Ces analyses posent les bases pour des **explorations futures**, notamment dans le cadre de **prédictions énergétiques** plus fines, en prenant en compte à la fois les **variations saisonnières** et **les caractéristiques régionales**. Cela ouvre également des pistes intéressantes pour une **gestion optimisée des ressources énergétiques** à travers des stratégies adaptées aux besoins réels.
""")
    #############################################################


elif page == pages[3]:
    #st.write("### Modélisation et Prédictions")
    st.markdown("<h1 style='text-align: center;'>Modélisation et Prédictions</h1>", unsafe_allow_html=True)
    
    # Afficher les résultats des modèles
    pd.set_option('display.max_columns', None)
    #result_models = pd.read_csv('result_models.csv', sep=';', header=0)
    results_algo = joblib.load('Modèles et résultats JOBLIB/results_df_algo.joblib')
    
    # Fonction pour formater les grands nombres
    def format_large_integers(val):
        if isinstance(val, (int, float)) and val > 1_000_000_000_000:
            return f'{val / 1_000_000_000_000:.2f}T'  # Trillions
        elif isinstance(val, (int, float)) and val > 1_000_000_000:
            return f'{val / 1_000_000_000:.2f}B'  # Billions
        elif isinstance(val, (int, float)) and val > 1_000_000:
            return f'{val / 1_000_000:.2f}M'  # Millions
        else:
            return val

    # Appliquer le formatage sur le DataFrame
    results_algo_formatted = results_algo.applymap(format_large_integers)

    

    
    st.write("---")
    #st.markdown("#### Analyse Comparée des Modèles de Machine Learning")
    st.write('##### Analyse Comparée des Modèles de Machine Learning')
    st.write("")
    st.write(results_algo_formatted)
    #st.write(results_algo)
    st.write("")
    st.write("""
    Pour l’entraînement, nous avons choisi 7 algorithmes couvrant une variété technique allant de la régression linéaire à la régression non linéaire, et du simple au complexe.
    Les algorithmes choisis sont : régression linéaire, régression ridge, lasso, elasticnet, decision tree, random forest, et xgboost.

    Dans le tableau présenté, nous avons toutes les métriques des modèles entraînés. Nous remarquons que **le modèle Random Forest** est le meilleur pour notre jeu de données, avec un score R² de 0,896. La différence entre le score d'entraînement (0,9666) et le score de test (0,896) est la plus faible, ce qui montre une bonne généralisation sans overfitting, contrairement à d'autres modèles.

    Le modèle **Decision Tree** montre une pérformance, avec un R² de 0,836, bien qu'il ait un peu plus de variance entre le train et le test score mais il montre de l'overfitting avec un train score de 1.

    En revanche, **les méthodes linéaires** (régression linéaire, ridge, lasso, et elasticnet) n’ont pas bien fonctionné, comme on peut le voir avec des R² négatifs et des erreurs significativement plus élevées. Cela montre que ces méthodes ne sont pas adaptées à la nature des données de ce projet.
""")
    st.write("---")
    st.write('##### Optimisation des Modèles avec la méthode Grid Search')
    st.write("")
 
    results_grid = joblib.load('Modèles et résultats JOBLIB/results_df_grid.joblib')
    st.write(results_grid)

    #st.image("Images\post-gridsearch.png")
    st.write("")
    st.write("""
    Après les premiers résultats, nous avons conservé uniquement les modèles non linéaires et appliqué la méthode d’optimisation **Grid Search** pour ajuster les hyperparamètres. Cette optimisation permet de réduire les risques de surapprentissage (overfitting) et d'améliorer les performances.

    Dans le tableau présenté, nous observons que grâce **Grid Search** a amélioré les performances des modèles en gardant les meilleurs hyperparamètres, notamment pour le **Decision Tree** et le **XGBoost**, tant en termes de train score que de test score, ainsi que sur la métrique R².

    **Le modèle Random Forest** reste le plus performant avec un R² de 0,895 et un score de prédiction proche de 90%. Le **XGBoost** se rapproche avec un R² de 0,876, tandis que le **Decision Tree** suit avec un R² de 0,845.

    Ces résultats montrent que l’optimisation via Grid Search a permis d’améliorer les performances des modèles et de réduire le risque de surapprentissage, surtout pour le **Random Forest**, qui présente une très bonne généralisation.
    """)

    # Charger les résultats sauvegardés
    results_path = 'Modèles et résultats JOBLIB/LRresults.pkl'
    results = joblib.load(results_path)


    #Afficher les features importances
    st.write("---")
    st.write("##### Feature importance Random Forest")
    st.write("")
    st.write("""Nous avons sélectionné Random Forest comme modèle final en raison de ses performances supérieures. Maintenant, explorons plus en détail les variables les plus influentes sur la prédiction, en utilisant la méthode de **Feature Importance**.
    """)
    st.image("Images/feature importance RandomForest.png")
    # Explication et analyse détaillée
    st.write("""

    **Analyse des résultats de la Feature Importance** :
    
    1. **Population** ressort comme la variable la plus importante avec une contribution de presque **25%**. Ce résultat suggère que la taille de la population est un facteur clé dans la consommation d’énergie. En effet, une population plus importante entraîne une demande énergétique plus élevée, ce qui est conforme aux attentes dans ce domaine.

    2. **Bioénergie** et **Échange physique** arrivent en deuxième et troisième position, avec des contributions proches de **20%**. L’importance de ces variables met en lumière le rôle significatif des sources d’énergie renouvelables, telles que la bioénergie, et les échanges d’énergie entre régions, qui influencent fortement le bilan énergétique global.

    3. **Thermique** et **Hydraulique** jouent aussi un rôle notable, avec des importances respectives de **9%** et **7%**. Ces résultats montrent que, bien que moins influentes que la population ou les énergies renouvelables, les sources d’énergie traditionnelles restent des contributeurs importants à la consommation d’énergie.
    
    4. **Température** contribue à environ **4%**, ce qui, bien que moins significatif, n’est pas négligeable. Cela souligne l’impact des variations climatiques sur la consommation d’énergie, notamment pour le chauffage ou la climatisation, en fonction des saisons et des conditions météorologiques.
    
    5. **Les variables régionales**, quant à elles, ont une influence pratiquement nulle dans ce modèle. Cela pourrait s’expliquer par le fait que la variable "population" capture déjà l’essentiel des variations entre les régions. La population étant corrélée à la densité urbaine et à la demande énergétique dans chaque région, elle pourrait résumer à elle seule ces différences géographiques. Cela signifie que les spécificités régionales sont moins pertinentes pour prédire la consommation énergétique dans ce modèle.

    **Interprétation globale** :

    Le modèle Random Forest met en évidence que des facteurs comme la taille de la population et l’apport d’énergies renouvelables jouent un rôle prédominant dans la prédiction de la consommation énergétique. L’absence d’importance des variables régionales suggère que ces dernières n’ajoutent pas d’informations supplémentaires au modèle, probablement parce que la population, qui varie entre les régions, capture déjà ces différences.

    Ces résultats soulignent la nécessité d’investir dans des infrastructures adaptées à la croissance démographique et à l'intégration accrue des sources d'énergie renouvelables. Ils renforcent aussi l'idée que les facteurs climatiques (température) et les sources d'énergie traditionnelles (thermique et hydraulique) ne doivent pas être négligés, bien qu’ils aient un impact moindre.
""")
    
    #st.title("Feature importance Decision Tree")
    #st.image("Images/feature importance DecisionTree.png")

    #st.title("Feature importance XGB")
    #st.image("/Images/feature importance XGB.png")

    # Afficher les images et les explications
    st.write("---")
    st.write("##### Analyse SHAP des variables influençant les prédictions du modèle")
    st.write("")
    st.write("""
    Les **SHAP values** (SHapley Additive exPlanations) sont une méthode puissante permettant non seulement d’identifier quelles variables sont les plus importantes dans un modèle, mais aussi de comprendre comment chaque variable influence individuellement les prédictions. Le graphique SHAP présenté nous montre l'impact de chaque variable sur la prédiction finale du modèle à travers une représentation colorée (les points rouges indiquant des valeurs élevées et les bleus des valeurs basses).""")
    
    st.image("Images/Shape Random Forest Regressor.png")
    st.write("""
    À travers le graphique ci-dessus, nous pouvons observer plus en détail comment chaque variable influence les prédictions du modèle.

    1. **Population** : Les valeurs basses de population (en bleu) réduisent les prédictions, tandis que les valeurs élevées (en rouge) augmentent fortement la consommation d'énergie. Cela montre une corrélation forte entre la population et la demande énergétique.

    2. **Bioénergie** : Les valeurs élevées de production bioénergétique augmentent les prédictions, suggérant que plus la bioénergie est disponible, plus elle a un effet positif sur la consommation ou la production d'énergie.

    3. **Thermique (therm)** : La même tendance est observée pour la variable thermique, où des valeurs élevées augmentent les prédictions, soulignant l'importance de l'énergie thermique dans le système énergétique.

    4. **Température (C°)** : Les valeurs basses de température (en bleu) augmentent les prédictions, ce qui pourrait indiquer que lorsque les températures sont basses, la demande d'énergie augmente, probablement en raison de l’utilisation accrue du chauffage.
    À l’inverse, les valeurs élevées de température (en rouge) réduisent les prédictions. Cela pourrait s'expliquer par une consommation d'énergie réduite lors des températures élevées, sauf pour des besoins de climatisation, mais globalement, la demande énergétique semble diminuer dans ces périodes plus chaudes.

    De manière générale, ces résultats montrent que la **population**, la **bioénergie**, et la **température** sont les principaux facteurs influençant la consommation d'énergie dans ce modèle. Les variables régionales et météorologiques jouent également un rôle, mais de façon plus marginale.
    """)
    
    #st.title("Shape de Decision Tree Regressor")
    #st.image("Images/Shape Decision Tree Regressor.png")
    #st.write("On remarque que les variables ayant le plus d'impact dans le modèle Decision Tree Regressor sont : population, therm, ech_phy")

    #st.title("Shape de XGB Regressor")
    #st.image("Images/Shape XGB Regressor.png")
    #st.write("On remarque que les variables ayant le plus d'impact dans le modèle XGB Regressor sont : population, bioen, therm, Température (C°)")
    st.write("---")
    st.write("##### Performance du Modèle : Prédictions vs Réalités")
    st.write("")

    st.write("""
    Pour cette comparaison, nous avons sélectionné trois régions aux caractéristiques distinctes en termes de consommation énergétique et de conditions météorologiques : **l’Île-de-France**, **Provence-Alpes-Côte d'Azur**, et **Nouvelle-Aquitaine**.

    Le premier graphique représente l’Île-de-France, une région fortement urbanisée avec une population dense et une demande énergétique importante. Cette forte demande s’explique par la présence d’infrastructures majeures, d'une industrie développée, ainsi que par la consommation des ménages.
    """)
    st.write("")
    st.image("Images/predic vs reel IDF.png")
    st.write("""

    **Analyse** :
    - La **consommation réelle moyenne** (ligne bleue) est globalement plus élevée en semaine, avec un pic notable le **jeudi**, suivie d'une forte baisse durant le week-end. Cela correspond à un schéma classique de consommation dans les grandes régions urbaines, où les activités industrielles et les infrastructures sont particulièrement sollicitées pendant les jours ouvrables, entraînant une demande énergétique plus importante. Cette baisse pendant le week-end reflète une baisse d'activité industrielle et économique.

    - **Les prédictions du modèle** (ligne orange en pointillés) suivent bien la tendance globale de la consommation réelle. Toutefois, on observe que le modèle tend à **sous-estimer légèrement** la consommation pour tous les jours de la semaine. Ce phénomène de sous-estimation pourrait s'expliquer par une modélisation insuffisante de certains facteurs critiques, comme les pics de demande énergétique liés à des événements particuliers, ou des fluctuations économiques quotidiennes dans la région.

    - **Week-end** : Le modèle **surestime la consommation** énergétique le week-end. Cela indique que le modèle ne capte pas parfaitement les dynamiques spécifiques de réduction d’activité observées en Île-de-France durant le week-end. Le modèle semble avoir des difficultés à différencier les jours de semaine des jours non ouvrés en termes d’impact sur la consommation énergétique.

    - **Écart entre les vraies valeurs et les prédictions** : Cet écart semble être plus prononcé durant les jours de **pic de consommation** comme le jeudi et durant le week-end. Cette observation met en lumière la difficulté du modèle à capturer avec précision les variations importantes de consommation lors des jours de pointe. Cela pourrait être dû à la nature complexe des besoins énergétiques dans une région aussi dense que l'Île-de-France, où plusieurs facteurs (conditions météorologiques, variations d’activités économiques, événements sociaux) influencent la consommation de manière non linéaire.
    """)
    st.write("")
    st.write("""
Le deuxième graphique représente la région Provence-Alpes-Côte d'Azur (PACA), une région caractérisée par un climat méditerranéen et une consommation énergétique influencée par des besoins spécifiques tels que la climatisation durant l'été.
""")    

    st.image("Images/predic vs reel PAC.png")
    st.write("""
**Analyse ** :
- La **consommation réelle moyenne** (ligne bleue) présente un pic marqué en milieu de semaine, avec une baisse continue à partir du jeudi jusqu'au week-end. Cela peut refléter la réduction des activités économiques en fin de semaine, typique des régions à forte composante touristique et résidentielle.

- Les **prédictions du modèle** (ligne orange) suivent la tendance générale mais, encore une fois, tendent à **sous-estimer la consommation en semaine**. Le modèle prédit également une consommation plus élevée pour le week-end par rapport aux valeurs réelles observées.

- Le modèle semble avoir des difficultés à capturer les spécificités des variations de la consommation énergétique, en particulier durant le week-end, probablement en raison de la nature fluctuante des besoins énergétiques dans cette région, influencés par le tourisme et les conditions climatiques.
""")

    st.write("""
Le troisième graphique représente la région Nouvelle-Aquitaine, une région plus étendue avec une densité de population moindre, mais avec des industries spécifiques et des variations climatiques importantes.
""")
    st.image("Images/predic vs reel NAQ.png")
    st.write("""
**Analyse** :
- La **consommation réelle moyenne** (ligne bleue) montre une tendance similaire aux autres régions, avec un pic en milieu de semaine et une baisse progressive jusqu'au week-end. 

- Le modèle semble à nouveau **sous-estimer les valeurs en semaine**, bien qu’il capture bien la tendance globale. On observe une **surestimation de la consommation** durant le week-end, similaire aux observations faites pour les autres régions.

- L’écart entre les valeurs réelles et prédites est particulièrement visible le jeudi, ce qui peut indiquer des difficultés du modèle à prendre en compte certains facteurs régionaux spécifiques, tels que la variabilité de l'activité industrielle et agricole en Nouvelle-Aquitaine.
""")
             
    st.write("")
    st.write("""
**Conclusion :**

En résumé, pour les trois régions étudiées (**Île-de-France**, **Provence-Alpes-Côte d'Azur**, et **Nouvelle-Aquitaine**), le modèle **Random Forest**  capture bien les tendances générales de consommation énergétique, notamment les variations entre les jours de semaine et le week-end.

Toutefois, des **sous-estimations en semaine** et des **surestimations durant le week-end** sont observées dans toutes les régions. Ces écarts montrent que le modèle pourrait être amélioré en prenant en compte des facteurs régionaux spécifiques, comme l'activité industrielle, les événements particuliers, ou les effets climatiques plus précis.

Bien que les dynamiques globales soient correctement appréhendées, affiner ces aspects pourrait améliorer la précision des prédictions pour mieux refléter les fluctuations réelles de la consommation énergétique dans chaque région.
    """)

    st.write("---")
    st.write("##### Prédictions")

    st.write("""
    Pour enrichir l'analyse et offrir une dimension prédictive, nous avons intégré une fonctionnalité permettant 
    à l'utilisateur de prédire la consommation énergétique en fonction de paramètres clés tels que la date, la 
    région, la population et l'heure. Ces prédictions sont générées selon notre modèle de machine learning, offrant 
    ainsi un outil interactif pour anticiper les besoins en énergie et optimiser la gestion des ressources.
    """)
    st.write("")
    model_path = "Modèles et résultats JOBLIB/Random_Forest_Regressor_model.pkl"
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
    'année': [year],
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
    'jour_semaine_sin': [np.sin(2 * np.pi * (date_input.weekday() + 1) / 7)],
    'jour_semaine_cos': [np.cos(2 * np.pi * (date_input.weekday() + 1) / 7)],
    'mois_sin': [np.sin(2 * np.pi * month / 12)],
    'mois_cos': [np.cos(2 * np.pi * month / 12)]
})
    
        # Bouton pour ajouter ou écraser les données
    if st.button("Prédire la consommation"):
        predicted_conso = model.predict(new_data)
        # Affichage du résultat
        st.write(f"La consommation énergétique prédite: {predicted_conso[0]:.2f} MW")

    # Affichage des données ajoutées
    st.write("Données ajoutées:")
    st.write(new_data)

    
elif page == pages[4]:
    st.markdown("<h1 style='text-align: center;'>Conculsion et Perspectives</h1>", unsafe_allow_html=True)
    st.write("---")

    st.markdown("""
Dans ce projet, nous avons analysé et modélisé les consommations énergétiques des différentes régions de la France métropolitaine entre 2013 et 2022, à partir des données fournies par **la RTE**. Le volume des données traité était conséquent, avec près de **2 millions d'enregistrements** et **31 variables**, ce qui a nécessité une **exploration minutieuse** et des **méthodes avancées de nettoyage et d'analyse** pour garantir des résultats fiables.



L'une des étapes cruciales de ce projet a été l'exploration rigoureuse de chacune des 31 variables. Un point particulièrement important a été la **sélection des variables les plus complètes** pour garantir la qualité du modèle. Certaines variables présentaient des taux de complétude aussi faibles que **20%**, ce qui aurait pu biaiser les résultats si elles avaient été conservées.

- **Gestion des valeurs manquantes** : Malgré nos efforts pour compléter les données, notamment par des tentatives de contact avec **RTE**, certaines variables ne pouvaient pas être récupérées ou complétées. Nous avons donc été contraints de supprimer ces variables incomplètes afin de préserver l'intégrité des résultats.
- **Sélection des variables pertinentes** : En nous concentrant sur les variables les plus complètes et les plus pertinentes et celles que nous avons pu compléter, nous avons pu garantir une modélisation fiable sans risque de biais lié à des données manquantes.
- **Encodage des variables** : Pour préparer les données à l’entraînement des modèles de machine learning, nous avons appliqué des étapes d’encodage des variables catégorielles (comme les régions et les variables temporelles) afin d'assurer une convergence optimale des modèles.



Nous avons ensuite utilisé des techniques de **machine learning supervisé**, notamment le modèle **Random Forest**, pour prédire la consommation énergétique future. Nos résultats sont globalement satisfaisants, le modèle capturant correctement les tendances générales de consommation par région, jours de la semaine, et horaires. Cependant, pour affiner encore davantage les prédictions, il serait nécessaire d'enrichir le modèle avec des variables supplémentaires.



Il est important de noter que le modèle **Random Forest**, bien que performant, est aussi gourmand en **ressources CPU et mémoire**. Dans le cadre de ce projet, nous avons dû limiter certaines optimisations et ajustements hyperparamétriques à cause des contraintes matérielles liées à l'utilisation d'ordinateurs personnels. Cela s'applique également à l'utilisation du **Grid Search**, qui permet de trouver les meilleurs hyperparamètres, mais qui, en raison de la complexité de ce processus, n'a pas pu être exploité à son maximum en raison des limitations de ressources.

Cependant, dans un environnement plus adapté, tel qu’une infrastructure serveur ou sur le cloud, le modèle et le **Grid Search** pourraient être exploités à leur **plein potentiel**, avec davantage d'arbres ou une plus grande profondeur pour le **Random Forest**, et une recherche plus exhaustive d’hyperparamètres pour le **Grid Search**. Ces optimisations pourraient non seulement améliorer la précision des prédictions, mais aussi accélérer le traitement des données et maximiser l'efficacité du modèle.



Pour améliorer la précision des prédictions, plusieurs axes pourraient être explorés :
- **Ajout de données socio-économiques** : Telles que le taux d'urbanisation, les habitudes de consommation, et les réglementations thermiques, qui influencent directement la demande énergétique.
- **Optimisation des hyperparamètres** : Ajuster davantage les paramètres des modèles, par exemple en modifiant la profondeur des arbres dans les modèles de forêt aléatoire, ou en augmentant le nombre d'itérations dans les algorithmes d’apprentissage.
- **Analyse non supervisée** : L’utilisation de techniques de clustering pourrait identifier des groupes de régions avec des profils de consommation similaires, ou des anomalies dans les comportements énergétiques régionaux.


Le projet a ouvert la voie à plusieurs études potentielles qui pourraient enrichir notre compréhension des dynamiques énergétiques en France. Parmi ces pistes, nous proposons :
- **Prédiction des productions renouvelables (éolienne et solaire)** en fonction des conditions météorologiques, pour anticiper la disponibilité de ces sources d'énergie.
- **Analyse de l'impact des prix de l’électricité** sur la consommation, notamment en période de pic.
- **Détection d’anomalies** dans les schémas de consommation, pour identifier des comportements énergétiques atypiques ou inefficaces.
- **Corrélations entre les différentes filières de production énergétique**, pour mieux comprendre comment elles se complètent ou s'opposent dans la satisfaction des besoins énergétiques nationaux.



#### Conclusion Générale



Ce projet a été un **projet complet de bout en bout**, couvrant l’ensemble du **processus décisionnel en analyse de données**. Nous avons combiné l’exploration des données, leur nettoyage, l'encodage des variables, la modélisation via des techniques de **machine learning**, et enfin l’interprétation des résultats pour comprendre les dynamiques énergétiques en France.

Ce projet nous a permis de mettre en pratique toutes les étapes essentielles à une prise de décision basée sur des données réelles et complexes. De l’exploration des variables pertinentes à l’utilisation de modèles prédictifs, chaque phase a contribué à fournir des résultats solides et à orienter la réflexion autour de la consommation énergétique.

Bien que les prédictions actuelles soient prometteuses, nous voyons dans ce projet une étape vers une compréhension encore plus fine des dynamiques de consommation énergétique. L’intégration de nouvelles variables socio-économiques, ainsi que des techniques avancées d’optimisation, pourraient faire de ce modèle un outil encore plus robuste pour anticiper les besoins énergétiques futurs et optimiser la gestion des ressources dans les années à venir.
""")
