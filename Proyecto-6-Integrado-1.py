#!/usr/bin/env python
# coding: utf-8

# # Introducción
# 
# En este proyecto, analizamos un conjunto de datos sobre videojuegos, centrado en sus ventas, puntuaciones de críticos y usuarios, y otras características relevantes. El análisis tiene como objetivo principal entender las tendencias de ventas y popularidad de los videojuegos a lo largo del tiempo, así como identificar los factores que pueden influir en su éxito comercial.
# 
# # Objetivos del Análisis
# 
# - Explorar la distribución y tendencias de las ventas de videojuegos por año, por plataforma y region.
# 
# 
# 
# - Analizar las puntuaciones de críticos y usuarios para diferentes géneros de videojuegos.
# 
# 
# 
# - Identificar correlaciones significativas entre las ventas y las puntuaciones de críticos y usuarios.
# 
# 
# 
# - Realizar pruebas de hipótesis para determinar si existen diferencias significativas en las ventas entre diferentes plataformas y géneros.
# 
# # Descripción del Conjunto de Datos
# 
#  El conjunto de datos utilizado en este análisis contiene información detallada sobre videojuegos, incluyendo:
# 
# - Nombre del videojuego
# 
# 
# 
# - Plataforma: La consola o sistema en el que se lanzó el videojuego (e.g., PS2, Wii, DS).
# 
# 
# 
# - Año de lanzamiento
# 
# 
# 
# - Género
# 
# 
# 
# - Ventas por región: Ventas en Norteamérica (NA), Europa (EU), Japón (JP) y otras regiones (Other).
# 
# 
# 
# - Puntuaciones de críticos y usuarios: Valoraciones numéricas dadas por críticos profesionales y usuarios.
# 
# 
# 
# - Clasificación: Clasificación de contenido del videojuego (e.g., E para todos, M para adultos).
# 
# 
# 
# # Procesamiento y Limpieza de Datos
# 
# Antes de realizar el análisis, se llevaron a cabo varias etapas de preprocesamiento y limpieza de datos:
# 
# ## . Conversión de Columnas: Los nombres de las columnas se convirtieron a minúsculas para una manipulación más sencilla.
# 
# 
# 
# ## . Manejo de Valores Nulos:
# 
# 
# 
# - Los valores "tbd" (to be determined) en la columna user_score se reemplazaron por NaN y la columna se convirtió al tipo de dato float.
# 
# 
# 
# - Se eliminaron las filas con valores nulos en name, genre y year_of_release.
# 
# 
# 
# - Los valores nulos en critic_score y user_score se completaron con la mediana de su respectivo género.
# 
# 
# 
# - Los valores nulos en la columna rating se reemplazaron con "unknown".
# 
# 
# 
# ## . Cálculo de Ventas Totales: Se calculó una nueva columna total_sales como la suma de las ventas en todas las regiones.
# 
# 
# 
# # Análisis Exploratorio de Datos
# 
# Se realizaron varias visualizaciones y análisis exploratorios para entender las tendencias en los datos:
# 
# - Distribución de Juegos por Año: Se analizó la cantidad de juegos lanzados por año para identificar los períodos de mayor actividad en la industria.
# 
# 
# 
# - Ventas por Plataforma: Se compararon las ventas totales entre diferentes plataformas para identificar las más exitosas.
# 
# 
# 
# - Ventas por Género: Se exploraron las ventas por género para entender qué tipos de juegos son más populares.
# 
# 
# 
# # Próximos Pasos
# 
# siguiente paso en este análisis será formular y probar hipótesis específicas sobre las ventas de videojuegos y sus correlaciones con las puntuaciones de críticos y usuarios. Esto incluirá:
# 
# - Pruebas de Hipótesis: Determinar si existen diferencias significativas en las ventas entre diferentes plataformas y géneros.
# 
# 
# 
# - Análisis de Correlación: Evaluar las relaciones entre las puntuaciones y las ventas para identificar factores clave de éxito.
# 
# 
# 
# Este análisis proporcionará una comprensión más profunda de los factores que impulsan el éxito de los videojuegos y ayudará a guiar futuras decisiones en la industria.

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats as st


# In[2]:


df= pd.read_csv('/datasets/games.csv')


# 
# <div class="alert alert-block alert-success">
# <b>Comentario del revisor:</b> <a class="tocSkip"></a>
#     
# Has realizado un excelente trabajo al importar los datos y las bibliotecas necesarias.
# 
# </div>

# In[3]:


df.columns= df.columns.str.lower()
df.sample(5)


# In[4]:


df['user_score'] = df['user_score'].replace('tbd', np.nan) #Reemplazamos los valores 'tbd' que se encuentran en nuestra columna por NaN
df['user_score'] = df['user_score'].astype(float) #Ahora cambiamos el formato de object a float para poder realizar estudios a futuro


# ### Observaciones:
# 
# Pudimos darnos cuenta al momento de convertir la columna a float que el error nos indicaba que existian datos str 'tbd' en nuestra columna por que no nos permitia realizar el cambio de formato a str, en este caso reemplazamos esos datos que supones que es la abreviatura en ingles de (to be determined), en español (por confirmar) a NaN con el metodo numpy (np.nan).
# 
# El resto de las columnas tienen un formato correcto para poder trabajar en ellas.

# In[5]:


df.info()


# In[6]:


df= df.dropna(subset=['name', 'genre', 'year_of_release'])


# <div class="alert alert-block alert-info">
#     <b>Comentario del revisor:</b> <a class="tocSkip"></a>
#     
# En ocasiones podemos completar los valores faltantes con información recaba mediante una investigación. Es muy común que cuando trabajamos con datos en la vida real, mucho de estos vengan con valores nulos que en ocasiones tendremos que completar con proxys
# </div>

# In[7]:


genre_median= df.groupby('genre')[['critic_score', 'user_score']].median()
genre_median


# In[8]:


genres= genre_median.index
genres


# In[9]:


median_critic= genre_median['critic_score'].tolist()
median_critic


# In[10]:


dict_median_critic= dict(zip(genres, median_critic))
dict_median_critic


# In[11]:


for genre, critic in dict_median_critic.items():
    filtered_genre= df['genre'] == genre
    df.loc[filtered_genre, 'critic_score'] = df.loc[filtered_genre, 'critic_score'].fillna(critic)


# In[12]:


median_users= genre_median['user_score'].tolist()
median_users


# In[13]:


dict_median_users= dict(zip(genres, median_users))
dict_median_users


# In[14]:


for genre, users in dict_median_users.items():
    filtered_users_genre= df['genre'] == genre
    df.loc[filtered_users_genre, 'user_score'] = df.loc[filtered_users_genre, 'user_score'].fillna(users)


# <div class="alert alert-block alert-success">
#     <b>Comentario del revisor:</b> <a class="tocSkip"></a>
#     
# Muy buen trabajo. Cuando trabajamos con distribuciones sesgasdas se recomienda usar la mediana para completar los valores nulos
# </div>

# In[15]:


df['rating'] = df['rating'].fillna('unknown')


# In[16]:


df['year_of_release']= df['year_of_release'].astype(int)


# In[17]:


df.info()


# In[18]:


df['total_sales']= df['na_sales'] + df['eu_sales'] + df['jp_sales'] + df['other_sales']
columns= ['name',
          'platform',
          'year_of_release',
          'genre',
          'na_sales',
          'eu_sales',
          'jp_sales',
          'other_sales',
          'total_sales',
          'critic_score',
          'user_score',
          'rating',]

df= df.reindex(columns=columns)
df.head()


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor:</b> <a class="tocSkip"></a>
#     
# Hola! Muy buen trabajo en la sección, ajustaste los nombres de las columnas a minúsculas con el uso de la función str.lower(), cambiaste el tipo de variable de dos de las variables de la base de datos, consideraste ajustar los valores ausentes de las variables identificadas de score y muy buen trabajo con la suma de todas las ventas. 
#     
#    
# 
# </div>

# # Juegos lanzados en diferentes años

# In[19]:


filtered_game_years= df['year_of_release'].value_counts().sort_index()
filtered_game_years.plot(kind= 'bar',
                        title= 'Video juegos por año',
                        xlabel= 'Año',
                        ylabel= 'Cantidad de juegos',
                        rot= 50,
                         figsize= [10,5],
                        color='Turquoise')

plt.show()


# ### Observaciones: 
# 
# Podemos visualizar que los mejores 5 años en ventas de videojuegos son desde el 2007 al 2011.

# In[20]:


filtered_sales_platform= df.groupby('platform')['total_sales'].sum()
filtered_sales_platform.sort_values(inplace=True)
filtered_sales_platform


# In[21]:


filtered_sales_platform.plot(kind= 'bar',
                            title= 'Ventas Totales por juego',
                            xlabel= 'Plataforma',
                            ylabel= 'Ventas totales a nivel mundial en millones',
                            figsize= [10,5],
                            rot= 50,
                            color= 'Turquoise')

plt.show()


# ### Observaciones:
# 
# Como podemos visualizar en las ventas totales, el PS2 lidera como la consola con mas ventas hasta el momento teniendo en cuenta que esta consola tiene en el mercado desde el año 2000, mientras que las otras plataformas fueron lanzadas desde el 2005 en adelante.   

# In[22]:


platform_max_sales= ['DS', 'Wii', 'PS3', 'X360', 'PS2']
df_filtered= df[df['platform'].isin(platform_max_sales)]
df_filtered.head()


# In[23]:


pivot_df_filtered= df_filtered.pivot_table(index='year_of_release',
                                    columns='platform',
                                    values='total_sales',
                                    aggfunc='sum')


# In[24]:


sns.set(rc={'figure.figsize': [15,8]})
sns.heatmap(pivot_df_filtered, annot=True, fmt='.2f', cmap='crest', linewidth=.01)
plt.title('Ventas Totales por Año/Plataforma', fontsize= 14)
plt.xlabel('Plataformas')
plt.ylabel('Año de lanzamiento')


# ### Observación
# 
# En el Heatmap se observa que la plataforma DS tiene ventas para el año 1985, sin embargo, esa plataforma se lanzó en el año 2004. Por tanto, se buscarán los videojuegos en cuestión de ese año con un filtro.

# In[25]:


df[(df['platform'] == 'DS') & (df['year_of_release']== 1985)]


# ### Observación
# 
# Se tiene un sólo videojuego registrado para ese año Strongest Tokyo University Shogi DS, el cuál fue lanzado en el 2008 en japon. Por lo tanto, se cambiará el año de 1985 por 2008.

# In[26]:


df.loc[15957, 'year_of_release'] = 2008


# In[27]:


df[(df['platform'] == 'DS') & (df['name']== 'Strongest Tokyo University Shogi DS')]


# In[28]:


platform_max_sales_1= ['DS', 'Wii', 'PS3', 'X360', 'PS2']
df_filtered_1= df[df['platform'].isin(platform_max_sales_1)]
df_filtered_1.head()


# In[29]:


pivot_df_filtered_1= df_filtered_1.pivot_table(index='year_of_release',
                                    columns='platform',
                                    values='total_sales',
                                    aggfunc='sum')


# In[30]:


sns.set(rc={'figure.figsize': [15,8]})
sns.heatmap(pivot_df_filtered_1, annot=True, fmt='.2f', cmap='crest', linewidth=.01)
plt.title('Ventas Totales por Año/Plataforma', fontsize= 14)
plt.xlabel('Plataformas')
plt.ylabel('Año de lanzamiento')


# ### Observaciones:
# 
# podemos visualizar en nuestro Headmap, que ya la mayorias de estas 5 plataformas para la fecha actual si nos fijamos bien 2 de ellas (DS, PS2) ya no estan generando ganancias y las otras 3 (PS3, Wii, X360) si estan generando aun un poco de ingresos pero ya estan por desaparecer lo podemos evidenciar con los pocos ingresos generados en los ultimos años en los cuales los ingresos han ido decayendo, vamos a estudiar el porque de esto. 

# In[31]:


pivot_df_filtered_1.plot(
                       kind= 'line',
                       style='.-',
                       figsize= [14,8],
                       fontsize= 12,
                       rot= 45,
                       grid= False
                       )
plt.title('Lanzamientos y Ventas de Plataformas por Año', fontsize= 20)
plt.ylabel('Ventas totales por año en millones', fontsize= 15)
plt.xlabel('Años', fontsize= 15)


# ### Observacion:
# 
# La plataforma/consola con mayores ventas es PS2 desde el 2000 hasta el 2004. En el 2004 son lanzadas al mercado otras plataformas, por lo que la competencia crece y con ello las ventas de PS2 comienzan a disminuir.
# 
# Luego podemos onservar que la proxima plataforma/consola con mayores ventas es el Wii luego de 3 años de haberse lanzado al mercado en el 2009 podemos ver un gran aumentos en las ventas, luego de ese año las ventas empezaron a caer, tomando luego el primer lugar el X360 en el año 2010 luego de ese año las ventas tambien empiezan a caer para X360, tomando ahora el primer lugar el PS3 en el año 2011.
# 
# Podemos observar en el grafico que las ventas de todas las desde el año 2011 fueron decayendo hasta el año 2016 donde se puede evidenciar que en wii se mantiene como la 3era consola mas vendida, el X360 en la segunda posicion y el PS3 en la primera posicion. teniendo encuenta que la diferencia de ventas no es muy grande estan las 3 muy cerca entre si.

# # tiempo de vida de las plataformas

# In[32]:


def era_group(year):
    """
    La función devuelve el grupo de época de los juegos de acuerdo con el año de lanzamiento usando estas reglas:
    —'retro'   para año < 2000
    —'modern'  para 2000 <= año < 2010
    —'recent'  para año >= 2010
    —'unknown' para buscar valores año (NaN)
    """

    if year < 2000:
        return 'retro'
    elif year < 2010:
        return 'modern'
    else:
        return 'recent'


df['era_group'] = df['year_of_release'].apply(era_group)
print(df.head())


# In[33]:


# se filtra el DataFrame con las plataformas retro
df_retro= df[df['era_group'] == 'retro']
df_retro.head()


# In[34]:


df_retro_sales= df_retro.groupby('platform')['total_sales'].sum().sort_values()
df_retro_sales


# In[35]:


df_retro_sales.plot(kind= 'bar',
             figsize= [10,5],
              rot= 50,
              color= 'Turquoise' )

plt.title('Ventas totales plataformas retro', fontsize= 20)
plt.ylabel('Cantidad de ventas totales en millones')
plt.xlabel('Plataformas')
plt.show()


# ### Observacion:
# 
# Las 10 plataformas retro (con lanzamiento menores al año 2000) con más ventas globales entre 1980 y 2000 fueron DC, GEN, SAT, PC, 2600, N64, SNES, GB, NES y PS.

# # Ventas globales de las plataformas populares retro en la época reciente

# In[36]:


# se crea una lista con las plataformas retro

platform_retro_list= df_retro_sales.index
platform_retro_list


# In[37]:


# se filtra el DataFrame df con el método isin() solo con las 5 plataformas retro con más ventas

df_retro_recent= df[df['platform'].isin(platform_retro_list)]
df_retro_recent= df_retro_recent[df_retro_recent['year_of_release'] > 2000]
df_retro_recent.head()


# In[38]:


df_retro_recent_total_sales= df_retro_recent.groupby('platform')['total_sales'].sum()
df_retro_recent_total_sales


# In[39]:


# se concatena el Series df_retro_recent y df_retro_recent_total_sales
concat_retro_platforms = pd.concat([df_retro_sales, df_retro_recent_total_sales], axis='columns')

# se cambia el nombre de las columnas
concat_retro_platforms.columns = ['total_sales_before_2000', 'total_sales_after_2000']
concat_retro_platforms['total_sales_after_2000']= concat_retro_platforms['total_sales_after_2000'].fillna(0)
concat_retro_platforms.reset_index()


# In[40]:


# se grafican las ventas globales para las plataformas retro
concat_retro_platforms.plot(kind= 'bar',
       rot= 45,
       figsize= [14,6],
       grid= False
       )

plt.title('Ventas Globales por Plataforma Retro', fontsize= 15)
plt.xlabel('Plataforma', fontsize= 12)
plt.ylabel('Ventas globales en usd (en millones)', fontsize= 12)
plt.legend(['Total Sales Before 2000', 'Total Sales After 2000'], fontsize= 12)
plt.show()


# ### Observaciones:
# 
# En nuestra visualizacion podemos evidenciar que de las 10 plataformas existentes antes de los 2000, solo 3 llegaron a generar ingresos notables despues de los 200 las cuales son (GB, PS y PC) siendo PC la plataforma con mayores ventas entre el grupo de 3 mencionado anteriormente. 
# 
# 
# Es impresionante como PC en la unica plataforma que logro sobrevivir a nuevas plataformas luego de los años 2000 teniendo en cuenta que esta plataforma en sus tiempo se encontraba en el top 7 de popularidad. siendo PS la mas popular para la epoca la cual en la actualidad fue superada por PC.

# #  tiempo de vida de las plataformas

# In[41]:


# se agrupan los datos por plataforma y se calcula el año mínimo y el máximo 
lifes_platforms = df.groupby('platform')['year_of_release'].agg(['min', 'max'])
# se renombran las columnas
lifes_platforms.columns = ['year_release_min', 'year_release_max']
lifes_platforms


# In[42]:


# se ordenan los datos de menor a mayor con base a la columna 'year_release_min'
# así estarán ordenadas las plataformas de acuerdo al primer lanzamiento de un videojuego

lifes_platforms= lifes_platforms.sort_values(by='year_release_min')
lifes_platforms


# In[43]:


# se crea una columna 'lifespan' para calcular el tiempo de vida
lifes_platforms['lifes'] = lifes_platforms['year_release_max'] - lifes_platforms['year_release_min']
lifes_platforms['appear']= lifes_platforms['year_release_min'].diff().fillna(0) 
lifes_platforms= lifes_platforms.sort_values(by='lifes')
lifes_platforms


# In[44]:


# se grafica el tiempo de vida de las plataformas
lifes_platforms['lifes'].plot(kind= 'bar',
       rot= 45,
       figsize= [14,6],
       grid= False
       )

plt.title('Tiempo de Vida para las plataformas', fontsize= 15)
plt.xlabel('Plataforma', fontsize= 12)
plt.ylabel('Años', fontsize= 12)

plt.show()


# ### Observaciones:
# 
# Con los nuevos datos analizados podemos evidenciar que de todas las plataformas retro, PC es la unica que se mantiene con vida en el mercado con 31 años desde que fue lanzada.

# In[45]:


# se calcula el tiempo de vida promedio para las plataformas 'retro'
print('El tiempo de vida promedio para las plataformas Retro es:')
lifes_platforms[lifes_platforms.index.isin(platform_retro_list)]['lifes'].mean()


# In[46]:


# se guardan en un array las plataformas de la era reciente, filtrando el DataFrame df
modern_recent_platforms_list = df[~(df['era_group'] == 'retro')]['platform'].unique()
modern_recent_platforms_list


# In[47]:


# Eliminamos de la lista anterior las plataformas lanzadas antes del 2000
# DC, WS, N64, PS, GB, PC, se elimina de la lista de plataformas recientes
plataformas_a_eliminar = ['PC', 'DC', 'WS', 'N64', 'PS', 'GB']

# Filtrar plataformas que no están en la lista de eliminación
modern_recent_platforms_list = [plataformas for plataformas in modern_recent_platforms_list 
                                if plataformas not in plataformas_a_eliminar]
modern_recent_platforms_list


# In[48]:


# se calcula el tiempo de vida promedio de las plataformas recientes
print('El tiempo de vida promedio para las plataformas Recientes es:')
lifes_platforms[lifes_platforms.index.isin(modern_recent_platforms_list)]['lifes'].mean()


# In[49]:


#Se calcula el promedio en que tardo cada plataforma en aparecer

print('El tiempo promedio en que tardo cada plataforma en aparecer es:')
lifes_platforms['appear'].mean()


# ### Obervaciones:
# 
# Las 5 plataformas con más ventas globales son DS, PS2, PS3, X360 y Wii. Del año 2000 hasta 2004 la consola PS2 tenía mayores ventas, pero a partir del 2004 aparecen otras plataformas de nueva generción. Lo que conlleva a que las ventas del PS2 diminuyan y las de las nuevas plataformas aumenten. El Nintendo Wii tuvo la mayor cantidad de ventas globales a partir de su lanzamiento en 2006 hasta el 2009. A partir del 2010 las cosolas que lideran las ventas globales son X360 y PS3.
# 
# Las 10 platformas retro (lanzadas antes del 2000) que solían ser populares eran DC, GEN, SAT, PC, 2600, N64, SNES, GB, NES y PS. Sim embargo, a partir del año 2000 sus ventas desaparecen en algunas plataformas y en otras se pueden apreciar ventas pero muy bajas la unica que siguio en aumento fue PC de manera considerable, por lo que ya no lideran las ventas globales, lo cuál se debe al lanzamiento de consolas de nueva generación. Es importante recalcar que la única plataforma que se ha mantenido vigente y que aumentaron sus ventas fue para PC, sus ventas aumentaron aproximadamente en 401.80 %.
# 
# En promedio las plataformas retro su tiempo de vida es 6.71 años, Mientras que, para plataformas lanzadas a partir del 2000 su tiempo de vida promedio es 7.36 años.
# 
# El tiempo promedio que tardo en lanzarse una plataforma nueva al mercado es de 1.07 años.

# # Plataformas líderes en ventas
# 
# Dado que el tiempo de vida de las plataformas esta entre 6 y 8 años y que sólo una plataforma retro se mantienen vigente, o bien, con ventas después del 2000, 'PC'. Además, las plataformas con más ventas globales después del 2010 son PS3 y X360, asimismo, la consola PS4 es de nueva generación por tanto lo más probable es que desplace al PS3. Con base a lo anterior, sólo se tomarán en cuenta los datos a partir del año 2014, para tomar en cuenta solo plataformas con ventas relevantes y de nueva generación.

# In[50]:


# se filtra el DataFrame df para guardar los datos que pertencen al año 2014 en adelante

df_filtered_2014_2016= df[df['year_of_release']> 2013]
df_filtered_2014_2016


# In[51]:


# se agrupan los datos por plataforma con la suma de sus ventas totales

sales_total_2014_2016= df_filtered_2014_2016.groupby('platform')['total_sales'].sum()
sales_total_2014_2016= sales_total_2014_2016.sort_values()


# In[52]:


sales_total_2014_2016.plot(kind= 'bar',
       rot= 45,
       figsize= [14,6],
       grid= False
       )

plt.title('Ventas Totales de las Plataformas', fontsize= 15)
plt.xlabel('Plataforma', fontsize= 12)
plt.ylabel('Ventas totales en millones', fontsize= 12)

plt.show()


# ### Observaciones:
# 
# Como podemos ver en nuestro grafico, se puede verificar que cada vez que salen plataformas nuevas las anteriores se van desplazando y van perdiendo ventas al pasar los años hasta que desaparecen por completo.
# 
# Tambien podemos observar que la plataforma PSP para estas fechas selecionadas, estaba ya casi desapareciendo, PSP termina de generar registros de ventas en el año 2015. 

# In[53]:


sns.boxplot(x="platform", y="total_sales", data=df_filtered_2014_2016, palette="Set3") #Creamos el diagrama de caja
plt.xlabel("Plataformas")
plt.ylabel("Ventas Totales en millones")
plt.title("Distribucion de ventas totales por plataformas")
plt.show() #lo imprimimos


# ### Observaciones:
# 
# Podemos observar en nuestro diagrama de caja que los datos se encuentran sesgados de forma positiva hacia la derecha, lo que nos indica que la media es mayor que la mediana.
# 
# tambien podemos observar que el promedio de ventas entre las plataformas en muy parecido la mediana es igual o muy parecidas entre si, se mantiene alrededor de menos de 1 millon en ventas, por otro lado el promedio maximo de ventas se situa por debajo de los 2 millones en ventas, las plataformas PS4, XOne, WiiU, X360 y Wii, son las plataformas con promedios de ventas importantes. 
# 
# Dos de estas 5 plataformas tienen una gran cantidad de valores atipicos (PS4 y XOne), lo que nos indica que estas dos plataformas son las que tienen mayores ventas totales, muy por encima a su promedio. En otras palabras, estas plataformas tienen ventas excepcionales que superan por mucho el rendimiento promedio de las demás plataformas

# #  Graficos de dispersión para calcular la correlación entre las reseñas y las ventas.

# In[54]:


#Filtramos el DataFrame para que solo nos muestre los datos de la plataforma PS4

df_ps4= df_filtered_2014_2016[df_filtered_2014_2016['platform'] == 'PS4']
df_ps4


# In[55]:


# Crear diagramas de dispersión
plt.figure(figsize=(14, 6))

# Puntuación crítica frente a ventas totales
plt.subplot(1, 2, 1)
sns.scatterplot(x='critic_score', y='total_sales', data=df_ps4)
plt.title('Critic Score vs Total Sales')
plt.xlabel('Critic Score')
plt.ylabel('Total Sales')

# Puntuación del usuario frente a ventas totales
plt.subplot(1, 2, 2)
sns.scatterplot(x='user_score', y='total_sales', data=df_ps4)
plt.title('User Score vs Total Sales')
plt.xlabel('User Score')
plt.ylabel('Total Sales')

plt.tight_layout()
plt.show()


# In[56]:


# Calculate correlations
critic_corr = df_ps4['critic_score'].corr(df_ps4['total_sales'])
user_corr = df_ps4['user_score'].corr(df_ps4['total_sales'])

print('Correlación entre la puntuación de la crítica y las ventas totales: ', round(critic_corr, ndigits=2))
print('Correlación entre la puntuación de los usuarios y las ventas totales: ', round(user_corr, ndigits=2))


# <div class="alert alert-block alert-success">
#     <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Muy buen trabajo con el análisis de la relación estre las scores y las ventas. 
# </div>

# ### Observaciones:
# 
# La correlación entre el puntaje crítico y las ventas totales es de aproximadamente 0,35, lo que indica una relación positiva moderada.
# 
# La correlación entre la puntuación del usuario y las ventas totales es de aproximadamente -0,07, lo que indica una relación negativa muy débil.
# 
# ##### Correlación entre puntaje crítico y ventas totales (0,35):
# 
# Existe una relación positiva moderada.
# Esto significa que, en general, a medida que el puntaje crítico aumenta, las ventas totales también tienden a aumentar.
# Sin embargo, la correlación no es perfecta, lo que indica que otros factores también podrían influir en las ventas.
# 
# 
# ##### Correlación entre la puntuación del usuario y las ventas totales (-0,07):
# 
# Existe una relación negativa muy débil.
# Esto significa que, en general, a medida que la puntuación del usuario aumenta, las ventas totales tienden a disminuir un poco.
# Sin embargo, la correlación es tan débil que no podemos estar seguros de que exista una relación real entre estas dos variables.

# # Ventas de los mismos juegos de PS4 en otras plataformas

# In[57]:


# Creamos un array con los juegos del PS4 y lo guardamos en la nueva variable 
#Para luego realizar el filtro con los mismos juegos en otras plataformas

filtered_videogames= df_ps4['name'].values
filtered_videogames


# In[58]:


df_xone= df_filtered_2014_2016[df_filtered_2014_2016['platform'] == 'XOne']
df_xone= df_xone[df_xone['name'].isin(filtered_videogames)]
df_xone


# In[59]:


# Crear diagramas de dispersión
plt.figure(figsize=(14, 6))

# Puntuación crítica frente a ventas totales
plt.subplot(1, 2, 1)
sns.scatterplot(x='critic_score', y='total_sales', data=df_xone)
plt.title('Critic Score vs Total Sales')
plt.xlabel('Critic Score')
plt.ylabel('Total Sales')

# Puntuación del usuario frente a ventas totales
plt.subplot(1, 2, 2)
sns.scatterplot(x='user_score', y='total_sales', data=df_xone)
plt.title('User Score vs Total Sales')
plt.xlabel('User Score')
plt.ylabel('Total Sales')

plt.tight_layout()
plt.show()


# In[60]:


# Calculate correlations
critic_corr = df_xone['critic_score'].corr(df_xone['total_sales'])
user_corr = df_xone['user_score'].corr(df_xone['total_sales'])

print('Correlación entre la puntuación de la crítica y las ventas totales: ', round(critic_corr, ndigits=2))
print('Correlación entre la puntuación de los usuarios y las ventas totales: ', round(user_corr, ndigits=2))


# ### Observaciones:
# 
# ##### Correlación entre puntaje crítico y ventas totales (0,35):
# 
# Podemos observar que la comparacion de los mismos juegos de PS4 en la plataforma XOne, su correlacion entre critic_score y total_sales es igual a la plaforma PS4 (0.35) lo que quiere decir que Existe una relación positiva moderada. Esto significa que, en general, a medida que el puntaje crítico aumenta, las ventas totales también tienden a aumentar.
# 
# ##### Correlación entre la puntuación del usuario y las ventas totales (-0,11):
# 
# Existe una relación negativa muy débil en este caso es un poco mas baja que la anterior pero podemos confirmas que los usuarios de ambas plataformas actuan de forma muy parecida.
# Esto significa que, en general, a medida que la puntuación del usuario aumenta, las ventas totales tienden a disminuir un poco.

# In[61]:


df_pc= df_filtered_2014_2016[df_filtered_2014_2016['platform'] == 'PC']
df_pc= df_pc[df_pc['name'].isin(filtered_videogames)]
df_pc


# In[62]:


# Crear diagramas de dispersión
plt.figure(figsize=(14, 6))

# Puntuación crítica frente a ventas totales
plt.subplot(1, 2, 1)
sns.scatterplot(x='critic_score', y='total_sales', data=df_pc)
plt.title('Critic Score vs Total Sales')
plt.xlabel('Critic Score')
plt.ylabel('Total Sales')

# Puntuación del usuario frente a ventas totales
plt.subplot(1, 2, 2)
sns.scatterplot(x='user_score', y='total_sales', data=df_pc)
plt.title('User Score vs Total Sales')
plt.xlabel('User Score')
plt.ylabel('Total Sales')

plt.tight_layout()
plt.show()


# In[63]:


# Calculate correlations
critic_corr = df_pc['critic_score'].corr(df_pc['total_sales'])
user_corr = df_pc['user_score'].corr(df_pc['total_sales'])

print('Correlación entre la puntuación de la crítica y las ventas totales: ', round(critic_corr, ndigits=2))
print('Correlación entre la puntuación de los usuarios y las ventas totales: ', round(user_corr, ndigits=2))


# ### Observaciones:
# 
# ##### Correlación entre puntaje crítico y ventas totales (0,25):
# 
# Podemos observar que la comparacion de los mismos juegos de PS4 en la plataforma PC, su correlacion entre critic_score y total_sales es muy parecida a las plaformas PS4 y XOne (0.35) lo que quiere decir que Existe una relación positiva moderada. Esto significa que, en general, a medida que el puntaje crítico aumenta, las ventas totales también tienden a aumentar.
# 
# ##### Correlación entre la puntuación del usuario y las ventas totales (-0,04):
# 
# Existe una relación negativa muy débil en este caso es un poco mas baja que las anterior pero podemos confirmar que los usuarios de ambas plataformas actuan de forma muy parecida.
# Esto significa que, en general, a medida que la puntuación del usuario aumenta, las ventas totales tienden a disminuir un poco.

# # Distribución general de los juegos por género.

# In[64]:


df_filtered_2014_2016.head()


# In[65]:


group_genre_sales= df_filtered_2014_2016.groupby('genre')['total_sales'].sum().reset_index()
group_genre_sales= group_genre_sales.sort_values(by='total_sales')


# In[66]:


plt.figure(figsize=(14, 10))

sns.barplot(x="genre", y='total_sales',
             hue="genre",
             data=group_genre_sales)
plt.title('Ventas totales por genero')
plt.xlabel('Generos')
plt.ylabel('Ventas Totales en millones')


# ### Observaciones:
# 
# Podemos observar en nuestra grafica que los generos con mayores ventas son (Role-Playing, Sports, Shooter, Action) estos juegos estan alrededor de los 100 a 200 millones en ventas, y los generos con ventas menores son. (Puzzle, Strateggy, Simulation, Adventure, Platform, Racing, Fighting, Msc) estos juegos estan por debajo de los 40 millones en ventas.

# # Perfil de usuario para cada región

# In[67]:


#Agrupamos los datos por platform y na_sales para verificar las principales plataformas.

filtered_na= df_filtered_2014_2016.groupby('platform')['na_sales'].sum().reset_index().sort_values(by='na_sales', ascending=False)
filtered_na


# Las 5 plataformas populares en NA son (PS4, XOne, X360, 3DS, PS3)

# In[68]:


#Realizamos nuestro grafico 
plt.figure(figsize=(10, 8))

sns.barplot(x="platform", y='na_sales',
             hue="platform",
             data=filtered_na)
plt.title('Pricipales plataformas en NA')
plt.xlabel('Plataformas')
plt.ylabel('Ventas Totales en millones')


# ### Observaciones:
# 
# Podemos confirmar las 5 plataformas preferidas en Norteamérica, siendo PS4 la más popular con casi 100 millones en ventas. La plataforma XOne es la segunda más popular, muy cercana a PS4, con un poco más de 80 millones en ventas. Por otro lado, tenemos las plataformas X360, 3DS, PS3 y WiiU, con ventas que oscilan entre los 19 y los 30 millones. Cabe destacar que, de las 5 plataformas, PS4 y XOne son las preferidas en Norteamérica.

# In[69]:


#Agrupamos los datos por platform y eu_sales para verificar las principales plataformas.

filtered_eu= df_filtered_2014_2016.groupby('platform')['eu_sales'].sum().reset_index().sort_values(by='eu_sales', ascending=False)
filtered_eu


# Las 5 plataformas populares en EU son (PS4, XOne, PS3, PC, 3DS)

# In[70]:


plt.figure(figsize=(10, 8))

sns.barplot(x="platform", y='eu_sales',
             hue="platform",
             data=filtered_eu)
plt.title('Pricipales plataformas en EU')
plt.xlabel('Plataformas')
plt.ylabel('Ventas Totales en millones')


# ### Observaciones: 
# 
# Podemos confirmar las 5 plataformas preferidas en Europa, siendo PS4 la más popular con mas de 120 millones en ventas. La plataforma XOne es la segunda más popular, en este caso no tan cercana a PS4, con un poco más de 40 millones en ventas. Por otro lado, tenemos las plataformas PS3, PC, 3DS y X360, con ventas que oscilan entre los 17 y los 30 millones. Cabe destacar que, de las 5 plataformas, PS4 y XOne siguen siendo las preferidas en Europa.

# In[71]:


#Agrupamos los datos por platform y jp_sales para verificar las principales plataformas.

filtered_jp= df_filtered_2014_2016.groupby('platform')['jp_sales'].sum().reset_index().sort_values(by='jp_sales', ascending=False)
filtered_jp


# Las 5 plataformas populares en JP son (3DS, PS4, PSV, PS3, WiiU)

# In[72]:


plt.figure(figsize=(10, 8))

sns.barplot(x="platform", y='jp_sales',
             hue="platform",
             data=filtered_jp)
plt.title('Pricipales plataformas en JP')
plt.xlabel('Plataformas')
plt.ylabel('Ventas Totales en millones')


# ### Observaciones: 
# 
# En el caso de Japon podemos observa comportamientos totalmente diferentes, en este pais prefieren las plataformas 3DS las cuales tienen ventas mayores a 40 millones, luego esta la segunda plataforma preferida en japon la cual es PS4 con aproximadamente 15 millones en ventas, teniendo en cuenta que esta plataforma en las otras regiones es la plataforma con mas popularidad. luego le siguen el PSV con al rededor de 14 millones en venta, la cual tiene una popularidad muy parecida a PS4 en Japon, tambien tenemos PS3 al rededor de 11 millones de ventas y WiiU con al rededor de 8 millones de ventas.
# 
# Punto importante aca es recalcar que Japon no es igual de grande que Europa o NorteAmerica, pero aun asi genera muy buenos ingresos.

# ### Observaciones General:
# 
# Pudimos observar en la grafica, que en NA las 5 plataformas populares son (PS4, XOne, X360, 3DS, PS3) Mientras que en EU (PS4, XOne, PS3, PC, 3DS) tenemos casi las mismas plataformas la unica diferencia es que en NA prefieren el X360 y en EU prefieren la PC, tambien podemos observar un orden de preferencias distintas a NA, podemos observar que en NA y EU las dos plataformas preferidas son PS4 y XOne teniendo ingresos muy parecidos, mientras que en tercera posicion en NA es el X360 y en EU es el PS3, de cuarto lugar en NA es el 3DS en EU es la PC y de quinto lugar en NA es el PS3 en EU es el 3DS. 
# 
# Por otro lado en JP es muy diferente su plataforma preferida es el 3DS, luego tenemos en segunda posicion el PS4, en tercera el PSV, en cuarto el PS3 y por ultimo el WiiU. podriamos sacar una hipotesis de que a los de Japon les gustan mas son las plataformas portatiles ya que 3 de las 5 del top son portatiles.

# In[73]:


df_filtered_2014_2016.head()


# In[74]:


#Agrupamos nuestros datos por genre y los distintos continentes para verificar que generos son los principales en cada continente.

filtered_genre_region= df_filtered_2014_2016.groupby('genre')['na_sales', 'eu_sales', 'jp_sales'].sum().reset_index().sort_values(by='na_sales', ascending=False)
filtered_genre_region


# In[75]:


#Filtramos nuestro datos con el metodo tolist para poder realizar nuestro grafico

#Tomamos los valores de NA
list_genre_na= filtered_genre_region['genre'].tolist()
list_na= filtered_genre_region['na_sales'].tolist()

#Tomamos los valores de EU
list_genre_eu= filtered_genre_region['genre'].tolist()
list_eu= filtered_genre_region['eu_sales'].tolist()

#Tomamos los valores de JP
list_genre_jp= filtered_genre_region['genre'].tolist()
list_jp= filtered_genre_region['jp_sales'].tolist()


# In[76]:


#Creamos nuestro grafico de barras

fig = go.Figure(data=[
    go.Bar(name='NA', x=list_genre_na, y=list_na, marker_color='indianred'),
    go.Bar(name='EU', x=list_genre_eu, y=list_eu, marker_color='lightsalmon'),
    go.Bar(name='JP', x=list_genre_jp, y=list_jp)
])
# Change the bar mode
fig.update_layout(title_text='Generos principales en cada region', xaxis_title= 'Generos', yaxis_title= 'Ingresos totales en millones')
fig.update_layout(barmode='group')
fig.show()


# ### Observaciones: 
# 
# Podemos observar en nuestro grafico que los 5 generos preferidos en NA son (Shooter(79.02m), Action(72.53m), Sports(46.13m), Role-Playing(33.47m) y Misc(15.05m)), en EU son (Action(74.68m), Shooter(65.52m), Sports(45.73m), Role-Playing(28.17m) y Racing(14.13m)) como podemos ver en estas dos regiones son muy parecidas las preferencias de genero, la unica diferencia estaria en el ingreso generado por cada genero en cada region y en que en NA prefieren mas los juegos de Shooter y en EU los juegos de Action.
# 
# En el caso de Japon podemos ver que su principal genero es el Role-Playing (31.16m), luego tenemos los de Action (29.58m), estos dos generos son los mas jugados en Japon ya que luego tenemos Fighting(6.37m), Misc(5.61m) y Shooter (4.87m).
# 

# In[77]:


#Agrupamos nuestros datos por rating y los distintos continentes para verificar las clasificaciones en distintos paises.
filtered_esrb= df_filtered_2014_2016.groupby('rating')['na_sales', 'eu_sales', 'jp_sales'].sum().reset_index()

#Tomamos los valores de NA
list_rating_na= filtered_esrb['rating'].tolist()
list_na= filtered_esrb['na_sales'].tolist()

#Tomamos los valores de EU
list_rating_eu= filtered_esrb['rating'].tolist()
list_eu= filtered_esrb['eu_sales'].tolist()

#Tomamos los valores de JP
list_rating_jp= filtered_esrb['rating'].tolist()
list_jp= filtered_esrb['jp_sales'].tolist()


# In[78]:


#Creamos nuestro grafico de barras

fig = go.Figure(data=[
    go.Bar(name='NA', x=list_rating_na, y=list_na, marker_color='indianred'),
    go.Bar(name='EU', x=list_rating_eu, y=list_eu, marker_color='lightsalmon'),
    go.Bar(name='JP', x=list_rating_jp, y=list_jp)
])
# Change the bar mode
fig.update_layout(title_text='Clasificaciones ESBR en cada region', xaxis_title= 'Clasificaciones', yaxis_title= 'Ingresos por region en millones')
fig.update_layout(barmode='group')
fig.show()


# ### Observaciones:
# 
# Podemos verificar en nuestro grafico que la clasificacion con mayores ventas en NA y EU es la M con NA(96.42m) y EU(93.44m) luego podemos observar una gran cantidad de datos desconocidos, los cuales representan una gran cantidad de ingresos de entre 56 a 65 millones en ventas pero no podemos saber con exactitud a que categoria pertenecen. tendriamos que ponermos a buscar cada categoria de cada juego y asignarla o en su dado caso notificar al departamento encargado para verificar y corregir estos datos.
# 
# Por otro lado tenemos la clasificacion E igualmente con grandes ingresos en NA y EU las cuales son EU(58.06m) y NA(50.74m), Luego Tenemos la Clasificacion T con NA(38.95m) y EU(34.07m). Por ultimo tenemos la clasificacion E10+ la cual nuevamente tiene mayores ventas en NA(33.23m) y EU(26.16m)
# 
# Ahora visualicemos las clasificaciones en Japon la que genera mayores ventas es la clasificacion T (14.78m) le sigue la E (8.94m) luego M (8.01m) y por ultimo E10+ (4.46m).
# 
# En conclusion podemos ver que en las regiones NA y EU las datos se comportan muy similar con excepcion nuevamente en Japon que desde el principio a tenido comportamientos muy diferentes.
# Nuevamente debemos tener en cuanta de que existe una gran cantidad de datos desconocidos.

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor:</b> <a class="tocSkip"></a>
# 
# Excelente! Con este análisis por región ayuda a complementar el análisis general anterior y a hacer zoom a los resultados por cada una de las regiones.   
# 
# </div>

# # Prueba de las hipótesis
# 
# Nuestro estudio de hipotesis se realizara a la muestra de los años entre 2014 y 2016

# In[79]:


#Filtramos el promedio de calificaciones para cada plataforma 

xone_scores = df_filtered_2014_2016[df_filtered_2014_2016['platform']== 'XOne']
pc_scores = df_filtered_2014_2016[df_filtered_2014_2016['platform']== 'PC']


# In[85]:


# se realiza el test de levene para realizar una prueba de igualdad de varianzas entre los dos grupos

alpha= 0.05

platform_levene_results = st.levene(xone_scores['user_score'], pc_scores['user_score'])

print('El valor p en el test de levene es:', platform_levene_results.pvalue)

if platform_levene_results.pvalue < alpha:
    print('Rechazamos la hipotesis nula: las poblaciones tienen una varianza diferente')
else:
    print('No rechazamos la hipotesis nula: las poblaciones tienen varianzas iguales ')


# ### Observaciones:
# 
# De acuerdo al resultado del test de levene se rechaza la hipótesis nula, por lo que las varianzas no son iguales. Entonces el parámetro equal_var se coloca en False.

# In[84]:


# Se prueba las hipótesis
# valor de alfa
alpha= 0.05
# se asigna el resultado en 'results_score'
results_score = st.ttest_ind(xone_scores['user_score'], pc_scores['user_score'], equal_var= False)

print('El valor p es:', results_score.pvalue)

if results_score.pvalue < alpha:
    print('Se rechaza la hipótesis nula: ')
else:
    print('No se rechaza la hipótesis nula')


# ### Observaciones:
# 
# De acuerdo al resultado, no podemos rechazar la hipótesis nula de que las calificaciones de los usuarios para las plataformas Xbox One y PC son iguales. El resultado indica que las calificaciones de los usuarios no difiere para cada plataforma XOne o PC

# In[87]:


#Filtramos las calificaciones para cada Genero 

action_scores = df_filtered_2014_2016[df_filtered_2014_2016['genre']== 'Action']
sport_scores = df_filtered_2014_2016[df_filtered_2014_2016['genre']== 'Sports']


# In[88]:


# se realiza el test de levene para realizar una prueba de igualdad de varianzas entre los dos grupos

alpha= 0.05

genre_levene_results = st.levene(action_scores['user_score'], sport_scores['user_score'])

print('El valor p en el test de levene es:', genre_levene_results.pvalue)

if genre_levene_results.pvalue < alpha:
    print('Rechazamos la hipotesis nula: las poblaciones tienen una varianza diferente')
else:
    print('No rechazamos la hipotesis nula: las poblaciones tienen varianzas iguales ')


# ### Observaciones:
# 
# De acuerdo al resultado del test de levene se rechaza la hipótesis nula, por lo que las varianzas no son iguales. Entonces el parámetro equal_var se coloca en False.

# In[89]:


# Se prueba las hipótesis
# valor de alfa
alpha= 0.05
# se asigna el resultado en 'results_score'
results_genre = st.ttest_ind(action_scores['user_score'], sport_scores['user_score'], equal_var= False)

print('El valor p es:', results_genre.pvalue)

if results_genre.pvalue < alpha:
    print('Se rechaza la hipótesis nula:')
else:
    print('No se rechaza la hipótesis nula')


# ### Observaciones:
# 
# De acuerdo al resultado, podemos rechazar la hipótesis nula de que las calificaciones de los usuarios para los géneros de Acción y Deportes sean diferentes. El valor de p nos dice que existe una gran probabilidad de que existe una diferencia significativa entre las calificaciones promedio de los usuarios para los géneros de Acción y Deportes. Los usuarios claramente tienen preferencias distintas entre los juegos de Acción y Deportes.

# ### Observaciones:
# 
# La prueba de la hipotesis nos confirma que efectivamente las calificaciones promedio de los usuarios para los géneros de Acción y Deportes son diferentes lo que nos indica que es cierta esta hipotesis.
# 
# Esta hipotesis la formule filtrando el Dataframe por 'genre' y por 'user_score' y colocando como parametro equal_var= el cual es un parámetro opcional que especifica si las varianzas de las poblaciones deben considerarse iguales o no. Se pasa como equal_var = True o equal_var = False (True significa que consideramos las varianzas iguales, False significa que no).

# # Conclusiones Generales
# 
# ### Juegos lanzados en diferentes años:
# 
# ![Screenshot_1.jpg](attachment:Screenshot_1.jpg)
# 
# Podemos visualizar que los mejores 5 años en ventas de videojuegos son desde el 2007 al 2011.
# 
# ### Ventas totales por plataforma:
# 
# ![Screenshot_1-2.jpg](attachment:Screenshot_1-2.jpg)
# 
# Como podemos visualizar en las ventas totales, el PS2 lidera como la consola con mas ventas hasta el momento teniendo en cuenta que esta consola tiene en el mercado desde el año 2000, mientras que las otras plataformas fueron lanzadas desde el 2005 en adelante.
# 
# ### Ventas totales por año/plataforma:
# 
# ![Screenshot_1-3.jpg](attachment:Screenshot_1-3.jpg)
# 
# podemos visualizar en nuestro Headmap, que ya la mayorias de estas 5 plataformas para la fecha actual si nos fijamos bien 2 de ellas (DS, PS2) ya no estan generando ganancias y las otras 3 (PS3, Wii, X360) si estan generando aun un poco de ingresos pero ya estan por desaparecer lo podemos evidenciar con los pocos ingresos generados en los ultimos años en los cuales los ingresos han ido decayendo, vamos a estudiar el porque de esto.
# 
# ### Lanzamientos y ventas de plataformas por año:
# 
# ![Screenshot_1-4.jpg](attachment:Screenshot_1-4.jpg)
# 
# La plataforma/consola con mayores ventas es PS2 desde el 2000 hasta el 2004. En el 2004 son lanzadas al mercado otras plataformas, por lo que la competencia crece y con ello las ventas de PS2 comienzan a disminuir.
# 
# Luego podemos observar que la proxima plataforma/consola con mayores ventas es el Wii luego de 3 años de haberse lanzado al mercado (2006) en el 2009 podemos ver un gran aumentos en las ventas, luego de ese año las ventas empezaron a caer, tomando luego el primer lugar el X360 en el año 2010 luego de ese año las ventas tambien empiezan a caer para X360, tomando ahora el primer lugar el PS3 en el año 2011.
# 
# Podemos observar en el grafico que las ventas de todas las desde el año 2011 fueron decayendo hasta el año 2016 donde se puede evidenciar que en wii se mantiene como la 3era consola mas vendida, el X360 en la segunda posicion y el PS3 en la primera posicion. teniendo encuenta que la diferencia de ventas no es muy grande estan las 3 muy cerca entre si.
# 
# ## Tiempo de vida de las plataformas
# 
# ### Ventas totales de las plataformas retro:
# 
# ![Screenshot_1-5.jpg](attachment:Screenshot_1-5.jpg)
# 
# Las 10 plataformas retro (con lanzamiento menores al año 2000) con más ventas globales entre 1980 y 2000 fueron DC, GEN, SAT, PC, 2600, N64, SNES, GB, NES y PS.
# 
# ### Ventas globales de las plataformas populares retro en la época reciente:
# 
# ![Screenshot_1-6.jpg](attachment:Screenshot_1-6.jpg)
# 
# En nuestra visualizacion podemos evidenciar que de las 10 plataformas existentes antes de los 2000, solo 3 llegaron a generar ingresos notables despues de los 200 las cuales son (GB, PS y PC) siendo PC la plataforma con mayores ventas entre el grupo de 3 mencionado anteriormente.
# 
# Es impresionante como PC en la unica plataforma que logro sobrevivir a nuevas plataformas luego de los años 2000 teniendo en cuenta que esta plataforma en sus tiempo se encontraba en el top 7 de popularidad. siendo PS la mas popular para la epoca la cual en la actualidad fue superada por PC.
# 
# ### Tiempo de vida de las plataformas:
# 
# ![Screenshot_1-7.jpg](attachment:Screenshot_1-7.jpg)
# 
# Con los nuevos datos analizados podemos evidenciar que de todas las plataformas retro, PC es la unica que se mantiene con vida en el mercado con 31 años desde que fue lanzada.
# 
# - El tiempo de vida promedio para las plataformas Retro es:
#     casi 7 años
#     
# - El tiempo de vida promedio para las plataformas Recientes es:
#     al rededor de 7 años
#     
# - El tiempo promedio en que tardo cada plataforma en aparecer es:
#     al rededor de 1 año
#     
# Las 5 plataformas con más ventas globales son DS, PS2, PS3, X360 y Wii. Del año 2000 hasta 2004 la consola PS2 tenía mayores ventas, pero a partir del 2004 aparecen otras plataformas de nueva generción. Lo que conlleva a que las ventas del PS2 diminuyan y las de las nuevas plataformas aumenten. El Nintendo Wii tuvo la mayor cantidad de ventas globales a partir de su lanzamiento en 2006 hasta el 2009. A partir del 2010 las cosolas que lideran las ventas globales son X360 y PS3.
# 
# Las 10 platformas retro (lanzadas antes del 2000) que solían ser populares eran DC, GEN, SAT, PC, 2600, N64, SNES, GB, NES y PS. Sim embargo, a partir del año 2000 sus ventas desaparecen en algunas plataformas y en otras se pueden apreciar ventas pero muy bajas la unica que siguio en aumento fue PC de manera considerable, por lo que ya no lideran las ventas globales, lo cuál se debe al lanzamiento de consolas de nueva generación. Es importante recalcar que la única plataforma que se ha mantenido vigente y que aumentaron sus ventas fue para PC, sus ventas aumentaron aproximadamente en 401.80 %.
# 
# En promedio las plataformas retro su tiempo de vida es 6.71 años, Mientras que, para plataformas lanzadas a partir del 2000 su tiempo de vida promedio es 7.36 años.
# 
# El tiempo promedio que tardo en lanzarse una plataforma nueva al mercado es de 1.07 años.
# 
# ## Plataformas líderes en ventas
# 
# Dado que el tiempo de vida de las plataformas esta entre 6 y 7años y que sólo una plataforma retro se mantienen vigente, o bien, con ventas después del 2000, 'PC'. Además, las plataformas con más ventas globales después del 2010 son PS3 y X360, asimismo, la consola PS4 es de nueva generación por tanto lo más probable es que desplace al PS3. Con base a lo anterior, sólo se tomarán en cuenta los datos a partir del año 2014, para tomar en cuenta solo plataformas con ventas relevantes y de nueva generación.
# 
# ### Ventas totales de las plataformas recientes:
# 
# ![Screenshot_1-8.jpg](attachment:Screenshot_1-8.jpg)
# 
# Como podemos ver en nuestro grafico, se puede verificar que cada vez que salen plataformas nuevas las anteriores se van desplazando y van perdiendo ventas al pasar los años hasta que desaparecen por completo.
# 
# Tambien podemos observar que la plataforma PSP para estas fechas selecionadas, estaba ya casi desapareciendo, PSP termina de generar registros de ventas en el año 2015.
# 
# ### Distribucion de ventas totales por plataformas:
# 
# ![Screenshot_1-9.jpg](attachment:Screenshot_1-9.jpg)
# 
# Podemos observar en nuestro diagrama de caja que los datos se encuentran sesgados de forma positiva hacia la derecha, lo que nos indica que la media es mayor que la mediana.
# 
# tambien podemos observar que el promedio de ventas entre las plataformas en muy parecido la mediana es igual o muy parecidas entre si, se mantiene alrededor de menos de 1 millon en ventas, por otro lado el promedio maximo de ventas se situa por debajo de los 2 millones en ventas, las plataformas PS4, XOne, WiiU, X360 y Wii, son las plataformas con promedios de ventas importantes.
# 
# Dos de estas 5 plataformas tienen una gran cantidad de valores atipicos (PS4 y XOne), lo que nos indica que estas dos plataformas son las que tienen mayores ventas totales, muy por encima a su promedio. En otras palabras, estas plataformas tienen ventas excepcionales que superan por mucho el rendimiento promedio de las demás plataformas.
# 
# ## Graficos de dispersión para calcular la correlación entre las reseñas y las ventas
# 
# ### Correlacion entre las reseñas y ventas de PS4:
# 
# ![Screenshot_1-10.jpg](attachment:Screenshot_1-10.jpg)
# 
# Correlación entre puntaje crítico y ventas totales (0,35):
# Existe una relación positiva moderada. Esto significa que, en general, a medida que el puntaje crítico aumenta, las ventas totales también tienden a aumentar. Sin embargo, la correlación no es perfecta, lo que indica que otros factores también podrían influir en las ventas.
# 
# Correlación entre la puntuación del usuario y las ventas totales (-0,07):
# Existe una relación negativa muy débil. Esto significa que, en general, a medida que la puntuación del usuario aumenta, las ventas totales tienden a disminuir un poco. Sin embargo, la correlación es tan débil que no podemos estar seguros de que exista una relación real entre estas dos variables.
# 
# ## Ventas de los mismos juegos de PS4 en otras plataformas
# 
# ### Correlacion entre las reseñas y ventas de XOne:
# 
# ![Screenshot_1-11.jpg](attachment:Screenshot_1-11.jpg)
# 
# Correlación entre puntaje crítico y ventas totales (0,35):
# Podemos observar que la comparacion de los mismos juegos de PS4 en la plataforma XOne, su correlacion entre critic_score y total_sales es igual a la plaforma PS4 (0.35) lo que quiere decir que Existe una relación positiva moderada. Esto significa que, en general, a medida que el puntaje crítico aumenta, las ventas totales también tienden a aumentar.
# 
# Correlación entre la puntuación del usuario y las ventas totales (-0,11):
# Existe una relación negativa muy débil en este caso es un poco mas baja que la anterior pero podemos confirmas que los usuarios de ambas plataformas actuan de forma muy parecida. Esto significa que, en general, a medida que la puntuación del usuario aumenta, las ventas totales tienden a disminuir un poco.
# 
# ### Correlacion entre las reseñas y ventas de PC:
# 
# ![Screenshot_1-12.jpg](attachment:Screenshot_1-12.jpg)
# 
# Correlación entre puntaje crítico y ventas totales (0,25):
# Podemos observar que la comparacion de los mismos juegos de PS4 en la plataforma PC, su correlacion entre critic_score y total_sales es muy parecida a las plaformas PS4 y XOne (0.35) lo que quiere decir que Existe una relación positiva moderada. Esto significa que, en general, a medida que el puntaje crítico aumenta, las ventas totales también tienden a aumentar.
# 
# Correlación entre la puntuación del usuario y las ventas totales (-0,04):
# Existe una relación negativa muy débil en este caso es un poco mas baja que las anterior pero podemos confirmar que los usuarios de ambas plataformas actuan de forma muy parecida. Esto significa que, en general, a medida que la puntuación del usuario aumenta, las ventas totales tienden a disminuir un poco.
# 
# ## Distribución general de los juegos por género
# 
# ### Ventas totales por genero:
# 
# ![Screenshot_1-13.jpg](attachment:Screenshot_1-13.jpg)
# 
# Podemos observar en nuestra grafica que los generos con mayores ventas son (Role-Playing, Sports, Shooter, Action) estos juegos estan alrededor de los 100 a 200 millones en ventas, y los generos con ventas menores son. (Puzzle, Strateggy, Simulation, Adventure, Platform, Racing, Fighting, Msc) estos juegos estan por debajo de los 40 millones en ventas.
# 
# ## Perfil de usuario para cada región
# 
# ### Principales plataformas en NorteAnmerica:
# 
# ![Screenshot_1-14.jpg](attachment:Screenshot_1-14.jpg)
# 
# Podemos confirmar las 5 plataformas preferidas en Norteamérica, siendo PS4 la más popular con casi 100 millones en ventas. La plataforma XOne es la segunda más popular, muy cercana a PS4, con un poco más de 80 millones en ventas. Por otro lado, tenemos las plataformas X360, 3DS, PS3 y , con ventas que oscilan entre los 19 y los 30 millones. Cabe destacar que, de las 5 plataformas, PS4 y XOne son las preferidas en Norteamérica.
# 
# ### Principales plataformas en Europa:
# 
# ![Screenshot_1-15.jpg](attachment:Screenshot_1-15.jpg)
# 
# Podemos confirmar las 5 plataformas preferidas en Europa, siendo PS4 la más popular con mas de 120 millones en ventas. La plataforma XOne es la segunda más popular, en este caso no tan cercana a PS4, con un poco más de 40 millones en ventas. Por otro lado, tenemos las plataformas PS3, PC,3DS, con ventas que oscilan entre los 17 y los 30 millones. Cabe destacar que, de las 5 plataformas, PS4 y XOne siguen siendo las preferidas en Europa.
# 
# ### Principales plataformas en Japon:
# 
# ![Screenshot_1-16.jpg](attachment:Screenshot_1-16.jpg)
# 
# En el caso de Japon podemos observa comportamientos totalmente diferentes, en este pais prefieren las plataformas 3DS las cuales tienen ventas mayores a 40 millones, luego esta la segunda plataforma preferida en japon la cual es PS4 con aproximadamente 15 millones en ventas, teniendo en cuenta que esta plataforma en las otras regiones es la plataforma con mas popularidad. luego le siguen el PSV con al rededor de 14 millones en venta, la cual tiene una popularidad muy parecida a PS4 en Japon, tambien tenemos PS3 al rededor de 11 millones de ventas y WiiU con al rededor de 8 millones de ventas.
# 
# Punto importante aca es recalcar que Japon no es igual de grande que Europa o NorteAmerica, pero aun asi genera muy buenos ingresos.
# 
# #### Observaciones General:
# Pudimos observar en la grafica, que en NA las 5 plataformas populares son (PS4, XOne, X360, 3DS, PS3) Mientras que en EU (PS4, XOne, PS3, PC, 3DS) tenemos casi las mismas plataformas la unica diferencia es que en NA prefieren el X360 y en EU prefieren la PC, tambien podemos observar un orden de preferencias distintas a NA, podemos observar que en NA y EU las dos plataformas preferidas son PS4 y XOne teniendo ingresos muy parecidos, mientras que en tercera posicion en NA es el X360 y en EU es el PS3, de cuarto lugar en NA es el 3DS en EU es la PC y de quinto lugar en NA es el PS3 en EU es el 3DS.
# 
# Por otro lado en JP es muy diferente su plataforma preferida es el 3DS, luego tenemos en segunda posicion el PS4, en tercera el PSV, en cuarto el PS3 y por ultimo el WiiU. podriamos sacar una hipotesis de que a los de Japon les gustan mas son las plataformas portatiles ya que 3 de las 5 del top son portatiles.
# 
# ### Generos principales en cada region:
# 
# ![Screenshot_1-17.jpg](attachment:Screenshot_1-17.jpg)
# 
# Podemos observar en nuestro grafico que los 5 generos preferidos en NA son (Shooter(79.02m), Action(72.53m), Sports(46.13m), Role-Playing(33.47m) y Misc(15.05m)), en EU son (Action(74.68m), Shooter(65.52m), Sports(45.73m), Role-Playing(28.17m) y Racing(14.13m)) como podemos ver en estas dos regiones son muy parecidas las preferencias de genero, la unica diferencia estaria en el ingreso generado por cada genero en cada region y en que en NA prefieren mas los juegos de Shooter y en EU los juegos de Action.
# 
# En el caso de Japon podemos ver que su principal genero es el Role-Playing (31.16m), luego tenemos los de Action (29.58m), estos dos generos son los mas jugados en Japon ya que luego tenemos Fighting(6.37m), Misc(5.61m) y Shooter (4.87m).
# 
# 
# ### Ingresos de cada ESRB por region:
# 
# ![Screenshot_1-18.jpg](attachment:Screenshot_1-18.jpg)
# 
# Podemos verificar en nuestro grafico que la clasificacion con mayores ventas en NA y EU es la M con NA(96.42m) y EU(93.44m) luego podemos observar una gran cantidad de datos desconocidos, los cuales representan una gran cantidad de ingresos de entre 56 a 65 millones en ventas pero no podemos saber con exactitud a que categoria pertenecen. tendriamos que ponermos a buscar cada categoria de cada juego y asignarla o en su dado caso notificar al departamento encargado para verificar y corregir estos datos.
# 
# Por otro lado tenemos la clasificacion E igualmente con grandes ingresos en NA y EU las cuales son EU(58.06m) y NA(50.74m), Luego Tenemos la Clasificacion T con NA(38.95m) y EU(34.07m). Por ultimo tenemos la clasificacion E10+ la cual nuevamente tiene mayores ventas en NA(33.23m) y EU(26.16m)
# 
# Ahora visualicemos las clasificaciones en Japon la que genera mayores ventas es la clasificacion T (14.78m) le sigue la E (8.94m) luego M (8.01m) y por ultimo E10+ (4.46m).
# 
# En conclusion podemos ver que en las regiones NA y EU las datos se comportan muy similar con excepcion nuevamente en Japon que desde el principio a tenido comportamientos muy diferentes. Nuevamente debemos tener en cuanta de que existe una gran cantidad de datos desconocidos.
# 
# ## Prueba de las hipotesis
# 
# - Las calificaciones promedio de los usuarios para las plataformas Xbox One y PC son las mismas.
# 
# La prueba de hipotesis de las calificaciones promedio de los usuarios para las plataformas Xbox One y PC no son iguales por ende no es cierta esta hipotesis.
# 
# - Las calificaciones promedio de los usuarios para los géneros de Acción y Deportes son diferentes.
# 
# La prueba de la hipotesis nos confirma que efectivamente las calificaciones promedio de los usuarios para los géneros de Acción y Deportes son diferentes lo que nos indica que es cierta esta hipotesis.
# 
# 
# # Recomendaciones
# 
# Basado en los datos analisados podemos dar las siguientes recomendaciones y observaciones a tener en cuenta para obtener buenos resultados en el año 2017.
# 
# 
# 
# - Cada plataforma tiene un tiempo de vida al rededor de 6 a 7 años de vida.
# 
# 
# 
# - Cada vez que salen plataformas modernas las anteriores empiezan a ser desplazadas.
# 
# 
# 
# - EL tiempo promedio que se ha lanzado una nueva plataforma al mercado es aproximadamente con 1 año de diferencia.
# 
# 
# 
# - La unica plataforma retro que aun se encuentra en la actualidad es PC siendo unas de las plataformas preferidas por los Europeos.
# 
# 
# 
# - En general, a medida que el puntaje crítico aumenta, las ventas totales también tienden a aumentar y a medida que la puntuación del usuario aumenta, las ventas totales tienden a disminuir un poco. Este comportamiento lo evidenciamos en varias de las plataformas las cuales se comportan de manera muy similar.
# 
# 
# 
# - Los principales generos a tener en cuenta a nivel mundial para obtener mayores ingresos ya que son los preferidos son Role-Playing, Sports, Shooter, Action.
# 
# 
# 
# - las 5 plataformas preferidas en NorteAmerica son: PS4, XOne, X360, 3DS y PS3 en esta plataformas es en las que debemos enfocarnos a la hora de realizar alguna estrategia de venta.
# 
# 
# 
# - Las 5 plataformas preferidas en Europa son: PS4, XOne, PS3, PC y 3DS. en esta plataformas es en las que debemos enfocarnos a la hora de realizar alguna estrategia de venta.
# 
# 
# 
# - Las 5 plataformas preferidas en Japon son: 3DS, PS4, PSV, PS3 y WiiU, en esta region podemos observar que 3 de 5 plataformas son portatiles por lo que podemos deducir que los Japoneses prefieren las plataformas portatiles. en esta plataformas es en las que debemos enfocarnos a la hora de realizar alguna estrategia de venta.
# 
# 
# 
# - A la hora de tomar alguna decision de estrategias de ventas ya sabemos que generos prefieren los NorteAmericanos los cuales son: (Shooter(79.02m), Action(72.53m), Sports(46.13m), Role-Playing(33.47m) y Misc(15.05m))
# 
# 
# 
# - A la hora de tomar alguna decision de estrategias de ventas en base a los generos ya sabemos que en Europa prefieron los siguientes generos: (Action(74.68m), Shooter(65.52m), Sports(45.73m), Role-Playing(28.17m) y Racing(14.13m))
# 
# 
# 
# - A la hora de tomar alguna decision de estrategias de ventas en base a los generos preferidos de los Japoneses sabemos que son los siguientes: Role-Playing (31.16m), Action (29.58m), Fighting(6.37m), Misc(5.61m) y Shooter (4.87m).
# 
# 
# 
# - Tambien podemos tener en cuenta para futuros lanzamientos de Videojuegos cuales son las regiones con mayores ventas dependiendo de su ESRB: los juegos con clasificacion M son las mas vendidas en Norteamerica y Europa, la segunda clasificacion preferida por los NA y EU son las E de Tercer lugar en las mismas regiones tenemos las T por ultimo las E10+. Teniendo esto presente podemos tener en cuenta cuanto podria generar en ingresos un juego lanzado dependiendo de su clasificacion y tener una idea de quienes son nuestros potenciales clientes en dichas regiones es este caso sabemos que son personas +17 o asi deberia ser.
# 
#     En el caso de Japon es Diferente los Japones prefieren en primer instacia los juegos de clasificacion T, le siguen la clasificacion E, M y E10+. Teniendo esto presente podemos tener en cuenta cuanto podria generar en ingresos un juego lanzado dependiendo de su clasificacion y tener una idea de quienes son nuestros potenciales clientes en japon en este caso sabemos que son personas +13 o asi deberia ser.

# <div class="alert alert-block alert-success">
# <b>Comentario revisor</b> <a class="tocSkip"></a>
# 
# Excelente trabajo con el desarrollo de la conclusión y de las recomendaciones. Resumen todos los resultados del proyecto
#     
# </div>

# <div class="alert alert-block alert-warning">
# <b>Comentario revisor</b> <a class="tocSkip"></a>
# 
# En general creo que hiciste un muy buen trabajo con el proyecto, pudiste limpiar y trabajar las bases de datos de beuna manera. Además, el análisis explorario de datos fue completo al mostrar resultados relevantes que pueden ser de mucha utilidad para la toma de decisiones y desarrollaste las pruebas de hipótesis de una buena manera. No obstante, recuerda que siempre podemos mejorar y te menciono algunos puntos que debes considerar:
# 
# 
# 
# *  Considerar eliminar registros atipicos que puedan sesgar nuestros resultados. 
#     
# *  Considerar desarrollar un análisis para comprobar los supuestos de la prueba de hipótesis (varianzas iguales)
#     
# </div>
