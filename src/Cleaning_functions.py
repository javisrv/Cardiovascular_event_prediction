#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np


# # Cleaning functions

# ## 1.Functions to correct the categorical variables

# ### 1.1 Gender

# In[18]:


def correct_gender(dataframe):
    """
    Receive the dataframe and replace in the gender column 1 with m and 2 with f.
    
    Parameters
    ----------
    dataframe: pandas.core.frame.DataFrame
           
    Returns
    -------
    pandas.core.series.Series: 
        Returns the modified dataframe.
    """
    dataframe['gender'] = dataframe.gender.map({1: 'm', 2:'f'})
    
    return dataframe


# ### 1.2 Cholesterol, Gluc, Smoke, Alco, Active

# In[19]:


# Funcion que convierte el valor de las variables categoricas en categorias
def correct_categorical(dataframe):
    """
    Receive a dataframe, iterate over the categorical variables and map them to their respective categories.
    
    Parameters
    ----------
    dataframe: pandas.core.frame.DataFrame
        
    Returns
    -------
    pandas.core.series.Series: 
        Returns the modified dataframe.
    """
    info = { 'cholesterol': {1:'normal', 2:'bordering', 3:'high'},
                'gluc': {1:'normal', 2: 'prediabetes', 3: 'diabetes'},
                'smoke':{1:'yes', 0:'no'},
                'alco':{1:'yes', 0:'no'},
                'active':{1:'yes', 0:'no'}}
    for col, mapping in info.items():
        dataframe[col] = dataframe[col].map(mapping)
        
    return dataframe


# ## 2.Functions to correct the continuous variables

# ### 2.1 Blood pressure cleaning function

# In[20]:


def systolic(dataframe, systolic, dyastolic):
    """
    Receive a dataframe and two columns, which represent the systolic and diastolic blood pressure respectively.
    Corrects systolic pressure values to be within physiological ranges.
    
    Parameters
    ----------
    dataframe: pandas.core.frame.DataFrame
    systolic: pandas.core.frame.Series
        Series with systolic blood pressure (SBP) values.
    dyastolic: pandas.core.frame.Series
        Series with dyastolic blood pressure (DBP) values.
    
    Returns
    -------
    pandas.core.series.Series: 
        Returns the modified dataframe.
    """
    for i in dataframe.index:
        # If SBP < 0:
        if (dataframe.at[i, systolic] < 0):
            dataframe.at[i, systolic] = abs(dataframe.at[i, systolic])
        # If SBP == 1:
        elif dataframe.at[i, systolic] == 1:
            dataframe.at[i, systolic] = dataframe.at[i, systolic] * 100
        # If SBP > 2 and <= 7:
        elif (dataframe.at[i, systolic] >= 2) and (dataframe.at[i, systolic] <= 7):
            # If DBP >= 60 and <= 110 (this is because a physiological range is sought for DBP)
            if (dataframe.at[i, dyastolic] >= 60) and (dataframe.at[i, dyastolic] <= 110):
                dataframe.at[i, systolic] = dataframe.at[i, dyastolic] + 40
            else:
                dataframe.loc[i, systolic] = np.nan    
        # If SBP > 8 and <= 20:
        elif (dataframe.at[i, systolic] >= 8) and (dataframe.at[i, systolic] <= 20):
            dataframe.at[i, systolic] = dataframe.at[i, systolic] * 10
        # If SBP >= 21 and <= 79:
        elif (dataframe.at[i, systolic] >= 21) and (dataframe.at[i, systolic] <= 79):
            if (dataframe.at[i, dyastolic] >= 60) and (dataframe.at[i, dyastolic] <= 110):
                dataframe.at[i, systolic] = dataframe.at[i, dyastolic] + 40
            else:
                dataframe.loc[i, systolic] = np.nan
        # If SBP > 200:
        elif (dataframe.at[i, systolic] > 200):
            dataframe.at[i, systolic] = 200
            
    return dataframe


# In[21]:


def dyastolic(dataframe, systolic, dyastolic): 
    """
    Receive a dataframe and two columns, which represent the systolic and diastolic blood pressure respectively.
    Corrects dyastolic blood pressure values to be within physiological ranges.
    
    Parameters
    ----------
    dataframe: pandas.core.frame.DataFrame
    systolic: pandas.core.frame.Series
        Series with systolic blood pressure (SBP) values.
    dyastolic: pandas.core.frame.Series
        Series with dyastolic blood pressure (DBP) values.
    
    Returns
    -------
    pandas.core.series.Series: 
        Returns the modified dataframe.
    """
    for i in dataframe.index:
        # If DBP < 0:
        if (dataframe.at[i, dyastolic] < 0):
            dataframe.at[i, dyastolic] = abs(dataframe.at[i, dyastolic])
        # If DBP == 0:    
        elif dataframe.at[i, dyastolic] == 0:
            # If SBP >= 60 and <= 110 (this is because a physiological range is sought for SBP)
            if (dataframe.at[i, systolic] >= 90) and (dataframe.at[i, systolic] <= 160):
                dataframe.at[i, dyastolic] = dataframe.at[i, systolic] - 40
            else:
                dataframe.loc[i, dyastolic] = np.nan
        # If DBP == 1:
        elif dataframe.at[i, dyastolic] == 1:
            dataframe.at[i, dyastolic] = dataframe.at[i, dyastolic] * 100
        # If DBP >= 6 and <= 11:
        elif (dataframe.at[i, dyastolic] >= 6) and (dataframe.at[i, dyastolic] <= 11):
            dataframe.at[i, dyastolic] = dataframe.at[i, dyastolic] * 10
        # If DBP >= 12 and <= 20:
        elif (dataframe.at[i, dyastolic] >= 12) and (dataframe.at[i, dyastolic] <= 20):
            dataframe.at[i, dyastolic] = dataframe.at[i, dyastolic] * 10
        # If DBP >= 21 and <= 49:
        elif (dataframe.at[i, dyastolic] >= 21) and (dataframe.at[i, dyastolic] <= 49):
            dataframe.at[i, dyastolic] = 50
        
    return dataframe


# In[22]:


def remove_zeros(dataframe, dyastolic):
    """
    Receive a dataframe and one column, which represent the diastolic blood pressure.
    Removes zeros for those diastolic blood pressure values with 4 or 5 digits. When normally that should be between 2 or 3.
    
    Parameters
    ----------
    dataframe: pandas.core.frame.DataFrame
    dyastolic: pandas.core.frame.Series
        Series with dyastolic blood pressure (DBP) values.
    
    Returns
    -------
    pandas.core.series.Series: 
        Returns the modified dataframe.
    """
    for i in dataframe.index:
        # If DBP has 3 digits:
        if (dataframe.at[i, dyastolic] >= 500) and (dataframe.at[i, dyastolic] <= 999): 
            dataframe.at[i, dyastolic] = dataframe.at[i, dyastolic] // 10   
        # If DBP has 4 digits:
        if (dataframe.at[i, dyastolic] >= 1000): 
            dataframe.at[i, dyastolic] = dataframe.at[i, dyastolic] // 100
            
    return dataframe


# In[23]:


def check_bp(dataframe, systolic, dyastolic):
    """
    Receive a dataframe and two columns, which represent the systolic and diastolic blood pressure respectively.
    Perform a check where the systolic blood pressure should be greater than the diastolic. 
    If this doesn't happen, then reverse the values to make it happen.
    
    Parameters
    ----------
    dataframe: pandas.core.frame.DataFrame
    systolic: pandas.core.frame.Series
        Series with systolic blood pressure (SBP) values.
    dyastolic: pandas.core.frame.Series
        Series with dyastolic blood pressure (DBP) values.
    
    Returns
    -------
    pandas.core.series.Series: 
        Returns the modified dataframe.
    """
    for i in dataframe.index:
        # If DBP > SBP:
        if dataframe.at[i, dyastolic] > dataframe.at[i, systolic]:
            dataframe.loc[i, [systolic, dyastolic]] = (dataframe.loc[i, [dyastolic, systolic]].values)
            
    return dataframe


# In[24]:


def dyastolic_final(dataframe, dyastolic):
    """
    Receive a dataframe and and one column, which represent the diastolic blood pressure.
    Make a last correction, since 110 is considered as the normal upper limit of the DBP.
    
    Parameters
    ----------
    dataframe: pandas.core.frame.DataFrame
    dyastolic: pandas.core.frame.Series
        Series with dyastolic blood pressure (DBP) values.
    
    Returns
    -------
    pandas.core.series.Series: 
        Returns the modified dataframe.
    """    
    for i in dataframe.index:
        # If DBP > 110:
        if dataframe.at[i, dyastolic] > 110:
            dataframe.at[i, dyastolic] = 110
            
    return dataframe


# ### 2.2 Age

# In[25]:


def correct_age(dataframe):
    """
    Remove age-related columns that will not be used. 
    Creates the "AgeCat" column that contains the discretized age in years. 
    
    Parameters
    ----------
    dataframe: pandas.core.frame.DataFrame
        
    Returns
    -------
    pandas.core.series.Series: 
        Returns the modified dataframe.
    """
    # Remove age-related columns that will not be used.
    dataframe.drop(columns = ['age','AgeGroup'], inplace = True)  
    # Creates the "AgeCat" column that contains the discretized age in years.
    bins = [0, 30, 40, 50, 60, 70]
    dataframe['AgeCat'] = pd.cut(dataframe.AgeinYr, bins, right = False)
    
    return dataframe


# ### 2.3 Height

# In[26]:


def remove_outliers(dataframe, col, lo_lim, up_lim):
    """
    Eliminate outliers based on the selected limits.
    
    Parameters
    ----------
    dataframe: pandas.core.frame.DataFrame
        DataFrame to parse.
    col: pandas.core.frame.Series
        Column to parse.
    lo_lim: int
        Lower limit.
    up_lim: int
        Upper limit.  
        
    Returns
    -------
    pandas.core.series.Series
        DataFrame with outliers removed.
    """
    print("Shape of the initial dataframe:", dataframe.shape)
    out_data = dataframe[(dataframe[col] > lo_lim) & (dataframe[col] < up_lim)]
    print("Shape of the dataframe after removing outliers:", out_data.shape)
    out_data.reset_index(drop = True, inplace = True)
    
    return out_data


# ### 2.4 Weight

# In[27]:


def correct_weight(dataframe):
    """
    Corrects weights less than 50 kg based on the theoretical weight according to height
    
    Parameters
    ----------
    dataframe: pandas.core.frame.DataFrame
        
    Returns
    -------
    pandas.core.series.Series: 
        Returns the modified dataframe
    """
    dataframe['new_weight'] = dataframe.apply(lambda x: x['height'] - 100 if x['weight'] < 50 else x['weight'], axis = 1)
    dataframe['weight'] = dataframe['new_weight']
    dataframe.drop(columns = 'new_weight', inplace = True)
    
    return dataframe


# ### 2.5 BMI

# In[28]:


def create_BMI(dataframe):
    """
    Based on the weight and height, it calculates the BMI, replacing the "BMI" column of the original dataset.
    Also it creates a column with categorized BMI, replacing the "BMICat" column.
        
    Parameters
    ----------
    dataframe: pandas.core.frame.DataFrame
        DataFrame to parse
        
    Returns
    -------
    pandas.core.series.DataFrame: 
         Returns the modified dataframe
    """
    # Se eliminan las columnas BMI y BMI cat
    dataframe.drop(columns = ['BMI', 'BMICat'], inplace = True)
    # Se crean las columnas bmi y bmi_cat
    dataframe['bmi'] = ((dataframe['weight'] / (dataframe['height'] ** 2)) * 10000).round(2)
    dataframe['bmi_cat'] = pd.cut(dataframe.bmi, bins = [0, 18.5, 24.9, 29.9, 70], labels = ['Underweight', 'Normal Weight', 'Overweight', 'Obesity'])
    
    return dataframe


# ## 3.Function for scanning individual columns in a standardized way

# In[29]:


def exploracion(dataframe, col):
    """
    Make a description and a boxplot of the selected column.
    
    Parameters
    ----------
    dataframe: pandas.core.frame.DataFrame
        DataFrame to parse.
    col: pandas.core.frame.Series
        Column to parse.
           
    Returns
    -------
    pandas.core.series.Series: 
        Description of the selected column.
        
    matplotlib.axes._subplots.AxesSubplot: 
        Boxplot of the selected column.
    """
    describe = dataframe[col].describe()
    graphic = sns.boxplot(data = dataframe, x = col)
    title = ("Column distribution: " + col.title())
    plt.title(title)
    plt.show()

    print("Column description", col.title(), "\n", describe)


# ## 4.Functions to calculate the Framingham and European risk score

# ### 4.1 Framingham

# In[30]:


#Framingham Risk Score for Women

def framingham_women(row):
    """
        Calcula el score de Framingham para mujeres
        Devuelve el score como un entero
    """
    import random as rd
    points=0
    # Puntaje por edad
    if  row['AgeinYr'] <=34: points -= 7
    elif row['AgeinYr'] <=39: points -= 3
    elif row['AgeinYr'] <=44: points -= 0
    elif row['AgeinYr'] <=49: points += 3
    elif row['AgeinYr'] <=54: points += 6
    elif row['AgeinYr'] <=59: points +=8
    elif row['AgeinYr'] <=64: points +=10
    elif row['AgeinYr'] <=69: points +=12
    elif row['AgeinYr'] <=74: points +=14
    else: points +=16
    
    # Puntaje por niveles de colesterol total y tabaquismo
    if row['AgeinYr'] <=39:
        if row['cholesterol'] =='normal':  points += rd.choice([0,4]) # hay 2 subcategorias que no puedo definir 
        if row['cholesterol'] =='limitrofe': points += 8
        else: points += rd.choice([11,13]) # hay 2 subcategorias que no puedo definir 
        if row['smoke'] =='si': points += 9
    elif row['AgeinYr'] <=49:
        if row['cholesterol'] =='normal':  points += rd.choice([0,3]) # hay 2 subcategorias que no puedo definir 
        if row['cholesterol'] =='limitrofe': points += 6
        else: points += rd.choice([8,10]) # hay 2 subcategorias que no puedo definir 
        if row['smoke'] =='si': points += 7
    elif row['AgeinYr'] <=59:
        if row['cholesterol'] =='normal':  points += rd.choice([0,2]) # hay 2 subcategorias que no puedo definir 
        if row['cholesterol'] =='limitrofe': points += 4
        else: points += rd.choice([5,7]) # hay 2 subcategorias que no puedo definir 
        if row['smoke'] =='si': points += 4
    elif row['AgeinYr'] <=69:
        if row['cholesterol'] =='normal':  points += rd.choice([0,1]) # hay 2 subcategorias que no puedo definir 
        if row['cholesterol'] =='limitrofe': points += 2
        else: points += rd.choice([3,4]) # hay 2 subcategorias que no puedo definir 
        if row['smoke'] =='si': points += 2
    else:
        if row['cholesterol'] =='normal':  points += rd.choice([0,2]) # hay 2 subca tegorias que no puedo definir 
        if row['cholesterol'] =='limitrofe': points += 1
        else: points += 2
        if row['smoke'] =='si': points += 1

    # Puntaje según nivel de colesterol HDL
    #Hace un choices entre -1, 1 y 2 con los weights puestos segun active, bmi_cat y gluc
    prob_HDL = [0.33,0.34,0.33]
    if row['bmi_cat']=='Obesidad' or row['bmi_cat']== 'Sobrepeso': 
        prob_HDL[0] -= 0.15; prob_HDL[1] += 0.05; prob_HDL[2] += 0.10
    if row['active']=='no': 
        prob_HDL[0] -= 0.10; prob_HDL[1] += 0.04; prob_HDL[2] += 0.06
    if row['gluc'] == 'diabetes': 
        prob_HDL[0] -= 0.08; prob_HDL[1] += 0.04; prob_HDL[2] += 0.04
    points += rd.choices([-1,1,2], weights=prob_HDL)[0]

    # Puntaje por niveles de TAS
    if row['TAS'] <= 120: points += rd.choice([0,1])    # Discrimina entre valores de TAS con y sin tratamiento
    elif row['TAS'] <= 129: points += rd.choice([1,3])  # como no tenemos esa información se selecciona al azar
    elif row['TAS'] <=139: points += rd.choice([2,4])
    elif row['TAS'] <=159: points += rd.choice([3,5])
    else: points += rd.choice([4,6])
    
    return points


# In[31]:


#Framingham Risk Score for Men 
def framingham_men(row):
    """
        Calcula el score de Framingham para hombres
        Devuelve el score como un entero
    """
    import random as rd
    points=0
    # Puntaje por edad
    if  row['AgeinYr'] <=34: points -= 9
    elif row['AgeinYr'] <=39: points -= 4
    elif row['AgeinYr'] <=44: points -= 0
    elif row['AgeinYr'] <=49: points += 3
    elif row['AgeinYr'] <=54: points += 6
    elif row['AgeinYr'] <=59: points +=8
    elif row['AgeinYr'] <=64: points +=10
    elif row['AgeinYr'] <=69: points +=11
    elif row['AgeinYr'] <=74: points +=12
    else: points +=13

    # Puntaje por niveles de colesterol total y tabaquismo
    if row['AgeinYr'] <=39:
        if row['cholesterol'] =='normal':  points += rd.choice([0,4]) # hay 2 subcategorias que no puedo definir 
        if row['cholesterol'] =='limitrofe': points += 7
        else: points += rd.choice([9,1]) # hay 2 subcategorias que no puedo definir 
        if row['smoke'] =='si': points += 8
    elif row['AgeinYr'] <=49:
        if row['cholesterol'] =='normal':  points += rd.choice([0,3]) # hay 2 subcategorias que no puedo definir 
        if row['cholesterol'] =='limitrofe': points += 5
        else: points += rd.choice([6,8]) # hay 2 subcategorias que no puedo definir 
        if row['smoke'] =='si': points += 5
    elif row['AgeinYr'] <=59:
        if row['cholesterol'] =='normal':  points += rd.choice([0,2]) # hay 2 subcategorias que no puedo definir 
        if row['cholesterol'] =='limitrofe': points += 3
        else: points += rd.choice([4,5]) # hay 2 subcategorias que no puedo definir 
        if row['smoke'] =='si': points += 3
    elif row['AgeinYr'] <=69:
        if row['cholesterol'] =='normal':  points += rd.choice([0,1]) # hay 2 subcategorias que no puedo definir 
        if row['cholesterol'] =='limitrofe': points += 1
        else: points += rd.choice([2,3]) # hay 2 subcategorias que no puedo definir 
        if row['smoke'] =='si': points += 1
    elif row['cholesterol'] =='alto': 
        points += 1
        if row['smoke'] =='si': points += 1

    # Puntaje según nivel de colesterol HDL
    #Hace un choices entre -1, 1 y 2 con los weights puestos segun active, bmi_cat y gluc
    prob_HDL = [0.25,0.25,0.25,0.25]
    if row['bmi_cat']=='Obesidad' or row['bmi_cat']== 'Sobrepeso': 
        prob_HDL[0] -= 0.10;prob_HDL[1] -= 0.10; prob_HDL[2] += 0.1; prob_HDL[3] += 0.1
    if row['active']=='no': 
        prob_HDL[0] -= 0.10;prob_HDL[1] -= 0.05; prob_HDL[2] += 0.05; prob_HDL[3] += 0.1
    if row['gluc'] == 'diabetes': 
        prob_HDL[0] -= 0.05;prob_HDL[1] -= 0.05; prob_HDL[2] += 0.05; prob_HDL[3] += 0.05
    points += rd.choices([-1,0,1,2], weights=prob_HDL)[0]

    # Puntaje por niveles de TAS
    if row['TAS'] <= 129: points += rd.choice([0,1])      # Discrimina entre valores de TAS con y sin tratamiento
    elif row['TAS'] <=139: points += rd.choice([1,2])           # como no tenemos esa información se selecciona al azar
    elif row['TAS'] <=159: points += rd.choice([1,2])
    else: points += rd.choice([2,3])
    
    return points


# ### 4.2 SCORE

# In[32]:


def create_table_ESC():
    """
    Create a multidimensional array with the data from the SCORE cardiovascular risk table.
    Returns the array of numpy.
    """

    esc_table = [
              [# Women
                  [# Women NON smoker
                      [# Non smoker between 65-69 years
                      [10,10,11,12],
                      [8,9,9,9],
                      [7,7,7,8],
                      [5,6,6,6]
                      ],
                      [# Non smoker between 60-64 years
                      [7,8,8,9],
                      [6,6,7,7],
                      [5,5,5,6],
                      [4,4,4,5]
                      ],

                      [# Non smoker between 55-59 years
                      [5,6,6,7],
                      [4,4,5,5],
                      [3,3,4,4],
                      [3,3,3,3]
                      ],

                      [# Non smoker between 50-54 years
                      [4,4,5,5],
                      [3,3,4,4],
                      [2,2,3,3],
                      [2,2,2,2]
                      ],

                      [# Non smoker between 45-49 years
                      [3,3,3,4],
                      [2,2,3,3],
                      [2,2,2,2],
                      [1,1,1,2]
                      ],

                      [# Non smoker between 40-44 years
                      [2,2,3,3],
                      [1,2,2,2],
                      [1,1,1,2],
                      [1,1,1,1]
                      ]],
                    [# Women smoker
                      [# Smoker between 65-69 years
                      [15,16,7,18],
                      [13,13,14,15],
                      [10,11,12,12],
                      [9,9,9,10]
                      ],
                        
                      [# Smoker between 60-64 years
                      [12,13,14,15],
                      [10,11,11,12],
                      [8,8,9,10],
                      [6,7,7,8]
                      ],
                        
                      [# Smoker between 55-59 years
                      [10,11,11,12],
                      [8,8,9,10],
                      [6,7,7,8],
                      [5,5,6,6]
                      ],
                        
                      [# Smoker between 50-54 years
                      [8,8,9,10],
                      [6,6,7,8],
                      [5,5,6,6],
                      [3,4,4,5]
                      ],
                        
                      [# Smoker between 45-49 years
                      [6,7,8,9],
                      [5,5,6,6],
                      [3,4,4,5],
                      [3,3,3,4]
                      ],
                        
                      [# Smoker between 40-44 years
                      [5,5,6,7],
                      [3,4,5,5],
                      [3,3,3,4],
                      [2,2,2,3]
                      ]]
                  ],
                [# Men NON smoker
                    [# Men NON smoker
                      [# Non smoker between 65-69 years
                      [14,15,17,18],
                      [12,13,14,15],
                      [10,11,12,13],
                      [8,9,10,10]
                      ],
                        
                      [# Non smoker between 60-64 years
                      [11,12,13,15],
                      [9,10,11,12],
                      [7,8,9,10],
                      [6,7,7,8]
                      ],
                        
                      [# Non smoker between 55-59 years
                      [9,10,11,12],
                      [7,8,9,10],
                      [5,6,7,8],
                      [4,5,6,6]
                      ],
                        
                      [# Non smoker between 50-54 years
                      [7,8,9,10],
                      [5,6,7,8],
                      [4,5,5,6],
                      [3,4,4,5]
                      ],
                        
                      [# Non smoker between 45-49 years
                      [5,6,7,8],
                      [4,5,5,6],
                      [3,4,4,5],
                      [2,3,3,4]
                      ],
                        
                      [# Non smoker between 40-44 years
                      [4,5,6,7],
                      [3,4,4,5],
                      [2,3,3,4],
                      [2,2,2,3]
                      ]],
                    [# Mean smoker
                      [# Smoker between 65-69 years
                      [20,22,23,25],
                      [17,18,20,21],
                      [14,15,17,18],
                      [12,13,14,15]
                      ],
                        
                      [# Smoker between 60-64 years
                      [17,18,20,22],
                      [14,15,17,18],
                      [11,13,14,15],
                      [9,10,11,12]
                      ],
                        
                      [# Smoker between 55-59 years
                      [14,16,17,20],
                      [11,13,14,16],
                      [9,10,11,13],
                      [7,8,9,10]
                      ],
                        
                      [# Smoker between 50-54 years
                      [11,13,15,17],
                      [9,10,12,14],
                      [7,8,9,11],
                      [5,6,7,8]
                      ],
                      
                      [# Smoker between 45-49 years
                      [9,11,13,15],
                      [7,8,10,12],
                      [5,7,8,9],
                      [4,5,6,7]
                      ],
                        
                      [# Smoker between 40-44 years
                      [8,9,11,13],
                      [6,7,8,10],
                      [4,5,6,8],
                      [3,4,5,6]
                      ]]
                  ]
                ]
    return  numpy.array(esc_table, dtype='object')


# In[34]:


def esc_score(row,table):
    """
        Receives a row and the table of the Systematic COronary Risk Evaluation II (SCORE II) and calculates the cardiovascular risk
        Returns the corresponding score as an integer
    """
    # Position 0 is gender:
        # 0 female
        # 1 male
    # Position 1 is smoker: 
        # 0 non-smoker
        # 1 smoker
    # Position 2 is age: 
        # 0 65-69 years
        # 1 60-64 years 
        # 2 55-59 years
        # 3 50-54 years
        # 4 45-49 years 
        # 5 40-44 years
    # Position 3 is systolic blood pressure level:
        # 3 es menos 120
        # 2 menos 140 
        # menos de 160
        # 0 mas 160
    # Position 4 is cholesterol level: 
        # 0 less or equal than 149 mg/dL, 
        # 1 150-199 mg/dL
        # 2 200-249 mg/dL
        # 3 more or equal than 250 mg/dL

    
    # Position for sex
    if row['gender'] == 'f':
        p1 = 0
    else: 
        p1 = 1
        
    # Position for smoke
    if row['smoke'] =='no': 
        p2=0
    else: 
        p2=1
        
    # Position for years
    if row['AgeinYr'] >= 65:
        p3 = 0
    elif row['AgeinYr'] >= 60 and row['AgeinYr'] <= 64 :
        p3 = 1
    elif row['AgeinYr'] >= 55 and row['AgeinYr'] <= 59 :
        p3 = 2
    elif row['AgeinYr'] >= 50 and row['AgeinYr'] <= 54 :
        p3 = 3
    elif row['AgeinYr'] >= 45 and row['AgeinYr'] <= 49 :
        p3 = 4
    else:
        p3 = 5
        
    # Position for systolic blood pressure level   
    if row['TAS'] >= 160: 
        p4 = 0
    elif row['TAS'] >= 140 and row['TAS'] <= 159: 
        p4 = 1
    elif row['TAS'] >= 120 and row['TAS'] <= 139: 
        p4 = 2
    else: 
        p4 = 3
        
    # Position for cholesterol level
    if row['cholesterol'] == 'normal': 
        p5 = rd.choice([0,1])
    elif row['cholesterol'] == 'bordering': 
        p5 = 2
    else: 
        p5 = 3
    
    # Returns the score according to the SCORE table
    return table[p1, p2, p3, p4][p5]

