import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import sklearn
from sklearn import preprocessing
from math import pi
import seaborn as sns
import warnings
import imblearn
import mplsoccer
warnings.filterwarnings('ignore')
plt.style.use('ggplot')

#leitura dados
players = pd.read_csv("C:/Users/cheri/PycharmProjects/FIFA/players_fifa23.csv")
#filtro de colunas
cols = ['ID', 'Name', 'Age', 'Height', 'Weight',
       'Overall', 'Potential', 'Growth', 'TotalStats',
       'BaseStats', 'BestPosition', 'Club', 'ValueEUR', 'WageEUR',
       'ReleaseClause', 'ContractUntil', 'ClubJoined', 'OnLoad',
       'PreferredFoot', 'IntReputation', 'WeakFoot','Nationality',
       'SkillMoves', 'AttackingWorkRate', 'DefensiveWorkRate', 'PaceTotal',
       'ShootingTotal', 'PassingTotal', 'DribblingTotal', 'DefendingTotal',
       'PhysicalityTotal', 'Crossing', 'Finishing', 'HeadingAccuracy',
       'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy',
       'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility',
       'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength',
       'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision',
       'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle',
       'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']
players = players[cols]
#agrupamento por faixa de campo
lateral = ("LB","LWB","RB","RWB")
zagueiro = "CB"
meia = ("CAM","CM","CDM")
ala = ("LM", "LW", "RM", "RW")
atacante = ("CF", "ST")
goleiro = "GK"
grupos = [players['BestPosition'].isin(lateral), players['BestPosition'] == zagueiro,
          players['BestPosition'].isin(meia), players['BestPosition'].isin(ala),
          players['BestPosition'].isin(atacante), players['BestPosition'] == goleiro]
grupos_nomes = "lateral", "zagueiro", "meia", "ala", "atacante", "goleiro"
players['GroupPosition'] = np.select(grupos, grupos_nomes)
players = players.rename(columns={'PaceTotal':'Pace', 'ShootingTotal':'Shooting', 'PassingTotal':'Passing',
                                  'DribblingTotal':'DribblingT', 'DefendingTotal':'Defending',
                                  'PhysicalityTotal':'Physicality'})

models_names = []
models_scores = []
models_auc_scores = []

def grafico_dist_altura(players):
    #grafico distribuicao altura por posicao
    result =players.rename(columns={'GroupPosition':'Posições'})
    medias = result.groupby(['Posições'])['Height'].mean()
    sns.displot(x=result['Height'], hue = result['Posições'], kde="kde", bins=20)
    plt.axvline(x=medias["meia"], ls='--', label='meia', color="#e24a34")
    plt.axvline(x=medias["atacante"], ls='--', label='atacante', color="#348abd")
    plt.axvline(x=medias["ala"], ls='--', label='ala', color="#988dd4")
    plt.axvline(x=medias["goleiro"], ls='--', label='goleiro', color="#747576")
    plt.axvline(x=medias["zagueiro"], ls='--', label='zagueiro', color="#fcbe54")
    plt.axvline(x=medias["lateral"], ls='--', label='lateral', color="#8db942")

    plt.legend()
    plt.xlabel('Altura')
    plt.ylabel('Jogadores')
    plt.title('Distribuição da Altura')
    plt.show()

def grafico_relacao_idade(players):
    #atributos vs idade
    fig, axes = plt.subplots(nrows=2,ncols=3, sharex=True)
    sns.regplot(ax=axes[0,0],x="Age", y="Pace",data=players,order=3, line_kws={"linewidth":3},)
    axes[0,0].set(ylabel='Ritmo', xlabel='Idade')
    sns.regplot(ax=axes[0,1],x="Age", y="Physicality",data=players,order=3, line_kws={"linewidth":3},)
    axes[0,1].set(ylabel='Físico', xlabel='Idade')
    sns.regplot(ax=axes[0,2],x="Age", y="Defending",data=players,order=3, line_kws={"linewidth":3},)
    axes[0,2].set(ylabel='Defesa', xlabel='Idade')
    sns.regplot(ax=axes[1,0],x="Age", y="Passing",data=players,order=3, line_kws={"linewidth":3},)
    axes[1,0].set(ylabel='Passe', xlabel='Idade')
    sns.regplot(ax=axes[1,1],x="Age", y="DribblingT",data=players,order=3, line_kws={"linewidth":3},)
    axes[1,1].set(ylabel='Drible', xlabel='Idade')
    sns.regplot(ax=axes[1,2],x="Age", y="Shooting",data=players,order=3, line_kws={"linewidth":3},)
    axes[1,2].set(ylabel='Chute', xlabel='Idade')
    plt.show()

def grafico_relacao_salario(players):
    #salario vs overall
    sns.regplot(y='WageEUR',x='Overall',data=players, order=7)
    plt.title('Salário vs Overall')
    plt.xlabel('Euros')
    plt.ylabel('Overall')
    plt.show()

def grafico_relacao_posicoes(players):
    #posicoes vs valor
    players = players.rename(columns={'ValueEUR': 'Euros'})
    df_ala, df_ata, df_gol, df_lat, df_mei, df_zag = [x for _, x in players.groupby(players['GroupPosition'])]
    df = [df_ala, df_ata, df_gol, df_lat, df_mei, df_zag]
    m_ala = df[0].groupby(['Overall'])['Euros'].mean()
    m_ata = df[1].groupby(['Overall'])['Euros'].mean()
    m_gol = df[2].groupby(['Overall'])['Euros'].mean()
    m_lat = df[3].groupby(['Overall'])['Euros'].mean()
    m_mei = df[4].groupby(['Overall'])['Euros'].mean()
    m_zag = df[5].groupby(['Overall'])['Euros'].mean()
    df_ala = pd.DataFrame(data=m_ala)
    df_ala = {'Overall':m_ala.index, 'Euros': df_ala['Euros']}
    df_ala['pos'] = "ala"
    m_ala = pd.DataFrame(data=df_ala)
    df_ata = pd.DataFrame(data=m_ata)
    df_ata = {'Overall':m_ata.index, 'Euros': df_ata['Euros']}
    df_ata['pos'] = "atacante"
    m_ata = pd.DataFrame(data=df_ata)
    df_gol = pd.DataFrame(data=m_gol)
    df_gol = {'Overall':m_gol.index, 'Euros': df_gol['Euros']}
    df_gol['pos'] = "goleiro"
    m_gol = pd.DataFrame(data=df_gol)
    df_lat = pd.DataFrame(data=m_lat)
    df_lat = {'Overall':m_lat.index, 'Euros': df_lat['Euros']}
    df_lat['pos'] = "lateral"
    m_lat = pd.DataFrame(data=df_lat)
    df_mei = pd.DataFrame(data=m_mei)
    df_mei = {'Overall':m_mei.index, 'Euros': df_mei['Euros']}
    df_mei['pos'] = "meia"
    m_mei = pd.DataFrame(data=df_mei)
    df_zag = pd.DataFrame(data=m_zag)
    df_zag = {'Overall':m_zag.index, 'Euros': df_zag['Euros']}
    df_zag['pos'] = "zagueiro"
    m_zag = pd.DataFrame(data=df_zag)

    result = pd.concat([m_ala,m_ata,m_gol,m_lat,m_mei,m_zag])
    result['pos'] = result['pos'].astype('|S')
    result['pos'] = result['pos'].astype(str).str[2:-1]

    data = result.rename(columns={'pos': 'Posições'})
    sns.relplot(data=data,x='Overall',y='Euros',hue='Posições', hue_order=["atacante","meia","ala","zagueiro","lateral","goleiro"])
    plt.plot(m_ata['Overall'], m_ata['Euros'])
    plt.plot(m_mei['Overall'], m_mei['Euros'])
    plt.plot(m_ala['Overall'], m_ala['Euros'])
    plt.plot(m_zag['Overall'], m_zag['Euros'])
    plt.plot(m_lat['Overall'], m_lat['Euros'])
    plt.plot(m_gol['Overall'], m_gol['Euros'])
    plt.title('Valor de mercado vs Overall')
    plt.yscale('log')
    plt.show()

def grafico_heatmap(players):
    hm=sns.heatmap(players[['Age', 'Overall', 'Potential', 'ValueEUR', 'WageEUR', 'ReleaseClause',
           'SkillMoves', 'AttackingWorkRate', 'DefensiveWorkRate', 'Pace',
           'Shooting', 'Passing', 'Dribbling', 'Defending',
           'Physicality']].corr(), annot = True, linewidths=.5, cmap='Blues')
    hm.set_title(label='Heatmap', fontsize=20)
    plt.show()

def grafico_radarchart_posicoes(players):
    data = players
    data = data.rename(columns={'Potential': 'Potencial', 'Shooting': 'Chute', 'Passing': 'Passe',
                                      'DribblingT': 'Drible', 'Defending': 'Defesa',
                                      'Physicality': 'Físico', 'Pace': 'Ritmo'})
    df = data.loc[data['GroupPosition'] == "ala"]
    df2 = df[['Overall','Potencial','Ritmo','Chute','Passe','Drible','Defesa','Físico']]
    p1 = df2.mean(axis=0)
    df = data.loc[data['GroupPosition'] == "atacante"]
    df2 = df[['Overall','Potencial','Ritmo','Chute','Passe','Drible','Defesa','Físico']]
    p2 = df2.mean(axis=0)
    df = data.loc[data['GroupPosition'] == "lateral"]
    df2 = df[['Overall','Potencial','Ritmo','Chute','Passe','Drible','Defesa','Físico']]
    p3 = df2.mean(axis=0)
    df = data.loc[data['GroupPosition'] == "meia"]
    df2 = df[['Overall','Potencial','Ritmo','Chute','Passe','Drible','Defesa','Físico']]
    p4 = df2.mean(axis=0)
    df = data.loc[data['GroupPosition'] == "zagueiro"]
    df2 = df[['Overall','Potencial','Ritmo','Chute','Passe','Drible','Defesa','Físico']]
    p5 = df2.mean(axis=0)

    d1 = pd.DataFrame(p1).transpose()
    c1 = d1.columns
    c1 = c1.values.tolist()
    d1 = d1.values.tolist()
    r1 = d1[0]
    df = pd.DataFrame({'r':r1,'t':c1})
    d2 = pd.DataFrame(p2).transpose()
    d2 = d2.values.tolist()
    r2 = d2[0]
    df2 = pd.DataFrame({'r':r2,'t':c1})
    d3 = pd.DataFrame(p3).transpose()
    d3 = d3.values.tolist()
    r3 = d3[0]
    df3 = pd.DataFrame({'r':r3,'t':c1})
    d4 = pd.DataFrame(p4).transpose()
    d4 = d4.values.tolist()
    r4 = d4[0]
    df4 = pd.DataFrame({'r':r4,'t':c1})
    d5 = pd.DataFrame(p5).transpose()
    d5 = d5.values.tolist()
    r5 = d5[0]
    df5 = pd.DataFrame({'r':r5,'t':c1})
    categories = c1
    N = len(categories)
    values = r1
    angles = [n / float(N) * 2 * pi for n in range(N)]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=r1,theta=categories,fill='toself',name='ala'    ))
    fig.add_trace(go.Scatterpolar(r=r2,theta=categories,fill='toself',name='atacante'    ))
    fig.add_trace(go.Scatterpolar(r=r3,theta=categories,fill='toself',name='lateral'    ))
    fig.add_trace(go.Scatterpolar(r=r4,theta=categories,fill='toself',name='meia'    ))
    fig.add_trace(go.Scatterpolar(r=r5,theta=categories,fill='toself',name='zagueiro'    ))
    fig.show()

def grafico_media_toptimes(players):
    df = players.sort_values('Overall', ascending=False).groupby('Club', sort=False).head(16)
    plt.figure()
    ax = df.groupby(['Club'])['Overall'].mean().sort_values(ascending = False).head(10).plot.barh(color='#244e73')
    plt.gca().invert_yaxis()
    ax.set_xlabel('Overall')
    ax.set_ylabel('Clube')
    for bars in ax.containers:
        ax.bar_label(bars)
    ax.set_title("Média ordenada dos 16 melhores jogadores de cada time")
    plt.show()

def gerar_todos_graficos(players):
    grafico_dist_altura(players)
    grafico_media_toptimes(players)
    grafico_relacao_salario(players)
    grafico_relacao_posicoes(players)
    grafico_relacao_idade(players)
    grafico_radarchart_posicoes(players)
    grafico_heatmap(players)


#Function to check the classification report
def classification_report_fun(model_name, actual, predicted):
    text = "The Classification Report for"
    #print(f'text {model_name} Classifier:')
    #print(classification_report(actual, predicted))

def calculate_tpr_fpr(y_real, y_pred):
    #y_real: The list or series with the real classes
    #y_pred: The list or series with the predicted classes
    # Calculates the confusion matrix and recover each element
    cm = confusion_matrix(y_real, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    #tpr: The True Positive Rate of the classifier
    #fpr: The False Positive Rate of the classifier
    # Calculates tpr and fpr
    tpr = TP / (TP + FN)  # sensitivity - true positive rate
    fpr = 1 - TN / (TN + FP)  # 1-specificity - false positive rate
    return tpr, fpr


def pred_pos_lr(X_Train, Y_Train, X_Test, Y_Test):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(X_Train , Y_Train)
    Y_Pred_lr = lr.predict(X_Test)
    models_names.append("Logistic Regression")
    models_scores.append(lr.score(X_Test, Y_Test))
    classification_report_fun("Logistic Regression", Y_Test, Y_Pred_lr)
    Y_Proba_lr = lr.predict_proba(X_Test)
    auc_score_lr = roc_auc_score(Y_Test, Y_Proba_lr, multi_class = 'ovr', average = 'macro')
    models_auc_scores.append(auc_score_lr)

def pred_pos_rf(X_Train, Y_Train, X_Test, Y_Test):
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(criterion='gini',n_estimators=50)
    rf.fit(X_Train, Y_Train)
    Y_Pred_rf = rf.predict(X_Test)
    models_names.append("Random Forest")
    models_scores.append(rf.score(X_Test, Y_Test))
    classification_report_fun("Random Forest", Y_Test, Y_Pred_rf)
    Y_Proba_rf = rf.predict_proba(X_Test)
    auc_score_rf = roc_auc_score(Y_Test, Y_Proba_rf, multi_class = 'ovr', average = 'macro')
    models_auc_scores.append(auc_score_rf)

def pred_pos_dtc(X_Train, Y_Train, X_Test, Y_Test):
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier(max_features = 39,max_depth = 10)
    dt.fit(X_Train, Y_Train)
    Y_Pred_dt = dt.predict(X_Test)
    models_names.append("Decision Tree")
    models_scores.append(dt.score(X_Test, Y_Test))
    classification_report_fun("Decision Tree", Y_Test, Y_Pred_dt)
    Y_Proba_dt = dt.predict_proba(X_Test)
    auc_score_dt = roc_auc_score(Y_Test, Y_Proba_dt, multi_class = 'ovr', average = 'macro')
    models_auc_scores.append(auc_score_dt)

def pred_pos_abc(X_Train, Y_Train, X_Test, Y_Test):
    from sklearn.ensemble import AdaBoostClassifier
    ada = AdaBoostClassifier()
    ada.fit(X_Train, Y_Train)
    Y_Pred_ada = ada.predict(X_Test)
    models_names.append("Adaboost")
    models_scores.append(ada.score(X_Test, Y_Test))
    classification_report_fun("Adaboost", Y_Test, Y_Pred_ada)
    Y_Proba_ada = ada.predict_proba(X_Test)
    auc_score_ada = roc_auc_score(Y_Test, Y_Proba_ada, multi_class = 'ovr', average = 'macro')
    models_auc_scores.append(auc_score_ada)

def pred_pos_knn(X_Train, Y_Train, X_Test, Y_Test):
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors = 2)
    knn.fit(X_Train, Y_Train)
    Y_Pred_knn = knn.predict(X_Test)
    models_names.append("KNN2")
    models_scores.append(knn.score(X_Test, Y_Test))
    classification_report_fun("KNN2", Y_Test, Y_Pred_knn)
    Y_Proba_knn = knn.predict_proba(X_Test)
    auc_score_knn = roc_auc_score(Y_Test, Y_Proba_knn, multi_class = 'ovr', average = 'macro')
    models_auc_scores.append(auc_score_knn)

def pred_pos_cbc(X_Train, Y_Train, X_Test, Y_Test):
    from catboost import CatBoostClassifier
    cb = CatBoostClassifier(max_depth=10, iterations=10, learning_rate=0.3)
    cb.fit(X_Train, Y_Train)
    Y_Pred_cb = cb.predict(X_Test)
    models_names.append("CatBoost")
    models_scores.append(cb.score(X_Test, Y_Test))
    classification_report_fun("CatBoost", Y_Test, Y_Pred_cb)
    Y_Proba_cb = cb.predict_proba(X_Test)
    auc_score_cb = roc_auc_score(Y_Test, Y_Proba_cb, multi_class='ovr', average='macro')
    models_auc_scores.append(auc_score_cb)

def pred_pos_lgbm(X_Train, Y_Train, X_Test, Y_Test):
    from lightgbm import LGBMClassifier
    lgbm = LGBMClassifier(max_depth=15)
    lgbm.fit(X_Train, Y_Train)
    Y_Pred_lgbm = lgbm.predict(X_Test)
    models_names.append("light GBM")
    models_scores.append(lgbm.score(X_Test, Y_Test))
    classification_report_fun("light GBM", Y_Test, Y_Pred_lgbm)
    Y_Proba_lgbm = lgbm.predict_proba(X_Test)
    auc_score_lgbm = roc_auc_score(Y_Test, Y_Proba_lgbm, multi_class='ovr', average='macro')
    models_auc_scores.append(auc_score_lgbm)


def mostSimilar(jogador, data):
    jogador = jogador
    sel_cols = list(data.select_dtypes(include='int64'))
    sel_cols = sel_cols[1:]

    for i in data[sel_cols]:
        ma = max(data[i])
        mi = min(data[i])
        data[i] = (data[i] - mi) / (ma - mi)

    A = np.array(data[sel_cols])
    df_jogador = data[data['Name'] == jogador]
    B = np.array(df_jogador[sel_cols]).flatten()

    from numpy.linalg import norm
    cosine = np.dot(A, B) / (norm(A, axis=1) * norm(B))
    data['Similaridade'] = cosine
    print(data[['Name', 'Similaridade']].sort_values(by='Similaridade', ascending=False).head(20))


def pred_pos_xgb(X_Train, Y_Train, X_Test, Y_Test):
    from xgboost import XGBClassifier
    xgb = XGBClassifier()
    xgb.fit(X_Train, Y_Train)
    Y_Pred_xgb = xgb.predict(X_Test)
    models_names.append("XGB")
    models_scores.append(xgb.score(X_Test, Y_Test))
    classification_report_fun("XGB", Y_Test, Y_Pred_xgb)
    Y_Proba_xgb = xgb.predict_proba(X_Test)
    auc_score_xgb = roc_auc_score(Y_Test, Y_Proba_xgb, multi_class='ovr', average='macro')
    models_auc_scores.append(auc_score_xgb)

#limpar e preprocessar
data = players
data.drop(data[data["ContractUntil"].isnull()].index, axis=0, inplace=True)
le = preprocessing.LabelEncoder()
data["PreferredFoot"] = le.fit_transform(data["PreferredFoot"])
data["AttackingWorkRate"] = le.fit_transform(data["AttackingWorkRate"])
data["DefensiveWorkRate"] = le.fit_transform(data["DefensiveWorkRate"])
data["Club"] = le.fit_transform(data["Club"])
#combinar posicoes
merge_pos = {'LWB': 'LB', 'RWB': 'RB', 'CF': 'ST', 'CAM': 'CM', 'CDM': 'CM'}
data = data.replace({'BestPosition': merge_pos})
#mapear variavel resposta e excluir
mapping = {'GK': 0, 'CB': 1, 'LB': 2, 'RB': 3, 'CM': 4,
               'LM': 5, 'LW': 6, 'RM': 7, 'RW': 8, 'ST': 9}
data = data.replace({'BestPosition': mapping})
# dividir treino e resposta
X = data.drop(["BestPosition", 'GroupPosition', "Nationality", "ID"], axis=1)
Y = pd.DataFrame(data["BestPosition"])

from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.20, random_state=42)
X_Train = X_Train.drop(["Name"], axis=1)
test_names = X_Test["Name"]
X_Test = X_Test.drop(["Name"], axis=1)

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
oversample = RandomOverSampler(sampling_strategy='not majority')
X_Train, Y_Train = oversample.fit_resample(X_Train, Y_Train)
# visualizacao do oversampling
# sns.countplot(x="BestPosition", data = Y_Train)
# plt.show()
# print(f' X_shape: {X_Train.shape} \n y_shape: {Y_Train.shape}')
# preprocessamento min max
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_Train = mms.fit_transform(X_Train)
X_Test = mms.fit_transform(X_Test)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score

def gerar_todos_modelos(X_Train, Y_Train, X_Test, Y_Test):
    pred_pos_lr(X_Train, Y_Train, X_Test, Y_Test)
    pred_pos_rf(X_Train, Y_Train, X_Test, Y_Test)
    pred_pos_dtc(X_Train, Y_Train, X_Test, Y_Test)
    pred_pos_abc(X_Train, Y_Train, X_Test, Y_Test)
    pred_pos_knn(X_Train, Y_Train, X_Test, Y_Test)
    pred_pos_cbc(X_Train, Y_Train, X_Test, Y_Test)
    pred_pos_lgbm(X_Train, Y_Train, X_Test, Y_Test)
    pred_pos_xgb(X_Train, Y_Train, X_Test, Y_Test)

def prever_top(melhor_modelo, data):
    top = data.sort_values(by=["Overall"], ascending=False).head(500)
    top_pos = top["BestPosition"]
    top_names = top["Name"]
    top = top.drop(["Name", "GroupPosition", "BestPosition", "Nationality", "ID"], axis=1)
    from sklearn.preprocessing import MinMaxScaler
    mms = MinMaxScaler()
    top = mms.fit_transform(top)
    top = pd.DataFrame(top)

    if melhor_modelo == 'XGB':
        from xgboost import XGBClassifier
        modelo = XGBClassifier()
        modelo.fit(X_Train, Y_Train)
    elif melhor_modelo == 'Random Forest':
        from sklearn.ensemble import RandomForestClassifier
        modelo = RandomForestClassifier(criterion='gini', n_estimators=50)
        modelo.fit(X_Train, Y_Train)
    elif melhor_modelo == 'light GBM':
        from lightgbm import LGBMClassifier
        modelo = LGBMClassifier(max_depth=15)
        modelo.fit(X_Train, Y_Train)
    elif melhor_modelo == 'Logistic Regression':
        from sklearn.linear_model import LogisticRegression
        modelo = LogisticRegression()
        modelo.fit(X_Train, Y_Train)
    elif melhor_modelo == 'CatBoost':
        from catboost import CatBoostClassifier
        modelo = CatBoostClassifier(max_depth=10, iterations=10, learning_rate=0.3)
        modelo.fit(X_Train, Y_Train)
    elif melhor_modelo == 'Decision Tree':
        from sklearn.tree import DecisionTreeClassifier
        modelo = DecisionTreeClassifier(max_features=39, max_depth=10)
        modelo.fit(X_Train, Y_Train)
    elif melhor_modelo == 'KNN':
        from sklearn.neighbors import KNeighborsClassifier
        modelo = KNeighborsClassifier(n_neighbors=8)
        modelo.fit(X_Train, Y_Train)
    elif melhor_modelo == 'Adaboost':
        from sklearn.ensemble import AdaBoostClassifier
        modelo = AdaBoostClassifier()
        modelo.fit(X_Train, Y_Train)

    nome = []
    pos = []

    for i in range(top.shape[0]):
        pred_pos = modelo.predict(top.iloc[[i]])[0]
        pred = list(mapping.keys())[list(mapping.values()).index(pred_pos)]
        true_pos = list(mapping.keys())[list(mapping.values()).index(top_pos.iloc[i])]
        nome.append(top_names.iloc[i])
        pos.append(pred)

    df = pd.DataFrame({'Nome': nome, 'Posicao': pos})
    gk = df.loc[df['Posicao'] == 'GK'].head(1)
    lb = df[df['Posicao'].str.contains('LB','LWB')].head(1)
    rb = df[df['Posicao'].str.contains('RB','RWB')].head(1)
    cb = df.loc[df['Posicao'] == 'CB'].head(2)
    cm = df.loc[df['Posicao'] == 'CM'].head(2)
    lw = df[df['Posicao'].str.contains('LM', 'LW')].head(1)
    rw = df[df['Posicao'].str.contains('RM', 'RW')].head(1)
    st = df.loc[df['Posicao'] == 'ST'].head(2)

    escalacao = pd.concat([gk,lb,cb,rb,cm,lw,rw,st], ignore_index=True)
    print(escalacao)

    Y_predict = modelo.fit(X_Train, Y_Train).predict(X_Test)
    cm = confusion_matrix(Y_Test, Y_predict, normalize='true')
    hm = sns.heatmap(cm, annot=True, linewidths=.5, cmap='Blues', xticklabels=mapping, yticklabels=mapping)
    hm.set_title(label='Matriz de confusão', fontsize=20)
    plt.show()

    return modelo


gerar_todos_modelos(X_Train, Y_Train, X_Test, Y_Test)
comp = pd.DataFrame()
comp['Nome'] = models_names
comp['Score'] = models_scores
comp = comp.sort_values('Score', ascending=False, ignore_index=True)
print(comp)
melhor_modelo = comp['Nome'][0]
comp_auc = pd.DataFrame()
comp_auc['name'] = models_names
comp_auc['score'] = models_auc_scores
print(comp_auc)

modelo = prever_top(melhor_modelo, data)
gerar_todos_graficos(players)
mostSimilar('K. Mbappé', data)


