import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df_mat = pd.read_csv('student-mat.csv', sep=';')
df_por = pd.read_csv('student-por.csv', sep=';')


df_por


df_mat


var_bin = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
var_nom = ['Mjob', 'Fjob', 'reason', 'guardian']
var_num = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']


df_por.isnull().values.any() or df_mat.isnull().values.any()


df_por.describe()


df_mat.describe()


fig, axs = plt.subplots(1,2, figsize=(10,5), sharey=True)
axs[0].set_title('Português')
df_por.boxplot(['G1', 'G2', 'G3'], grid=False,ax=axs[0])
axs[1].set_title('Matemática')
df_mat.boxplot(['G1', 'G2', 'G3'], grid=False, ax=axs[1])
plt.show()


df_por[(df_por['G3']<=1) & (df_por['G1']get_ipython().getoutput("=0) & (df_por['G2']!=0)].head()")


print("Número de registros inconsistentes: ",len(df_mat[(df_mat['G3']<=1) & (df_mat['G1']get_ipython().getoutput("=0) & (df_mat['G2']!=0)]))")


df_mat[(df_mat['G3']<=1) & (df_mat['G1']get_ipython().getoutput("=0) & (df_mat['G2']!=0)].head()")


df_por.loc[(df_por['G3'] <=1), 'G3'] = df_por[df_por['G3']<=1].apply(lambda x: np.round((x['G1']+x['G2'])/2,0),axis=1).astype(int)


df_mat.loc[(df_mat['G3'] <=1), 'G3'] = df_mat[df_mat['G3']<=1].apply(lambda x: np.round((x['G1']+x['G2'])/2,0),axis=1).astype(int)


df_por.boxplot('G3', grid=False)
plt.show()


df_por[(df_por['G3']<=3)]


fig, ax = plt.subplots(1,3,figsize=(15,5))
sns.ecdfplot(df_por['G1'],ax=ax[0], label='Por')
sns.ecdfplot(df_mat['G1'],ax=ax[0], label='Mat')
sns.ecdfplot(df_por['G2'],ax=ax[1], label='Por')
sns.ecdfplot(df_mat['G2'],ax=ax[1], label='Mat')
sns.ecdfplot(df_por['G3'],ax=ax[2], label='Por')
sns.ecdfplot(df_mat['G3'],ax=ax[2], label='Mat')
ax[0].legend()
plt.show()


from scipy.stats import mannwhitneyu
from termcolor import colored
alpha=0.05
U,p = mannwhitneyu(df_por['G3'], df_mat['G3'],use_continuity=False)
print('Estatística:', U)
print('p-valor:', p)
if p<alpha:
    print(colored('Rejeitamos H0','red'))
else:
    print(colored('Não rejeitamos H0','green'))


from scipy.stats import probplot
fig, axs = plt.subplots(2, 6, figsize=(14,5))
axs[0,0].set_title('Por - G1')
axs[0,0].hist(df_por['G1'], alpha=0.5, edgecolor='white',linewidth=0.5)
axs[0,1].set_title('Por - G2')
axs[0,1].hist(df_por['G2'], alpha=0.5, edgecolor='white',linewidth=0.5)
axs[0,2].set_title('Por - G3')
axs[0,2].hist(df_por['G3'], alpha=0.5, edgecolor='white',linewidth=0.5)

axs[0,3].set_title('Mat - G1')
axs[0,3].hist(df_mat['G1'], alpha=0.5, edgecolor='white',linewidth=0.5)
axs[0,4].set_title('Mat - G2')
axs[0,4].hist(df_mat['G2'], alpha=0.5, edgecolor='white',linewidth=0.5)
axs[0,5].set_title('Mat - G3')
axs[0,5].hist(df_mat['G3'], alpha=0.5, edgecolor='white',linewidth=0.5)

_ = probplot(df_por['G1'], plot=axs[1,0])
_ = probplot(df_por['G2'], plot=axs[1,1])
_ = probplot(df_por['G3'], plot=axs[1,2])
_ = probplot(df_mat['G1'], plot=axs[1,3])
_ = probplot(df_mat['G2'], plot=axs[1,4])
_ = probplot(df_mat['G3'], plot=axs[1,5])
axs[1,0].set_title('Por - G1')
axs[1,1].set_title('Por - G2')
axs[1,2].set_title('Por - G3')
axs[1,3].set_title('Mat - G1')
axs[1,4].set_title('Mat - G2')
axs[1,5].set_title('Mat - G3')

plt.tight_layout()


from scipy.stats import kstest, norm
from termcolor import colored
alpha=0.05
s,p = kstest(df_por['G1'], norm(df_por['G1'].mean(), df_por['G1'].std()).cdf)
print('Estatística:', s)
print('p-valor:', p)
if p<alpha:
    print(colored('Rejeitamos H0','red'))
else:
    print(colored('Não rejeitamos H0','green'))


df = df_por.copy()


df['school'].replace({'GP': -1, 'MS': 1}, inplace=True)
df['sex'].replace({'F': -1, 'M': 1}, inplace=True)
df['address'].replace({'R': -1, 'U': 1}, inplace=True)
df['famsize'].replace({'LE3': -1, 'GT3': 1}, inplace=True)
df['Pstatus'].replace({'A': -1, 'T': 1}, inplace=True)
df['schoolsup'].replace({'no': -1, 'yes': 1}, inplace=True)
df['famsup'].replace({'no': -1, 'yes': 1}, inplace=True)
df['paid'].replace({'no': -1, 'yes': 1}, inplace=True)
df['activities'].replace({'no': -1, 'yes': 1}, inplace=True)
df['nursery'].replace({'no': -1, 'yes': 1}, inplace=True)
df['higher'].replace({'no': -1, 'yes': 1}, inplace=True)
df['internet'].replace({'no': -1, 'yes': 1}, inplace=True)
df['romantic'].replace({'no': -1, 'yes': 1}, inplace=True)


fig, ax1 = plt.subplots(figsize=(10,10))
df.hist(var_bin,ax=ax1)
plt.show()
plt.tight_layout()


fig, axs = plt.subplots(2,2, figsize=(7,7))
axs[0,0].hist(df['Mjob'])
axs[0,0].set_title('Mjob')
axs[0,1].hist(df['Fjob'])
axs[0,1].set_title('Fjob')
axs[1,0].hist(df['reason'])
axs[1,0].set_title('reason')
axs[1,1].hist(df['guardian'])
axs[1,1].set_title('guardian')
plt.show()


df['Mjob']=pd.get_dummies(df['Mjob']).replace({0:-1}).values.tolist()
df['Fjob']=pd.get_dummies(df['Fjob']).replace({0:-1}).values.tolist()
df['reason']=pd.get_dummies(df['reason']).replace({0:-1}).values.tolist()
df['guardian']=pd.get_dummies(df['guardian']).replace({0:-1}).values.tolist()

#Mjob = df_por[['Mjob']]
#from sklearn.preprocessing import OneHotEncoder
#enc = OneHotEncoder()
#enc.fit_transform(Mjob).toarray()


fig, ax1 = plt.subplots(figsize=(10,10))
df.hist(var_num,ax=ax1)
plt.show()
plt.tight_layout()


plt.figure(figsize=(10,10))
import seaborn as sns
Cor = df[var_num].corr(method='kendall')
mask = np.triu(np.ones_like(Cor, dtype=bool)) # Generate a mask for the upper triangle
ax = sns.heatmap(Cor, mask=mask, vmin=-1, vmax=+1, cmap='RdBu', linewidths=1, square=True, cbar_kws={"shrink": 0.8}) 
plt.show()


[{k:v.dropna().sort_values(ascending=False).to_dict()} for k,v in Cor[(abs(Cor)>0.5) & (Corget_ipython().getoutput("=1)].dropna(how='all').iterrows()]")


[{k:v.dropna().sort_values(ascending=False).to_dict()} for k,v in Cor[(abs(Cor)>=0.30) & (abs(Cor)<0.49) & (Corget_ipython().getoutput("=1)].dropna(how='all').iterrows()]")


[{k:v.dropna().sort_values(ascending=False).to_dict()} for k,v in Cor[(abs(Cor)<0.05) & (Corget_ipython().getoutput("=1)][['G3']].dropna(how='all').iterrows()]")


[{k:v.dropna().sort_values(ascending=False).to_dict()} for k,v in Cor[(abs(Cor)>=0.25) & (Corget_ipython().getoutput("=1)][['G3']].dropna(how='all').iterrows()]")


from sklearn.preprocessing import StandardScaler
scalerG3 = StandardScaler().fit(df[['G3']])
Y_num = scalerG3.transform(df[['G3']])
Y_cat = (df.G3 > 10).astype(int).replace({0:-1}).values


from sklearn.pipeline import Pipeline

num_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])

# cat_pipeline = Pipeline([
#     ('one_hot', OneHotEncoder(sparse=False)),
# ])

from sklearn.compose import ColumnTransformer
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, var_num[:-3]),
])

df2 = pd.DataFrame(
     data = full_pipeline.fit_transform(df_por),
     columns = var_num[:-3])


oh_Mjob=pd.get_dummies(df_por['Mjob']).replace({0:-1}).values
oh_Fjob=pd.get_dummies(df_por['Fjob']).replace({0:-1}).values
oh_reason=pd.get_dummies(df_por['reason']).replace({0:-1}).values
oh_guardian=pd.get_dummies(df_por['guardian']).replace({0:-1}).values
X = np.hstack((df2.values, df[var_bin].values, oh_Mjob, oh_Fjob, oh_reason, oh_guardian))


np.savetxt('X.txt',X,delimiter=',')
np.savetxt('Y_cat.txt',Y_cat)
np.savetxt('Y_num.txt',Y_num)
