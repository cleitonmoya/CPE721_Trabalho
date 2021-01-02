import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from termcolor import colored
from scipy.stats import probplot, norm, mannwhitneyu, kstest, chi2_contingency 
from sklearn.preprocessing import StandardScaler


df_mat = pd.read_csv('datasets/student-mat.csv', sep=';')
df_por = pd.read_csv('datasets/student-por.csv', sep=';')


df_por


df_mat


var_bin = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
var_nom = ['Mjob', 'Fjob', 'reason', 'guardian']
var_num = ['age', 'absences', 'G1', 'G2', 'G3']
var_ord = ['Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health']
var_int = var_ord + var_num # variáveis inteiras


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


alpha=0.05
U,p = mannwhitneyu(df_por['G3'], df_mat['G3'],use_continuity=False)
print('Estatística:', U)
print('p-valor:', p)
if p<alpha:
    print(colored('Rejeitamos H0','red'))
else:
    print(colored('Não rejeitamos H0','green'))


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


alpha=0.05
s,p = kstest(df_por['G1'], norm(df_por['G1'].mean(), df_por['G1'].std()).cdf)
print('Estatística:', s)
print('p-valor:', p)
if p<alpha:
    print(colored('Rejeitamos H0','red'))
else:
    print(colored('Não rejeitamos H0','green'))


df = df_por.copy()
df_num = df[num[:-3]]


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


df['g3_cat'] = (df.G3 > 10).astype(int).replace({0:-1})
Y_cla = df['g3_cat'].values.reshape(len(Y_cla),1)


fig, ax1 = plt.subplots(figsize=(10,10))
df.hist(var_bin + ['g3_cat'],ax=ax1)
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


df_nom = pd.DataFrame(index=df.index)
df_nom['Mjob']=pd.get_dummies(df['Mjob']).replace({0:-1}).values.tolist()
df_nom['Fjob']=pd.get_dummies(df['Fjob']).replace({0:-1}).values.tolist()
df_nom['reason']=pd.get_dummies(df['reason']).replace({0:-1}).values.tolist()
df_nom['guardian']=pd.get_dummies(df['guardian']).replace({0:-1}).values.tolist()

#Mjob = df_por[['Mjob']]
#from sklearn.preprocessing import OneHotEncoder
#enc = OneHotEncoder()
#enc.fit_transform(Mjob).toarray()


fig, ax1 = plt.subplots(figsize=(10,10))
df.hist(var_num,ax=ax1)
plt.show()
plt.tight_layout()


plt.figure(figsize=(10,10))
Cor = df[var_int].corr(method='kendall')
mask = np.triu(np.ones_like(Cor, dtype=bool)) # Generate a mask for the upper triangle
ax = sns.heatmap(Cor, mask=mask, vmin=-1, vmax=+1, cmap='RdBu', linewidths=1, square=True, cbar_kws={"shrink": 0.8}) 
plt.show()


[{k:v.dropna().sort_values(ascending=False).to_dict()} for k,v in Cor[(abs(Cor)>0.7) & (Corget_ipython().getoutput("=1)].dropna(how='all').iterrows()]")


plt.scatter(df['G2'], df['G3'])
plt.show()


plt.scatter(df['Dalc'], df['Walc'])
plt.show()


[{k:v.dropna().sort_values(ascending=False).to_dict()} for k,v in Cor[(abs(Cor)>=0.30) & (abs(Cor)<0.49) & (Corget_ipython().getoutput("=1)].dropna(how='all').iterrows()]")


[{k:v.dropna().sort_values(ascending=False).to_dict()} for k,v in Cor[(abs(Cor)<0.05) & (Corget_ipython().getoutput("=1)][['G3']].dropna(how='all').iterrows()]")


[{k:v.dropna().sort_values(ascending=False).to_dict()} for k,v in Cor[(abs(Cor)>=0.7) & (Corget_ipython().getoutput("=1)][['G3']].dropna(how='all').iterrows()]")


plt.figure(figsize=(10,10))
Cor_bin_num = df[var_bin+['g3_cat']+var_num].corr(method='pearson')
mask = np.triu(np.ones_like(Cor_bin_num, dtype=bool)) # Generate a mask for the upper triangle
ax = sns.heatmap(Cor_bin_num, mask=mask, vmin=-1, vmax=+1, cmap='RdBu', linewidths=1, square=True, cbar_kws={"shrink": 0.8}) 
plt.show()


[{k:v.dropna().sort_values(ascending=False).to_dict()} for k,v in Cor_bin_num[(abs(Cor_bin_num)>0.7) & (Cor_bin_numget_ipython().getoutput("=1)].dropna(how='all').iterrows()]")


[{k:v.dropna().sort_values(ascending=False).to_dict()} for k,v in Cor_bin_num[(abs(Cor_bin_num)<0.05) & (Cor_bin_numget_ipython().getoutput("=1)][['g3_cat']].dropna(how='all').iterrows()]")


[{k:v.dropna().sort_values(ascending=False).to_dict()} for k,v in Cor_bin_num[(abs(Cor_bin_num)<0.05) & (Cor_bin_numget_ipython().getoutput("=1)][['G3']].dropna(how='all').iterrows()]")


dfCont = df_por[var_nom].copy()
dfCont['g3_cat'] = (df.G3 > 10).astype(int)
dfCont


alpha=0.05
cont1 = pd.crosstab(dfCont['Mjob'], dfCont['g3_cat'], margins=False)
chi2, p, dof, ex = chi2_contingency(cont1.to_numpy(), correction=False)
print('p-valor:', p)
if p<alpha:
    print(colored('Rejeitamos H0, variáveis possuem dependência linear','green'))
else:
    print(colored('Não rejeitamos H0','rede'))


cont1 = pd.crosstab(dfCont['Fjob'], dfCont['g3_cat'], margins=False)
chi2, p, dof, ex = chi2_contingency(cont1.to_numpy(), correction=False)
print('p-valor:', p)
if p<alpha:
    print(colored('Rejeitamos H0, variáveis possuem dependência linear','green'))
else:
    print(colored('Não rejeitamos H0','rede'))


cont1 = pd.crosstab(dfCont['reason'], dfCont['g3_cat'], margins=False)
chi2, p, dof, ex = chi2_contingency(cont1.to_numpy(), correction=False)
print('p-valor:', p)
if p<alpha:
    print(colored('Rejeitamos H0, variáveis possuem dependência linear','green'))
else:
    print(colored('Não rejeitamos H0','rede'))


cont1 = pd.crosstab(dfCont['guardian'], dfCont['g3_cat'], margins=False)
chi2, p, dof, ex = chi2_contingency(cont1.to_numpy(), correction=False)
print('p-valor:', p)
if p<alpha:
    print(colored('Rejeitamos H0, variáveis possuem dependência linear','green'))
else:
    print(colored('Não rejeitamos H0','rede'))


alpha=0.05
cont1 = pd.crosstab(dfCont['Mjob'], dfCont['Fjob'], margins=False)
chi2, p, dof, ex = chi2_contingency(cont1.to_numpy(), correction=False)
print('p-valor:', p)
if p<alpha:
    print(colored('Rejeitamos H0, variáveis possuem dependência linear','green'))
else:
    print(colored('Não rejeitamos H0','rede'))


cont1


var_bin_exc = ['famsize', 'Pstatus', 'famsup' 'nursey', 'schoolsup', 'paid']
var_nom_exc = ['Fjob']

var_bin2 = list(set(var_bin)-set(var_bin_exc))
var_nom2 = list(set(var_nom)-set(var_nom_exc))
var_ord2 = var_ord
var_num2 = var_num[:-3]
var_int2 = var_int[:-3]

df_X = pd.DataFrame(index=df.index)
df_X = df_X.join([df[var_bin2], df_nom[var_nom2], df[var_ord2], df[var_num2]])

print(var)


len(var)


scalerG3 = StandardScaler().fit(df[['G3']])
Y_reg = scalerG3.transform(df[['G3']])


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

num_pipeline = Pipeline([('std_scaler', StandardScaler())])
full_pipeline = ColumnTransformer([("num", num_pipeline, var_num2)])

df_X[var_num2] = pd.DataFrame(data=full_pipeline.fit_transform(df), columns = var_num2)


df_X


oh_Mjob=pd.get_dummies(df_por['Mjob']).replace({0:-1}).values
oh_reason=pd.get_dummies(df_por['reason']).replace({0:-1}).values
oh_guardian=pd.get_dummies(df_por['guardian']).replace({0:-1}).values

X = np.hstack((df_X[var_bin2 + var_ord2 + var_num2].values, oh_Fjob, oh_reason, oh_guardian))
X.shape


np.savetxt('datasets/X.txt',X,delimiter=',')
np.savetxt('datasets/Y_cla.txt',Y_cla)
np.savetxt('datasets/Y_reg.txt',Y_reg)
