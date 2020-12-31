import numpy as np
from sklearn import svm
from sklearn.model_selection import  KFold, GridSearchCV


X = np.loadtxt('X.txt',delimiter=',')
Y_cat = np.loadtxt('Y_cat.txt')


C_range = np.arange(1,10.1,0.1)
gamma_range = np.logspace(-5,1,7)
param_grid = [dict(C=C_range, gamma=gamma_range)]

# Parameters
K =  10      # k-fold parameter

# Cross-validation
cv = KFold(n_splits=K)
grid = GridSearchCV(svm.SVC(kernel='rbf',cache_size=1000), param_grid=param_grid, cv=cv, n_jobs=-1,verbose=1)
grid.fit(X,y)


best_std_score = grid.cv_results_['std_test_score'][grid.best_index_]

# Results:
print("Best parameters: get_ipython().run_line_magic("s", " \nAccuracy: %0.3f \u00B1 %0.3f\"")
      % (grid.best_params_, grid.best_score_, best_std_score))
