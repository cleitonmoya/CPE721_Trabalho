X = np.loadtxt('X.txt',delimiter=',')
Y_cat = np.loadtxt('Y_cat.txt')


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix 
from neupy import algorithms, layers

nn = layers.join(
        layers.Input(43),
        layers.Tanh(150),
        layers.Tanh(1),
)

X_tr, X_val, y_tr, y_val = train_test_split(X, 
                                          y, 
                                          test_size = 0.4,
                                          random_state = 40)
y_tr = y_tr.reshape(len(y_tr),1)
y_val = y_val.reshape(len(y_val),1)


optimizer = algorithms.LevenbergMarquardt(nn, loss='mse', shuffle_data=False, verbose=True)


#optimizer = algorithms.Momentum(nn, loss='mse', momentum=0.99, step=0.01, shuffle_data=False, verbose=True, batch_size=None)


#optimizer = algorithms.GradientDescent(nn, step=0.1, loss='mse', shuffle_data=False, show_epoch=5, verbose=False, batch_size=None)


optimizer.train(X_tr, y_tr, X_val, y_val, epochs=10)


optimizer.plot_errors()


g = np.sign(optimizer.predict(X_val))


from sklearn.metrics import accuracy_score, confusion_matrix 
Acc = accuracy_score(y_val,g)
Cm = confusion_matrix(y_val,g)
print("Acurácia:", Acc)
print("Matriz de confusão:")
print(Cm)
