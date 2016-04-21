__author__ = 'JaclynNguyen'
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.datasets import make_hastie_10_2
import matplotlib.pyplot as plt

x, y = make_hastie_10_2(n_samples = 1000)
x_t, y_t = make_hastie_10_2(n_samples = 1000)




final_training_error = []
final_test_error = []
for i in [1, 50,100,200, 300, 500, 1000]:
    w = np.repeat(1/float(len(x)), len(x))
    g_m = []
    alphas = []
    training_error = []
    for j in range(i):
        model = DecisionTreeClassifier(max_depth=1)
        fitted = model.fit(x, y, sample_weight= w)
        g_m.append(fitted)
        predictions = fitted.predict(x)
        identity = np.array([1 if predictions[i] != y[i] else 0 for i in range(len( predictions))])
        err = np.sum(w * identity)/np.sum(w)
        a = np.log((1 - err)/err)
        alphas.append(a)
        w = w * np.exp(a * identity)


        gm_predict = a * predictions
        training_error.append(gm_predict)


    gm_output = sum(training_error)
    results = np.sign(gm_output)
    accuracy = np.sum([1 if results[i] == y[i] else 0 for i in range(len(results))])/float(len(results))
    final_training_error.append(1-accuracy)


    test_error = []
    for m in range(len(g_m)):
        test_prediction = g_m[m].predict(x_t)
        test_predictions = alphas[m] * test_prediction
        test_error.append(test_predictions)

    gm_testoutput = sum(test_error)
    testresults = np.sign(gm_testoutput)
    testaccuracy = np.sum([1 if testresults[l] == y_t[l] else 0 for l in range(len(testresults))])/float(len(testresults))
    final_test_error.append(1-testaccuracy)


fig = plt.figure()
line1, = plt.plot([1, 50,100,200, 300, 500, 1000], final_training_error, 'r', label = 'Training')
line2, = plt.plot([1, 50,100,200, 300, 500, 1000], final_test_error, 'b', label = 'Test')
plt.legend(handles = [line1,line2])
plt.title('Error Rate for Adaboost')
plt.ylabel('Test Error')
plt.xlabel('Number of Iterations')
plt.savefig('AdaboostError.png')
plt.show()


gbm_training_error = []
gbm_test_error = []

for g in [1, 50,100,200, 300, 500, 1000]:

    gbm_mdl = GradientBoostingClassifier(max_depth = 1, n_estimators=g)
    gbm_fit = gbm_mdl.fit(x, y)
    gbm_score = gbm_fit.score(x,y)
    gbm_training_error.append(1-gbm_score)


    gbm_score = gbm_fit.score(x_t,y_t)
    gbm_test_error.append(1-gbm_score)


fig = plt.figure()
line1, = plt.plot([1, 50,100,200, 300, 500, 1000], gbm_training_error, 'r', label = 'Training')
line2, = plt.plot([1, 50,100,200, 300, 500, 1000], gbm_test_error, 'b', label = 'Test')
plt.legend(handles = [line1,line2])
plt.title('Error Rate for Gradient Boosting')
plt.ylabel('Test Error')
plt.xlabel('Number of Iterations')
plt.savefig('GBMError.png')
plt.show()
