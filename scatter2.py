import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

df=pd.read_csv("bankdata.csv")
x=df.drop('Class',axis=1)
y=df['Class']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
print(x_train)

mlp=MLPClassifier()
# print(mlp)
# print(y_test)


e=mlp.fit(x_train,y_train)
pred=mlp.predict(x_test)
# print(pred)


con=confusion_matrix(y_test,pred)
print(con)
report=classification_report(y_test,pred)
print("Reports::",report)
score=accuracy_score(y_test,pred)
print("Score::",score)