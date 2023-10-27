import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error



#"C:\Users\gaus_\Downloads\train.csv"
#fileTrain = Path("C:\Users\gaus_\Downloads\train.csv")
#fileTest = Path("C:\Users\gaus_\Downloads\test.csv")
df = pd.read_csv(r"C:\Users\gaus_\Downloads\train.csv")

dfNumbericals= df.select_dtypes(['number']) 
dfObjects=df.select_dtypes(['object'])

# le = OneHotEncoder(sparse_output=True)
print(dfNumbericals.shape)
print(dfObjects.shape)
dfNumbericals =dfNumbericals.join(dfObjects)
print(dfNumbericals.shape)
dfNumbericals =dfNumbericals.join(pd.get_dummies(dfObjects,dummy_na=True))    
print(dfNumbericals.shape)
dfCorr =dfNumbericals.corr(numeric_only = True)

salesPrice=dfCorr.SalePrice

#print(type(dfCorr))
#print("saleprice here: ")
print(salesPrice)


print(salesPrice.shape) 
for (label,value) in salesPrice.items() :
    print("ITEM: ", label)
    print("ROW: ", value)
    if (value < 0.2 and value > -0.2 or pd.isna(value)):
        print("Item dropped: ", label )
        salesPrice=salesPrice.drop(label)
        #print("Item dropped: ", label )

print(salesPrice.shape)  
print(salesPrice)

#dfNumbericals.drop("Id")
for column in dfNumbericals.columns:
    if not(column in salesPrice):
        print("dropping ", column)
        del dfNumbericals[column]
        #dfNumbericals=dfNumbericals.drop(column)

dfNumbericals=dfNumbericals.fillna(dfNumbericals.mean())

y = dfNumbericals.pop("SalePrice")
X_train, X_test, y_train, y_test = train_test_split(dfNumbericals, y, test_size=0.33, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)

predictions = reg.predict(X_test)
print(y_test.to_numpy())
score=mean_squared_error(y_test,predictions)
print(score)
#print(dfNumbericals.shape)
#for ( column, x) in dfCorr.items():
 #   print("column, ", column)
  #  print("x, ", x.values)

#street = pd.get_dummies(df.Street,dummy_na=True) #samma kategorinamn, ÄNDRA!!!!!!!
#alley = pd.get_dummies(df.Alley) #samma kategornamn, ÄNDRA NU!
#MSZoning = pd.get_dummies(df.MSZoning)
#LotShape = pd.get_dummies(df.LotShape)
#LandContour = pd.get_dummies(df.LandContour)
#saleprice = df.SalePrice


#print(street)
#alley = alley.rename(columns={'Pave':'PaveAlley','Grvl':'GrvlAlley'})
#street = street.join(LotShape).join(LandContour).join(alley).join(MSZoning).join(df.SalePrice)

#print("street: ", street)
#print(alley)
#print(MSZoning)
#print(LandContour)
#print('SPEARMAN:')
#print(street.corr(method='pearson'))
#print('PEARSON:')
#print(street.corr(method='spearman'))
#print('KENDALL:')
#print(street.corr(method='kendall'))

# maxz = (df.corr()).SalePrice.max()
# df = (df.corr(method='spearman')).sort_values(by='SalePrice')
# print(df)





#dfT.to_csv("C:/Users/johan/OneDrive/Skrivbord/assignment/house_prices_ML/new_CSV.csv")

# df.drop("MSZoning",axis=1,inplace=True)
# print(df.head())

# df["MSZoning"]=labelMSZoning

# MSZoning= df.pop('MSZoning')
# extract data for X and y for both test and train modules
# print(df.head())
# df.to_csv('new _data.csv', index = True)
# dfTrain = df.iloc[:1000]         #train data from zero to 1000
# dfTest = df.iloc[1000:]         #test data from 1000 to end of dataFrame



# print(df.columns)

# train data 
# trainY = dfTrain.pop('SalePrice')
# cvxMSSubClass = dfTrain.pop('MSSubClass')
# enc = OneHotEncoder()s
# enc.fit(dfTrain) 

# trainX = dfTrain


# test data
# testY = dfTest.pop('SalePrice')
# testX = dfTest


# test = size.to_frame().join(price)
# reg = LinearRegression().fit(trainX, trainY)


# pred = reg.predict(testX)
# plt.scatter(pred,testY)
# plt.show()
#print('WARNING: Data corruption detected')
#inn = input('Are you sure you want to delete Windows32.exe? (y/n)')
#if inn=='y':
#    print('Deleting files...')
#    time.sleep(1)
#    print('Done')
#    print('1 file(s) deleted')
#else:
#    print('Deleting files...')
#    time.sleep(1)
#    print('Done')
#    print('1 file(s) deleted')


#test.plot(y ='SalePrice', x='LotArea', kind='scatter')






