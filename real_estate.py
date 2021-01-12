import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

dataset1 = pd.read_csv("Bengaluru_House_Data.csv")

row = dataset1.drop(["area_type","availability","society", "balcony"], axis = "columns")


'''
rom sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
row[:, 6] = imputer.fit_transform(row[:,6]) 
row.drop()
'''

row.isnull().sum()
row1 = row.dropna()
row1.isnull().sum()


row1["bhk"] = row1["size"].apply(lambda x : int(x.split()[0]))
row1["bhk"].unique()


row1["total_sqft"].unique()


print(row1[row1.bhk>12])


def size(a):
    try:
        float(a)
    except:
        return False
    return True


print(row1[~row1["total_sqft"].apply((size))].head(15))

data = row1[~row1["total_sqft"].apply(size)]



def rectify(a):
    x = a.split(" - ")
    if len(x) == 2:
        return (float(x[0])+float(x[1]))/2
    try:
        return float(a)
    except:
        return None
    
    
    
row2 = row1.copy()

print(rectify("85 arc"))

row2["total_sqft"] = row2["total_sqft"].apply(rectify)
row2.isnull().sum()
row2 = row2.dropna()


row3 = row2.copy()

row3["price_per_sqft"] = (row3["price"]*100000)/row3["total_sqft"]
row3["size"]
        
    
len(row2.location.unique())

row3.location = row3.location.apply(lambda x : x.strip())

location_count = row3.groupby("location")["location"].agg("count").sort_values(ascending = False)


location_less_than_10 = location_count[location_count<=10]



def location_category(x):
    if x in location_less_than_10:
        return "other_location"
    else:
        return x

row4 = row3.copy()

row4.location = row4.location.apply(location_category)


len(row4.location.unique())

print(row4[row4.total_sqft/row4.bhk<300])


row5 = row4[~(row4.total_sqft/row4.bhk<300)]



print(row5.price_per_sqft.describe())


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key,subdf in df.groupby("location"):
        m = np.mean(subdf.price_per_sqft)
        std = np.std(subdf.price_per_sqft)
        reduce_df = subdf[(subdf.price_per_sqft>(m-std)) & (subdf.price_per_sqft<=(m+std))]
        df_out = pd.concat([df_out,reduce_df],ignore_index = True)
    return df_out
    

row6 = remove_pps_outliers(row5)



def plot_scatter_chart(df,location):
    bhk2 = df[(df.location == location) & (df.bhk == 2)]
    bhk3 = df[(df.location == location) & (df.bhk == 3)]
    matplotlib.rcParams["figure.figsize"] =(15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color = "blue", label = "2 bhk", s = 50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker = "+", color = "red", label = "3 bhk", s = 50)
    plt.xlabel("Total  sqft area")
    plt.ylabel("price")
    plt.title(location)
    plt.plot()
    
    
    
    
plot_scatter_chart(row6,"Hebbal")


def remove_bhk_outlier(df):
    exclude_index = np.array([])
    for location,location_df in df.groupby("location"):
        bhk_stat = {}
        for bhk,bhk_df in location_df.groupby("bhk"):
            bhk_stat[bhk] = {
                    'mean' : np.mean(bhk_df.price_per_sqft),
                    'std' : np.std(bhk_df.price_per_sqft),
                    'count' : bhk_df.shape[0]
                    }
        for bhk,bhk_df in location_df.groupby("bhk"):
            stat = bhk_stat.get(bhk-1)
            if stat and stat["count"]>5:
                exclude_index = np.append(exclude_index,bhk_df[bhk_df.price_per_sqft<(stat["mean"])].index.values)
    return df.drop(exclude_index,axis = 'index')



row7 = remove_bhk_outlier(row6)


plot_scatter_chart(row7,"Hebbal")


print(row7.bath.unique())


print(row7[row7.bath>row7.bhk+2])


row8 = row7[~(row7.bath>=row7.bhk+2)]


row9 = row8.drop(["size","price_per_sqft"],axis = "columns")


dumies = pd.get_dummies(row9.location)
row10 = pd.concat([row9,dumies.drop(["other_location"],axis = "columns")],axis = "columns")
row11 = row10.drop('location',axis = 'columns')





"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le = LabelEncoder()
row9["location"] = le.fit_transform(row9["location"])
a = row9["location"]

ohe = OneHotEncoder(categorical_features = [0])
row9 = ohe.fit_transform(row9).toarray()
"""


x = row11.drop('price',axis = 'columns')
y= row11.price

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2,random_state = 10)



from sklearn.linear_model import LinearRegression
final_model = LinearRegression()
final_model.fit(x_train,y_train)
print(final_model.score(x_test,y_test))


def predict_price(location,sqft,bath,bhk):
    loc_index = np.where(x.columns == location)[0][0]
    a = np.zeros(len(x.columns))
    a[0] = sqft
    a[1] = bath
    a[2] = bhk
    if loc_index >= 0:
        a[loc_index] = 1
    return final_model.predict([a])[0]


print(predict_price("1st Phase JP Nagar",1000,2,2))
        
    










