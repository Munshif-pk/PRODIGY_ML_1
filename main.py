import pandas as pnd
from sklearn.linear_model import LinearRegression

LIR = LinearRegression()

try:
    tds = pnd.read_csv("train.csv", usecols=["LotArea", "SalePrice","FullBath","BedroomAbvGr"])
except FileNotFoundError:
    print("File not found.")
    exit()

x = tds[["LotArea","FullBath","BedroomAbvGr"]]
y = tds["SalePrice"]
LIR.fit(x,y)

ts = pnd.read_csv("test.csv", usecols=["LotArea","FullBath","BedroomAbvGr"])
tl = LIR.predict(ts)
ts["SalePrice"] = tl
print(ts)
ts.to_csv('test_result.csv',index=False)