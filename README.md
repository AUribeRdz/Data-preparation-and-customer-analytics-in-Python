# Data-preparation-and-customer-analytics-in-Python
# Conduct analysis on your client's transaction dataset and identify customer purchasing behaviours to generate insights and provide commercial recommendations.
#
#
#
# PYTHON CODE:
#### TASK 1 EXPLORATORY DATA ANALYSIS

### Loading Phyton libraries
pip install mlxtend
import pandas as pd
import datetime
from datetime import date, timedelta
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed
from numpy.random import randn
from scipy.stats import f_oneway
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

### Loading first file as dataframe to examinee
### Loading transactions data 
df = pd.read_csv('QVI_transaction_data.csv') 

### Examining dataframe (format, type of data, etc.)
df.columns
df.head()
# At this point DATE column needs to be changed as date type instead of object
df.dtypes
# At this point PROD_QTY column which is INT type shows outliner in MAX row (200). Need to examinee.
df. describe()
# At this point confirmed no null values are present
df.info()  

### Converting DATE column to date format
df['DATE'] = pd.to_datetime(df['DATE'])
df['DATE']

### Examining PROD_QTY column on 200 value to clarify what kind of outliner is.
# two purchases in the yearly scope by one single purchaser (LYLTY_CARD_NBR 226000) so these rows qualify to be dropped. 
# as consequence of this examination realized PROD_NAME column may keeps abbreviations (Chp instead of Chips)
df[df['PROD_QTY'] == 200]

# Outliners on chart view
df.plot(x="PROD_NAME", y="PROD_QTY", kind="scatter", title="Quantity by Product Name (Identifing Outliners)")

### Examining PROD_NAME within Chips abbreviation (Chp).
# At this point 5% (12,765 rows) of total data hold abbreviation Chp
df['PROD_NAME'].str.contains('chp', case=False).value_counts()

### Examining PROD_NAME within “Chip”.
# At this point 28% (74,570 rows) of total data hold “Chip” in product description.
df['PROD_NAME'].str.contains('chip', case=False).value_counts()

### Checking on kind of products in PROD_NAME column as summary way
# At this point most of the data (67%) does not include word chip or abbreviation chp
df['PROD_NAME'].value_counts()

### Dropping outliners detected with 200 in PROD_QTY
df = df[df.PROD_QTY != 200]
# At this point from the original 264,836 rows now left 264,834 after drop outliners
df

### Summarizing the dataframe to check for nulls 
# At this no nulls are present
df. info()

### Summarizing (in numbers) the dataframe to check for possible outliers
# At this point no outliners are present
df. describe()

### Summarizing (in chart) the dataframe to check for possible outliers
# At this point no outliners are present
df.plot(x="PROD_NAME", y="PROD_QTY", kind="scatter", title="Quantity by Product Name (Identifing Outliners)")

### Counting the number of transactions by date
df.groupby(['DATE']).size()

### Since Data came from range 2018-07-01 to 2019-06-30 checking what dates are missing because from describe() above no PROD_QTY = 0 are present and there are 364 (not 365) unique date in counting above
# At this point only 2018-12-25 is sale date missing as consequence of XMAS
dat = df['DATE'].sort_values()  # Only DATE column is passed to another dataframe already sorted (ascending)
print(dat)
dat2 = dat.drop_duplicates() # No duplicates dates are passed to another dataframe
print(dat2)
dates = dat2.tolist() # Dataframe with no duplicates is passed to list to check dates missed
# Here starts routine to check dates missed
start_date = dates[0]
end_date = dates[len(dates)-1]
numdays = (end_date - start_date).days
all_dates = []
for x in range (0, (numdays+1)):
   all_dates.append(start_date + datetime.timedelta(days = x))
   dates_missing = []
for i in range (0, len(all_dates)):
   if (all_dates[i] not in dates):
       dates_missing.append(all_dates[i])
   else:
       pass
dates_missing  # end routine to check dates missed

### Creating pack size feature & Checking frequency
# At this point top 8 pack sizes do 83% of total purchased transactions (175, 150, 134, 110, 170, 165, 300 & 330g)
df['PACK_SIZE'] = df.PROD_NAME.str.extract('(\d+)')
df.groupby(['PACK_SIZE']).size().sort_values(ascending = False)

### Plotting a histogram showing the number of transactions by pack size
df.groupby('PACK_SIZE').size().sort_values(ascending = False).plot(kind="bar", title="Transactions Count by Pack Size")

### Creating column which contains product brand  (using the first word in PROD_NAME) & Checking frequency
df['BRAND'] = df.PROD_NAME.str.split().str.get(0)
df.groupby(['BRAND']).size().sort_values(ascending = False)

### Checking on brand names column looks like they are of the same brands ('Red' & 'RRD') & Checking frequency
# Replacing “Red” with “RRD” in column BRAND
df['BRAND'].replace('Red', 'RRD', inplace=True)
df.groupby(['BRAND']).size().sort_values(ascending = False)

### Loading customer data 
df2 = pd.read_csv('QVI_purchase_behaviour.csv') 

### Examining dataframe (format, type of data, etc.)
df2.columns
df2.head()
df2.dtypes  # At this point there isn’t any column to change data type
df2.describe()  # At this point no INT related to outliners.
df2.info()  # At this point confirmed no null values are present.

### Summarizing on customer data to analyze
# At this point top 3 Lifestages Customers are Retirees, Older Singles-Couples and Young Singles-Couples with 20% each
df2.groupby(['LIFESTAGE']).size().sort_values(ascending = False)

# At this point Premium Customers are Mainstream (40%), Budget (34%) and Premium (26%)
df2.groupby(['PREMIUM_CUSTOMER']).size().sort_values(ascending = False)

### Merging transaction data to customer data
df = pd.merge(df , df2, how='left', on='LYLTY_CARD_NBR')

### Checking for nulls 
# At this point confirmed (all our customers in the transaction data has been accounted for in the customer dataset).
df.info()  

### Writing (exporting) df as csv
df.to_csv('transactions_customers_data.csv')


#### TASK 1 DATA ANALYSIS ON CUSTOMERS SEGMENTS

### Total sales by LIFESTAGE and PREMIUM_CUSTOMER – Summary
# At this point Top 3 Sales comes from Older Families – Budget (9%), Young Singles Couples – Mainstream (8%) and Retirees – Mainstream type Customers (8%) total here is 25%.
df.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER'])['TOT_SALES'].sum().sort_values(ascending = False)

### Total sales by LIFESTAGE and PREMIUM_CUSTOMER - Plot
df.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER'])['TOT_SALES'].sum().sort_values(ascending = False).plot(kind="bar",
title="Sales by Lifestage,Premium Customers", rot=90)

### Number of customers by LIFESTAGE and PREMIUM_CUSTOMER - Summary
# At this point combination on type of Customers are top 3: Older Families – Budget (9%), Retirees – Mainstream (8%) and Young singles couples – Mainstream (8%) total is 25%.
df.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER'])['LYLTY_CARD_NBR'].count().sort_values(ascending = False)

### Number of customers by LIFESTAGE and PREMIUM_CUSTOMER - Plot
df.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER'])['LYLTY_CARD_NBR'].count().sort_values(ascending = False). plot(kind="bar", title="Customers by Lifestage,Premium Customers", rot=90)

### Average number of units per customer by LIFESTAGE and PREMIUM_CUSTOMER - Summary
# At this point average quantity per Customer type is about 2 each per purchase.
df.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER'])['PROD_QTY'].mean().sort_values(ascending = False)

### Average number of units per customer by LIFESTAGE and PREMIUM_CUSTOMER - Plot
df.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER'])['PROD_QTY'].mean().sort_values(ascending = False).plot(kind="bar", title="Avg Qty by Lifestage,Premium Customers", rot=90)

#### Average price per unit by LIFESTAGE and PREMIUM_CUSTOMER - Summary
# At this point average price per Customer type is about USD$4 per purchase. The difference in average price per unit per Lifestage – Premium_Customer type isn't large among Customers type groups. However, Top groups willing to pay more (avg unit price per purchase) are: YOUNG SINGLES/COUPLES-Mainstream and MIDAGE SINGLES/COUPLES-Mainstream.
df['UNIT_PRICE'] = df['TOT_SALES']/df['PROD_QTY']
df.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER'])['UNIT_PRICE'].mean().sort_values(ascending = False)

#### Average price per unit by LIFESTAGE and PREMIUM_CUSTOMER - Plot
df.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER'])['UNIT_PRICE'].mean().sort_values(ascending = False).plot(kind="bar", title="Avg Price by Lifestage,Premium Customers", rot=90)

### Performing independent t-test among Top 2 groups previously revised for average price to check if this difference is statistically different among Lifestage – Premium on next 6 Customer types: 
### YOUNG SINGLES/COUPLES-Mainstream VS YOUNG SINGLES/COUPLES-Premium VS YOUNG SINGLES/COUPLES-Budget 
### VS MIDAGE SINGLES/COUPLES-Premium VS MIDAGE SINGLES/COUPLES- Premium VS MIDAGE SINGLES/COUPLES-Budget. 
# Analysis of Variance test. Running next code calculates and prints the test statistic and the p-value.
# seeding the random number generator
seed(1)
# generating independent samples
data11 = df[['UNIT_PRICE']][(df['LIFESTAGE']=='YOUNG SINGLES/COUPLES') & (df['PREMIUM_CUSTOMER']=='Mainstream')]
data1 = list(data11.UNIT_PRICE)
data22 = df[['UNIT_PRICE']][(df['LIFESTAGE']=='YOUNG SINGLES/COUPLES') & (df['PREMIUM_CUSTOMER']=='Premium')]
data2 = list(data22.UNIT_PRICE)
data33 = df[['UNIT_PRICE']][(df['LIFESTAGE']=='YOUNG SINGLES/COUPLES') & (df['PREMIUM_CUSTOMER']=='Budget')]
data3 = list(data33.UNIT_PRICE)
data44 = df[['UNIT_PRICE']][(df['LIFESTAGE']=='MIDAGE SINGLES/COUPLES') & (df['PREMIUM_CUSTOMER']=='Mainstream')]
data4 = list(data44.UNIT_PRICE)
data55 = df[['UNIT_PRICE']][(df['LIFESTAGE']=='MIDAGE SINGLES/COUPLES') & (df['PREMIUM_CUSTOMER']=='Premium')]
data5 = list(data55.UNIT_PRICE)
data66 = df[['UNIT_PRICE']][(df['LIFESTAGE']=='MIDAGE SINGLES/COUPLES') & (df['PREMIUM_CUSTOMER']=='Budget')]
data6 = list(data66.UNIT_PRICE)
# comparing samples
stat, p = f_oneway(data1, data2, data3, data4, data5, data6)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Same distributions (fail to reject H0)')
else:
	print('Different distributions (reject H0)')
# The interpretation of the p-value correctly rejects the null hypothesis (All sample distributions are equal) indicating that one or more sample means differ. This means that there is a significant difference among Lifestage – Premium Customer types studied. The unit price for mainstream, young singles and couples and mid-age singles and couples ARE significantly higher than that of budget or premium, young and midage singles and couples.

### Deep diving into Mainstream, young singles/couples customer which is one of the segments that contribute the most to sales to retain them or further increase sales (i.e. finding out if they tend to buy a particular brand of chips) applying Apriori function to extract frequent itemsets for association rule mining.
## Preparing data with target segment
basket = (df[(df.PREMIUM_CUSTOMER == 'Mainstream') & (df.LIFESTAGE == 'YOUNG SINGLES/COUPLES')].groupby(['TXN_ID','BRAND'])['PROD_QTY'].sum().unstack().reset_index().fillna(0).set_index('TXN_ID'))
basket.head()
##  One hot encoding of the data
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
basket_sets = basket.applymap(encode_units)
basket_sets.head()
## Generating frequent item sets that have a support of at least 1% (this number was chosen because is the minimum)
### Conclusion: Mainstream, young singles/couples customers only buy one BRAND at time and top 3 are: KETTLE (18%), DORITOS (12%) and PRINGLES (11%).
frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values('support', ascending = False)

### Deep diving into Mainstream, young singles/couples customer which is one of the segments that contribute the most to sales to retain them or further increase sales (i.e. finding out if they tend to buy a particular pack size of chips) applying Apriori function to extract frequent itemsets for association rule mining.
## Preparing data with target segment
basket2 = (df[(df.PREMIUM_CUSTOMER == 'Mainstream') & (df.LIFESTAGE == 'YOUNG SINGLES/COUPLES')].groupby(['TXN_ID','PACK_SIZE'])['PROD_QTY'].sum().unstack().reset_index().fillna(0).set_index('TXN_ID'))
basket2.head()
##  One hot encoding of the data
basket_sets2 = basket2.applymap(encode_units)
basket_sets2. head()
##  Generate frequent item sets that have a support of at least 1% (this number was chosen because is the minimum)
### Conclusion: Mainstream, young singles/couples customers only buy one PACK_SIZE at time and top 3 are: 175g (24%), 150g (16%) and 134g (11%).
frequent_itemsets2 = apriori(basket_sets2, min_support=0.01, use_colnames=True)
frequent_itemsets2.sort_values('support', ascending = False)
