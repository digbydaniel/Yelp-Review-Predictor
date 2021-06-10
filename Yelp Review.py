import pandas as pd
business = pd.read_json('yelp_business.json',lines=True)
review = pd.read_json('yelp_review.json', lines=True)
user = pd.read_json('yelp_user.json', lines=True)
checkin = pd.read_json('yelp_checkin.json', lines=True)
tip = pd.read_json('yelp_tip.json', lines=True)
photo = pd.read_json('yelp_photo.json', lines=True)

# pd.options.display.max_columns = 60
pd.options.display.max_colwidth = 500
business.head()
review.head()
user.head()
checkin.head()
tip.head()
photo.head()

len(business)
review.columns

user.describe()

business[business['business_id']  == '5EvUIR4IzCWUOm0PsUZXjA']['stars']
df = pd.merge(business, review, how='left', on='business_id')
df = pd.merge(df, user, how = 'left', on = 'business_id')
df = pd.merge(df, checkin, how = 'left', on = 'business_id')
df = pd.merge(df, tip, how = 'left', on = 'business_id')
df = pd.merge(df, photo, how = 'left', on = 'business_id')
print(df.columns)

features_to_remove = ['address','attributes','business_id','categories','city','hours','is_open','latitude','longitude','name','neighborhood','postal_code','state','time']
df.drop(features_to_remove, axis=1, inplace=True)
#determine if there is any NaN in our data set
df.isna().any()
#fill in any NaN
df.fillna({'weekday_checkins':0, 'weekend_checkins':0, 'average_tip_length':0, 'number_tips':0, 'average_caption_length':0, 'number_tips':0, 'number_pics':0}, inplace=True)
df.isna().any()

df.corr()

from matplotlib import pyplot as plt

# plot average_review_sentiment against stars here
plt.scatter(df['average_review_sentiment'], df['stars'], alpha=0.025)
plt.show

# plot average_review_length against stars here
plt.scatter(df['average_review_length'], df['stars'], alpha=0.025)
plt.show

# plot average_review_age against stars here
plt.scatter(df['average_review_age'], df['stars'], alpha=0.025)
plt.show

# plot number_funny_votes against stars here
plt.scatter(df['number_funny_votes'], df['stars'], alpha=0.025)
plt.show

features = df[['average_review_length', 'average_review_age']]
ratings = df['stars']

#split data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, ratings, test_size = 0.2, random_state = 1)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

model.score(X_train, y_train)
model.score(X_test, y_test)

sorted(list(zip(['average_review_length','average_review_age'],model.coef_)),key = lambda x: abs(x[1]),reverse=True)
y_predicted = model.predict(X_test)
plt.scatter(y_test, y_predicted, alpha = 0.02)
plt.show

# subset of only average review sentiment
sentiment = ['average_review_sentiment']

# subset of all features that have a response range [0,1]
binary_features = ['alcohol?','has_bike_parking','takes_credit_cards','good_for_kids','take_reservations','has_wifi']

# subset of all features that vary on a greater range than [0,1]
numeric_features = ['review_count','price_range','average_caption_length','number_pics','average_review_age','average_review_length','average_review_sentiment','number_funny_votes','number_cool_votes','number_useful_votes','average_tip_length','number_tips','average_number_friends','average_days_on_yelp','average_number_fans','average_review_count','average_number_years_elite','weekday_checkins','weekend_checkins']

# all features
all_features = binary_features + numeric_features

import numpy as np

# take a list of features to model as a parameter
def model_these_features(feature_list):
    
    #seperating data frame into x and y
    ratings = df.loc[:,'stars']
    features = df.loc[:,feature_list]
    
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(features, ratings, test_size = 0.2, random_state = 1)
    
    # don't worry too much about these lines, just know that they allow the model to work when
    # we model on just one feature instead of multiple features. Trust us on this one :)
    if len(X_train.shape) < 2:
        X_train = np.array(X_train).reshape(-1,1)
        X_test = np.array(X_test).reshape(-1,1)
    
    # fit the model to a linear regression
    model = LinearRegression()
    model.fit(X_train,y_train)
    
    # print scores
    print('Train Score:', model.score(X_train,y_train))
    print('Test Score:', model.score(X_test,y_test))
    
    # print the model features and their corresponding coefficients, from most predictive to least predictive
    print(sorted(list(zip(feature_list,model.coef_)),key = lambda x: abs(x[1]),reverse=True))
    
    # predict y
    y_predicted = model.predict(X_test)
    
    # plot the scatter
    plt.scatter(y_test,y_predicted)
    plt.xlabel('Yelp Rating')
    plt.ylabel('Predicted Yelp Rating')
    plt.ylim(1,5)
    plt.show()

model_these_features(all_features)