from utils import feature_normalize
import sklearn
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# Map X onto polynomial features and normalize
# We will consider a 6th order polynomial fit for the data

p = 6
poly = sklearn.preprocessing.PolynomialFeatures(degree=p,include_bias=False)

X_poly = poly.fit_transform(np.reshape(X,(len(X),1)))
print (x_poly)

X_poly, mu, sigma = utils.feature_normalize(X_poly)

# add a column of ones to X_poly
XX_poly = np.vstack([np.ones((X_poly.shape[0],)),X_poly.T]).T

# map Xtest and Xval into the same polynomial features
X_poly_test = poly.fit_transform(np.reshape(Xtest,(len(Xtest),1)))
X_poly_val = poly.fit_transform(np.reshape(Xval,(len(Xval),1)))

# normalize these two sets with the same mu and sigma

X_poly_test = (X_poly_test - mu) / sigma
X_poly_val = (X_poly_val - mu) / sigma

# add a column of ones to both X_poly_test and X_poly_val
XX_poly_test = np.vstack([np.ones((X_poly_test.shape[0],)),X_poly_test.T]).T
XX_poly_val = np.vstack([np.ones((X_poly_val.shape[0],)),X_poly_val.T]).T