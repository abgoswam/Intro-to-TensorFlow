# coding: utf-8

# <h1>2b. Machine Learning using tf.estimator </h1>
# 
# In this notebook, we will create a machine learning model using tf.estimator and evaluate its performance.  The dataset is rather small (7700 samples), so we can do it all in-memory.  We will also simply pass the raw data in as-is. 

# In[ ]:


# import datalab.bigquery as bq
import tensorflow as tf
import pandas as pd
import numpy as np
import shutil

print(tf.__version__)

# Read data created in the previous chapter.

# In[ ]:


# In CSV, label is the first column, after the features, followed by the key
CSV_COLUMNS = ['fare_amount', 'pickuplon', 'pickuplat', 'dropofflon', 'dropofflat', 'passengers', 'key']
FEATURES = CSV_COLUMNS[1:len(CSV_COLUMNS) - 1]
LABEL = CSV_COLUMNS[0]

df_train = pd.read_csv('./week2/taxi-train.csv', header=None, names=CSV_COLUMNS)
df_valid = pd.read_csv('./week2/taxi-valid.csv', header=None, names=CSV_COLUMNS)


# <h2> Input functions to read from Pandas Dataframe </h2>

# In[ ]:


def make_input_fn(df, num_epochs):
    return tf.estimator.inputs.pandas_input_fn(
        x=df,
        y=df[LABEL],
        batch_size=128,
        num_epochs=num_epochs,
        shuffle=True,
        queue_capacity=1000,
        num_threads=1
    )


# Our input function for predictions is the same except we don't provide a label

# In[ ]:


def make_prediction_input_fn(df, num_epochs):
    return tf.estimator.inputs.pandas_input_fn(
        x=df,
        y=None,
        batch_size=128,
        num_epochs=num_epochs,
        shuffle=True,
        queue_capacity=1000,
        num_threads=1
    )


# ### Create feature columns for estimator

# In[ ]:


def make_feature_cols():
    input_columns = [tf.feature_column.numeric_column(k) for k in FEATURES]
    return input_columns


# <h3> Linear Regression with tf.Estimator framework </h3>

# In[ ]:


tf.logging.set_verbosity(tf.logging.INFO)

OUTDIR = 'taxi_trained'
shutil.rmtree(OUTDIR, ignore_errors=True)  # start fresh each time

model = tf.estimator.LinearRegressor(
    feature_columns=make_feature_cols(), model_dir=OUTDIR)

model.train(input_fn=make_input_fn(df_train, num_epochs=10))


# Evaluate on the validation data (we should defer using the test data to after we have selected a final model).

# In[ ]:


def print_rmse(model, name, df):
    metrics = model.evaluate(input_fn=make_input_fn(df, 1))
    print('RMSE on {} dataset = {}'.format(name, np.sqrt(metrics['average_loss'])))


print_rmse(model, 'validation', df_valid)

# This is nowhere near our benchmark (RMSE of $6 or so on this data), but it serves to demonstrate what TensorFlow code looks like.  Let's use this model for prediction.

# In[ ]:


predictions = model.predict(input_fn=make_prediction_input_fn(df_valid, 1))
for i in range(5):
    print(next(predictions))

# This explains why the RMSE was so high -- the model essentially predicts the same amount for every trip.  Would a more complex model help? Let's try using a deep neural network.  The code to do this is quite straightforward as well.

# <h3> Deep Neural Network regression </h3>

# In[ ]:


tf.logging.set_verbosity(tf.logging.INFO)
shutil.rmtree(OUTDIR, ignore_errors=True)  # start fresh each time
model = tf.estimator.DNNRegressor(hidden_units=[32, 8, 2],
                                  feature_columns=make_feature_cols(), model_dir=OUTDIR)
model.train(input_fn=make_input_fn(df_train, num_epochs=100));
print_rmse(model, 'validation', df_valid)

# We are not beating our benchmark with either model ... what's up?  Well, we may be using TensorFlow for Machine Learning, but we are not yet using it well.  That's what the rest of this course is about!
# 
# But, for the record, let's say we had to choose between the two models. We'd choose the one with the lower validation error. Finally, we'd measure the RMSE on the test data with this chosen model.

# <h2> Benchmark dataset </h2>
# 
# Let's do this on the benchmark dataset.

# In[ ]:


import datalab.bigquery as bq
import numpy as np
import pandas as pd


def create_query(phase, EVERY_N):
    """
  phase: 1 = train 2 = valid
  """
    base_query = """
SELECT
  (tolls_amount + fare_amount) AS fare_amount,
  CONCAT(STRING(pickup_datetime), STRING(pickup_longitude), STRING(pickup_latitude), STRING(dropoff_latitude), STRING(dropoff_longitude)) AS key,
  DAYOFWEEK(pickup_datetime)*1.0 AS dayofweek,
  HOUR(pickup_datetime)*1.0 AS hourofday,
  pickup_longitude AS pickuplon,
  pickup_latitude AS pickuplat,
  dropoff_longitude AS dropofflon,
  dropoff_latitude AS dropofflat,
  passenger_count*1.0 AS passengers,
FROM
  [nyc-tlc:yellow.trips]
WHERE
  trip_distance > 0
  AND fare_amount >= 2.5
  AND pickup_longitude > -78
  AND pickup_longitude < -70
  AND dropoff_longitude > -78
  AND dropoff_longitude < -70
  AND pickup_latitude > 37
  AND pickup_latitude < 45
  AND dropoff_latitude > 37
  AND dropoff_latitude < 45
  AND passenger_count > 0
  """

    if EVERY_N == None:
        if phase < 2:
            # Training
            query = "{0} AND ABS(HASH(pickup_datetime)) % 4 < 2".format(base_query)
        else:
            # Validation
            query = "{0} AND ABS(HASH(pickup_datetime)) % 4 == {1}".format(base_query, phase)
    else:
        query = "{0} AND ABS(HASH(pickup_datetime)) % {1} == {2}".format(base_query, EVERY_N, phase)

    return query


query = create_query(2, 100000)
df = bq.Query(query).to_dataframe()

# In[ ]:


print_rmse(model, 'benchmark', df)


# RMSE on benchmark dataset is <b>9.41</b> (your results will vary because of random seeds).
# 
# This is not only way more than our original benchmark of 6.00, but it doesn't even beat our distance-based rule's RMSE of 8.02.
# 
# Fear not -- you have learned how to write a TensorFlow model, but not to do all the things that you will have to do to your ML model performant. We will do this in the next chapters. In this chapter though, we will get our TensorFlow model ready for these improvements.
# 
# In a software sense, the rest of the labs in this chapter will be about refactoring the code so that we can improve it.

# ## Challenge Exercise
# 
# Create a neural network that is capable of finding the volume of a cylinder given the radius of its base (r) and its height (h). Assume that the radius and height of the cylinder are both in the range 0.5 to 2.0. Simulate the necessary training dataset.
# <p>
# Hint (highlight to see):
# <p style='color:white'>
# The input features will be r and h and the label will be $\pi r^2 h$
# Create random values for r and h and compute V.
# Your dataset will consist of r, h and V.
# Then, use a DNN regressor.
# Make sure to generate enough data.
# </p>

# Copyright 2017 Google Inc. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License
