# coding: utf-8

# <h1> Getting started with TensorFlow </h1>
# 
# In this notebook, you play around with the TensorFlow Python API.

# In[1]:


import tensorflow as tf
import numpy as np

print(tf.__version__)

# <h2> Adding two tensors </h2>
# 
# First, let's try doing this using numpy, the Python numeric package. numpy code is immediately evaluated.

# In[2]:


a = np.array([5, 3, 8])
b = np.array([3, -1, 2])
c = np.add(a, b)
print(c)

# The equivalent code in TensorFlow consists of two steps:
# <p>
# <h3> Step 1: Build the graph </h3>

# In[ ]:


a = tf.constant([5, 3, 8])
b = tf.constant([3, -1, 2])
c = tf.add(a, b)
print(c)

# c is an Op ("Add") that returns a tensor of shape (3,) and holds int32. The shape is inferred from the computation graph.
# 
# Try the following in the cell above:
# <ol>
# <li> Change the 5 to 5.0, and similarly the other five numbers. What happens when you run this cell? </li>
# <li> Add an extra number to a, but leave b at the original (3,) shape. What happens when you run this cell? </li>
# <li> Change the code back to a version that works </li>
# </ol>
# 
# <p/>
# <h3> Step 2: Run the graph

# In[ ]:


with tf.Session() as sess:
    result = sess.run(c)
    print(result)

# <h2> Using a feed_dict </h2>
# 
# Same graph, but without hardcoding inputs at build stage

# In[ ]:


a = tf.placeholder(dtype=tf.int32, shape=(None,))  # batchsize x scalar
b = tf.placeholder(dtype=tf.int32, shape=(None,))
c = tf.add(a, b)
with tf.Session() as sess:
    result = sess.run(c, feed_dict={
        a: [3, 4, 5],
        b: [-1, 2, 3]
    })
    print(result)


# <h2> Heron's Formula in TensorFlow </h2>
# 
# The area of triangle whose three sides are $(a, b, c)$ is $\sqrt{s(s-a)(s-b)(s-c)}$ where $s=\frac{a+b+c}{2}$ 
# 
# Look up the available operations at https://www.tensorflow.org/api_docs/python/tf

# In[ ]:


def compute_area(sides):
    # slice the input to get the sides
    a = sides[:, 0]  # 5.0, 2.3
    b = sides[:, 1]  # 3.0, 4.1
    c = sides[:, 2]  # 7.1, 4.8

    # Heron's formula
    s = (a + b + c) * 0.5  # (a + b) is a short-cut to tf.add(a, b)
    areasq = s * (s - a) * (s - b) * (s - c)  # (a * b) is a short-cut to tf.multiply(a, b), not tf.matmul(a, b)
    return tf.sqrt(areasq)


with tf.Session() as sess:
    # pass in two triangles
    area = compute_area(tf.constant([
        [5.0, 3.0, 7.1],
        [2.3, 4.1, 4.8]
    ]))
    result = sess.run(area)
    print(result)

# <h2> Placeholder and feed_dict </h2>
# 
# More common is to define the input to a program as a placeholder and then to feed in the inputs. The difference between the code below and the code above is whether the "area" graph is coded up with the input values or whether the "area" graph is coded up with a placeholder through which inputs will be passed in at run-time.

# In[ ]:


with tf.Session() as sess:
    sides = tf.placeholder(tf.float32, shape=(None, 3))  # batchsize number of triangles, 3 sides
    area = compute_area(sides)
    result = sess.run(area, feed_dict={
        sides: [
            [5.0, 3.0, 7.1],
            [2.3, 4.1, 4.8]
        ]
    })
    print(result)

# ## tf.eager
# 
# tf.eager allows you to avoid the build-then-run stages. However, most production code will follow the lazy evaluation paradigm because the lazy evaluation paradigm is what allows for multi-device support and distribution. 
# <p>
# One thing you could do is to develop using tf.eager and then comment out the eager execution and add in the session management code.
# 
# <b>You may need to click on Reset Session to try this out.</b>

# In[ ]:


import tensorflow as tf

tf.enable_eager_execution()


def compute_area(sides):
    # slice the input to get the sides
    a = sides[:, 0]  # 5.0, 2.3
    b = sides[:, 1]  # 3.0, 4.1
    c = sides[:, 2]  # 7.1, 4.8

    # Heron's formula
    s = (a + b + c) * 0.5  # (a + b) is a short-cut to tf.add(a, b)
    areasq = s * (s - a) * (s - b) * (s - c)  # (a * b) is a short-cut to tf.multiply(a, b), not tf.matmul(a, b)
    return tf.sqrt(areasq)


area = compute_area(tf.constant([
    [5.0, 3.0, 7.1],
    [2.3, 4.1, 4.8]
]))

print(area)


# ## Challenge Exercise
# 
# Use TensorFlow to find the roots of a fourth-degree polynomial using [Halley's Method](https://en.wikipedia.org/wiki/Halley%27s_method).  The five coefficients (i.e. $a_0$ to $a_4$) of 
# <p>
# $f(x) = a_0 + a_1 x + a_2 x^2 + a_3 x^3 + a_4 x^4$
# <p>
# will be fed into the program, as will the initial guess $x_0$. Your program will start from that initial guess and then iterate one step using the formula:
# <img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/142614c0378a1d61cb623c1352bf85b6b7bc4397" />
# <p>
# If you got the above easily, try iterating indefinitely until the change between $x_n$ and $x_{n+1}$ is less than some specified tolerance. Hint: Use [tf.while_loop](https://www.tensorflow.org/api_docs/python/tf/while_loop)

# Copyright 2017 Google Inc. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License

def f_x(a0, a1, a2, a3, a4, x):
    return a0 + (a1 * tf.pow(x, 1)) + (a2 * tf.pow(x, 2)) + (a3 * tf.pow(x, 3)) + (a4 * tf.pow(x, 4))


def f1_x(a1, a2, a3, a4, x):
    return a1 + (2 * a2 * tf.pow(x, 1)) + (3 * a3 * tf.pow(x, 2)) + (4 * a4 * tf.pow(x, 3))


def f2_x(a2, a3, a4, x):
    return (2 * a2) + (3 * 2 * a3 * tf.pow(x, 1)) + (4 * 3 * a4 * tf.pow(x, 2))


def step(a0, a1, a2, a3, a4, x):
    f = f_x(a0, a1, a2, a3, a4, x)
    f1 = f1_x(a1, a2, a3, a4, x)
    f2 = f2_x(a2, a3, a4, x)

    _n = 2*f*f1
    _d = 2*tf.pow(f1, 2) - (f*f2)
    x1 = x - (_n / _d)
    return x1


def condition(a0, a1, a2, a3, a4, x):
    x_new = step(a0, a1, a2, a3, a4, x)
    return tf.abs(x_new - x) > tf.constant(0.00001)


def get_root(a, x):
    a0 = a[0]
    a1 = a[1]
    a2 = a[2]
    a3 = a[3]
    a4 = a[4]
    # i = 0

    res = tf.while_loop(condition, step, [a0, a1, a2, a3, a4, x])

    # while True:
    #     print("{0}:{1}".format(i, x))
    #
    #     x_ = step(a0, a1, a2, a3, a4, x)
    #     if tf.abs(x_ - x) < tf.constant(0.00001):
    #         break
    #
    #     x = x_
    #     i += 1
    # return x_

    return res


with tf.Session() as sess:
    # pass in two triangles
    root = get_root(tf.constant([1, 1, 1, 1, 0], dtype=tf.float32), tf.constant(2.0))
    result = sess.run(root)
    print(result)

# root = get_root(tf.constant([1, 1, 1, 1, 0], dtype=tf.float32), tf.constant(2.0))
# print(root)

