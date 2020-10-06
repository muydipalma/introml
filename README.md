# Por que necesitamos Algebra Lineal en Machine Learning?

Bienvenido al maravilloso mundo del machine learning! Antes de ponernos comodos veamos algunos ejemplos incomodos donde se nos manifiesta la necesidad de usar elementos del algebra lineal como: vectores, matrices, tranformaciones lineales, etc:

* Datasets
* Imagenes y video
* Variables categoricas
* Regresion Lineal 
* Regularizacion
* PCA y dimensionalidad
* Latent Semantic Analysis
* Sistemas de recomendacion
* Deep Learning

### Datasets 

La idea central de machine learning, es ajustar un modelo a un set de datos (dataset) de manera de poder predecir alguna variable (target) en funcion de otras variables explicativas (features). Para esto es necesario ordenar los datos en una tabla donde cada fila represente una observacion y que cada columna sea una feature que represente nuestra obvservacion.


Por ejemplo, imaginemos que queremos entrenar un modelo que nos permite predecir las ventas de un producto en funcion de la inversion que se hace en publicidad, nuestras columnas de caracteristicas seran los 3 tipos de medio de comunicacion. Television, radio y periodicos seran las features y Sales el target. Cada observacion sera un mes de los gastos. El dataset original tiene el registro de los ultimos 200 meses, aca vemos solo los ultimos 4:

```
TV,Radio,Newspaper,Sales
230.1,37.8,69.2,22.1
44.5,39.3,45.1,10.4
17.2,45.9,69.3,9.3
151.5,41.3,58.5,18.5
```

Esto en realidad es una **matriz**, una estructura basica del algebra lineal. Es mas, cuando uno parte el data set en features y target, lo que hace es tener una _matriz_ (X) y un  a **vector** target (y). Un vector es otro elemento fundamental del algebra lineal.


Cada final tiene la misma longitud, esto es el mismo numero de columnas, por lo tanto decimos la que data fue vectoriazada y cada observacion nueva sera un vector o varios que serviran para predecir su variable target.


### Imagenes y video

Quizas uno ya trabaje con imagenes o este intersado en computer vision (machine learning para imagenes), para esto cada imagen puede ser pensada como una matriz (ancho x alto) donde cada lugar representa un pixel, y el valor que sera asignado tiene que ver con el color. Si la imagen fuera en blanco y negro, el valor sera en escala de grises. Pero si fuera una imagen en colores, en realidad tendriamos 3 matrices, una para cada color (RGB).


Asi, una imagen no es mas que otro ejemplo de el uso de una _matriz_. Entonces cualquier operacion que querramos hacer con imagenes (cropping, scaling, shearing), necesariamente nos obliga a conocer la notacion y las operaciones con matrices. 

### Variables categoricas

Sometimes you work with categorical data in machine learning.

Perhaps the class labels for classification problems, or perhaps categorical input variables.

It is common to encode categorical variables to make them easier to work with and learn by some techniques. A popular encoding for categorical variables is the one hot encoding.

A one hot encoding is where a table is created to represent the variable with one column for each category and a row for each example in the dataset. A check, or one-value, is added in the column for the categorical value for a given row, and a zero-value is added to all other columns.

For example, the color variable with the 3 rows:

```
red
green
blue
...
``` 

Might be encoded as:
```
red, green, blue
1, 0, 0
0, 1, 0
0, 0, 1
...
```

Each row is encoded as a binary **vector**, a vector with zero or one values and this is an example of a **sparse representation**, a whole sub-field of linear algebra.

### Regresion Lineal

Linear regression is an old method from statistics for describing the relationships between variables.

It is often used in machine learning for predicting numerical values in simpler regression problems.

There are many ways to describe and solve the linear regression problem, i.e. finding a set of coefficients that when multiplied by each of the input variables and added together results in the best prediction of the output variable.

If you have used a machine learning tool or library, the most common way of solving linear regression is via a least squares optimization that is solved using matrix factorization methods from linear regression, such as an LU decomposition or a singular-value decomposition, or SVD.

Even the common way of summarizing the linear regression equation uses linear algebra notation:

```
y = A . b
```

Where y is the output variable A is the dataset and b are the model coefficients.

### Regularizacion

In applied machine learning, we often seek the simplest possible models that achieve the best skill on our problem.

Simpler models are often better at generalizing from specific examples to unseen data.

In many methods that involve coefficients, such as regression methods and artificial neural networks, simpler models are often characterized by models that have smaller coefficient values.

A technique that is often used to encourage a model to minimize the size of coefficients while it is being fit on data is called regularization. Common implementations include the L2 and L1 forms of regularization.

Both of these forms of regularization are in fact a measure of the magnitude or length of the coefficients as a vector and are methods lifted directly from linear algebra called the **vector norm**.

### PCA y dimensionalidad

Often, a dataset has many columns, perhaps tens, hundreds, thousands, or more.

Modeling data with many features is challenging, and models built from data that include irrelevant features are often less skillful than models trained from the most relevant data.

It is hard to know which features of the data are relevant and which are not.

Methods for automatically reducing the number of columns of a dataset are called dimensionality reduction, and perhaps the most popular method is called the principal component analysis, or PCA for short.

This method is used in machine learning to create projections of high-dimensional data for both visualization and for training models.

The core of the PCA method is a **matrix factorization** method from linear algebra. The eigendecomposition can be used and more robust implementations may use the singular-value decomposition, or SVD.

### Latent Semantic Analysis

In the sub-field of machine learning for working with text data called natural language processing, it is common to represent documents as large matrices of word occurrences.

For example, the columns of the matrix may be the known words in the vocabulary and rows may be sentences, paragraphs, pages, or documents of text with cells in the matrix marked as the count or frequency of the number of times the word occurred.

This is a sparse matrix representation of the text. **Matrix factorization** methods, such as the singular-value decomposition can be applied to this sparse matrix, which has the effect of distilling the representation down to its most relevant essence. Documents processed in this way are much easier to compare, query, and use as the basis for a supervised machine learning model.

This form of data preparation is called Latent Semantic Analysis, or LSA for short, and is also known by the name Latent Semantic Indexing, or LSI.

### Sistemas de recomendacion

Predictive modeling problems that involve the recommendation of products are called recommender systems, a sub-field of machine learning.

Examples include the recommendation of books based on previous purchases and purchases by customers like you on Amazon, and the recommendation of movies and TV shows to watch based on your viewing history and viewing history of subscribers like you on Netflix.

The development of recommender systems is primarily concerned with linear algebra methods. A simple example is in the calculation of the similarity between sparse customer behavior vectors using distance measures such as Euclidean distance or **dot products**.

Matrix factorization methods like the singular-value decomposition are used widely in recommender systems to distill item and user data to their essence for querying and searching and comparison.

### Deep Learning

Artificial neural networks are nonlinear machine learning algorithms that are inspired by elements of the information processing in the brain and have proven effective at a range of problems, not the least of which is predictive modeling.

Deep learning is the recent resurgence in the use of artificial neural networks with newer methods and faster hardware that allow for the development and training of larger and deeper (more layers) networks on very large datasets. Deep learning methods are routinely achieving state-of-the-art results on a range of challenging problems such as machine translation, photo captioning, speech recognition, and much more.

At their core, the execution of neural networks involves linear algebra data structures multiplied and added together. Scaled up to multiple dimensions, deep learning methods work with vectors, matrices, and even tensors of inputs and coefficients, where a **tensor** is a matrix with more than two dimensions.

Linear algebra is central to the description of deep learning methods via matrix notation to the implementation of deep learning methods such as Google’s TensorFlow Python library that has the word “tensor” in its name.



## Resumiendo:

Imagino que despues de todos estos ejemplos, estan convencidos de lo necesario que es saber algebra lineal. El lenguaje basico detras de todas las ideas de machine learning son los vectores matricas y transformaciones lineales. Sin esto, es muy dificil y menos provechosa cualquier texto o clase de ML. 





# Fundamentos de Algebra Lineal

A continuacion vamos a repasar y jugar con algunos conceptos que se muestran en la maravillosa [serie](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
de videos del canal [3Blue1Brown](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw)



# Vectores

<iframe width="800" height='480' src="https://www.youtube.com/embed/fNk_zzaMoSs" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

