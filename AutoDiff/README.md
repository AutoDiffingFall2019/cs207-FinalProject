# How to Use Package

## Installation

To begin, the user has to work in a `python` environment (preferably version >= 3.7). It is advisable for the user to create a new virtual environment for interacting with our package. To create a new virtual environment, enter the following command in the terminal:

`conda create -n env_autodiff python=3.7`

After which, activate the environment with the following command:

`conda activate env_autodiff`

Since we have used PyPI to host our package, users can download our Automatic Differentiation package with the following command in the terminal:

`pip install autodiffing`

As we have set up the pip package in a way such that the required dependencies will be installed during `pip install autodiffing`, users need not worry about not having the required dependencies when using the Automatic Differentiation package. 

In case users fail to get the required dependencies during pip install, users can still refer to the contents of requirements.txt below to pip install the main dependencies that are required. If not, users can visit https://pypi.org/project/autodiffing/#files to download the latest gunzip tar file and unzip the contents to get the requirements.txt file. In the directory with the unzipped folder containing the requirements.txt file, users need to run the following command in the terminal to download the required dependencies:

`pip install -r requirements.txt`

Within our requirements.txt, we have the a number of packages that come with the installation of `python` version 3.7 and our main packages, but the main packages that we require for our Automatic Differentiation package are: 

`numpy==1.17.4`\
`matplotlib==3.1.1`\
`scipy==1.3.2`
`math`

`numpy` is essential for our Automatic Differentiation package as we require it for the calculation of our elementary functions, and for dealing with arrays and matrices when there are vector functions and vector inputs.

`matplotlib` is needed for any potential visualization of our outputs.

`scipy` is a good package to have for its optimization and linear algebra abilities.

`math` is a package used for a couple numerical comparisons and to restrict the domains of certain functions.

## Using the Package

Once users have installed all the dependencies and the package itself, they may begin to use our package to quickly find derivatives of functions.  For this section, we walk through three different examples of how users can interact with the package for their purposes.

Note that we made some implementation and stylistic changes from Milestone 2 to the final documentation.  We will review a few of these changes here and will give greater justification in the software organization and implementation details sections for the changes.  First, the user should import necessary modules as:

```python
from AD.DualNumber import DualNumber
from AD import ElementaryFunctions as EF
from AD.Parallelized import Parallelized_AD
```

Whereas in Milestone 2, we limited user imports to the DualNumber and ElementaryFunctions modules, we now add functionality for vectorized inputs and outputs, which we implement in our Parallelized module (and specifically Parallelized_AD class).  Moreover, we extend our class to have reverse mode functionality, the background for which we described in the introduction and background sections.  Once users have imported our classes, they can initialize functions and variables as follows:

```python
# func and var will eventually be inputs to our Automatic Differentiation implementation
func = ['_x + sin(_y)*_z', 'sqrt(_y + _x) - cos(_z)']
varis = ['x', 'y', 'z']
```

Note that when the user initializes a function, he or she _must_ adhere to the format seen above.  That is, all variables must be preceded by a leading underscore '\_'.   This is a significant departure from our first two milestones, when we asked the user to set up variables as x = DualNumber(5), func = EF.Sin(x).  Our new choice for the user interface we believe makes our software more intuitive and easier to use.  In particular, rather than having to set up a new DualNumber for _every_ new variable, the user may just list the variables in an array.  Then, the user may just symbolically specify the function, which we believe more closely resembles a "natural" syntax.

That is, in the final documentation, we _hid_ the implementation details from the user and allowed them to interact with the class 'symbolically'.  We provide the user with a dictionary of functions, including, but not limited to, _sin_ , _cos_ , _log_ , and _logistic_ (as four examples).

So, for example, after the user has initialized the function and variables as above, he/she may use the reverse mode of automatic differentation as follows:

```python
>>> PAD = Parallelized_AD(fun = func, var = varis)
>>> PAD.get_Jacobian([1,2,3])
array([[ 1.        , -1.24844051,  0.90929743],
       [ 1.        , -8.35853265, 18.26372704]])

>>> PAD.get_value([1,2,3])
array([ 3.72789228, 19.26372704])

>>> PAD.get_Jacobian([1,2,3], forward=True)
array([[ 1.        , -1.24844051,  0.90929743],
       [ 1.        , -8.35853265, 18.26372704]])
```

Note that the default differentation mode is reverse mode, but this can be changed with setting forward=True as we did in the last line.  The user should always instatiate a Parallelized automatic differentation object before taking the Jacobian or returning a value.  During initialization, the user should pass in one argument for the function, which may be vector or scalar valued, as an array of strings (one for each function), and one argument for each variable, an array of strings for each variable's name.

To extract the Jacobian, the user must specify a point at which to take the Jacobian.  In the example above, the Jacobian was taken at [x,y,z] = [1,2,3].  If, for example, the user took the Jacobian at [1,2], or some other invalid point, then our Parallelized_AD class would throw an error.



However, it is more likely that users will want to use our class for more complicated uses.  Here we wish to find the Jacobian of the function

$$f_1(x,y,z) = x + \sin(y)z, f_2(x,y,z) = x+\sin(y)\exp(z)$$

That is, our function output is a two-dimensional vector which takes in a three-dimensional vector as an input.  The user could use our package as follows:

```python
import AD.ElementaryFunctions as EF
from AD.DualNumber import DualNumber
from AD.Parallelized import Parallelized_AD
func = ['_x + sin(_y)*_z', '_x + sin(_y)*exp(_z)']
PAD = Parallelized_AD(fun = func, var = ['x', 'y', 'z'])

print("JACOBIAN: ")
print(PAD.get_Jacobian([1,2,3]))

print("VALUE: ")
print(PAD.get_value([1,2,3]))
```

Now suppose the user wants to add a third function, with a fourth variable $w$.  The new function will look like:

$$f_1(x,y,z,w) = x + \sin(y)z, f_2(x,y,z,w) = x+\sin(y)\exp(z), f_3(x,y,z,w) = w$$

Our package allows the user to simply add a variable with our add_var() method and add a function with the add_function() method.  Perhaps for some reason the user wants to calculate derivatives using the forward mode, instead of the reverse mode.  We combine these steps in the 

```python
PAD.add_var('w')
PAD.add_function('_w')

print('JACOBIAN:')
print(PAD.get_Jacobian([1,2,3,4]))

print("Value: ")
print(PAD.get_value([1,2,3,4]))
```

Users have access to our dictionary of functions, including `sin`, `cos`, `tan`, `exp`, `power`, `arcSin`, `log`, `sqrt`, `arcSin`, `arcCos`, `arcTan`, `sinh`, `cosh`, `tanh`, and `logistic`.

Note that the syntax must be met _exactly_ (including capitalization!) during the string input to the function; failure to do so will result in either an `AssertionError` or erroneous results.

Our software can be used for a variety of purposes, including optimization and gradient descent algorithms.


