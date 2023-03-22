# LYCET: an automatic differentiation package

## Group Members:
$\color{rgb(0, 176, 255)}{\text{(L)oralee Ryan}}$

$\color{rgb(0, 176, 255)}{\text{(Y)anis Vandecasteele}}$

$\color{rgb(0, 176, 255)}{\text{(C)helsey Campillo}}$

$\color{rgb(0, 176, 255)}{\text{(E)laine Swanson}}$

$\color{rgb(0, 176, 255)}{\text{(T)adhg Looram}}$

## About Automatic Differentiation
Automatic Differentiation (AD) is a method which can be used to efficiently compute the derivative of a given function to machine precision. It does so by leveraging the power of the chain rule and the fact that complex functions are simply composites of other functions. Elemenary functions covered by our LYCET package include:
* Constant functions
* Rational powers of x
* Exponential functions and their inverses
* Trigonometric functions and their inverses
* Composite functions:
    * Addition and subtraction
    * Multiplication and division
    * Polynomial functions

If you would like to read an in-depth review on AD, you can take a look at our [\docs\documentation.ipynb](https://code.harvard.edu/CS107/team18/blob/main/docs/documentation.ipynb) file.

## How-to-use: 
1. Import the packages:
    1. For best practice, create a virtual environment: python3 -m venv /path/to/new/virtual/environment e.g. python3 -m venv my_venv

    2. Activate the virtual environment by using: . my_venv/bin/activate for the previous example

    3. Install the required dependencies given by pyproject.toml by running:

pip install --index-url https://test.pypi.org/simple/ LYCET_package --extra-index-url https://pypi.org/simple LYCET-package==0.1.0

2. Once the package is installed, create the driver script to define the function of which to evaluate
    1. Import LYCET and numpy:
        ```
        import numpy as np
        from LYCET_package import LYCET_operations_Forward as fm
        from LYCET_package.DualNumber import DualNumber
        ```
    2. Instantiate the dualnumber object, which will simultaneously evaluate f(x) and f'(x) using the real and dual attributes, respectively:
        ```
        x = 4 # this is the point at which you would like to evaluate
        z = DualNumber(x)
        ```
    3. Define f(x):
        ```
        f = fm.exp(z) + fm.sin(lop.exp(z))
        ```
    4. Obtain the results from forward AD:
        ```
        print(f.real)
        print(f.dual)
        ```

    _Please see [\src\demo.ipynb](https://code.harvard.edu/CS107/team18/blob/main/src/demo.ipynb), for more expansive demos for both Forward and Reverse Mode._


## Broader Impact
Differentiable programming has been applied to many research areas in hopes to increase speed and stability of calculations. On the surface, making a process of one's research/business more efficient and less expensive sounds positive and beneficial to interested parties. However, there are many instances in the programming realm that ignore or outright disregard the long-term ethical implications of doing so. For instance, differentiable programming is often used to investigate optimization. This overall strategy is often followed by decision making and implementation. This reasonable application may have problematic strategies hidden in the larger strategy space. For example, in the world of business, optimization is geared toward maximizing profits. What are the consequences of seeing this optimization through? For the employees and/or functions of the business? Will people lose their livelihoods? Their health insurance?

Our LYCET package is a framework that allows the user to input any complex function that our encoded elementary operations support. What this function relates to in research and the real world is seemingly out of our control. We can state a disclaimer and say that we are not responsible for how it is used post publication, but this no longer fits within the mindset of the programming community. Since we are making our package available on PYPI, anyone within the Python community will have access to it and apply it to their problem. There are limits of what we can do, but we choose to not ignore the consequences of our package being used or misused. Of course, the first step is addressing these issues and stating them within our package instructions. 

We suggest to anyone that downloads and uses LYCET, to be aware and educated about the process of automatic differentiation and consider the after effects that may follow. As a whole, the computing research community must do more to address the downsides of our innovations. 
We require downloaders to ask these important questions before use of our package:
- What are the short-term and long-term impacts of this process?
- What are the net positive or net negative impacts of this program?
- Will this have an impact on my well-being or the well-being of people now or in the future?

There are many analytics websites for PyPI package downloads (https://pypistats.org/) so we are able to see how many times our package has been downloaded. We are interested in who is using our software package, for what purpose, and any improvements they think of. We are hoping to facilitate an open conversation about the broader impact of our simple AD package. We hope you enjoy our work! 

## Software Inclusivity

In its current state, the tech-ecosystem needs to continue to foster an inclusive and diverse environment. The modern workforce is becoming increasingly diverse and it has been reported that this change is improving the complex nature of the work environment in several ways:
- Diverse and inclusive teams bring innovation
- Diverse teams are better at making decisions 
- Inclusivity leads to higher work engagement<br>
    - When companies foster a more inclusive work environment, 83% of Millennials are found to be actively engaged in their work
- American companies are simply becoming more diverse<br>
    - By 2044, more than half of all Americans are projected to belong to a minority group (those falling outside white non-Hispanic), accounting for the majority of the U.S. population by this time.

[Source](https://builtin.com/diversity-inclusion/diversity-in-the-workplace-statistics) 

We encourage the use of our LYCET package to be used in diverse and inclusive teams to solve real world problems. <br>
We have identified a few barriers that might arise when working with and executing our package.
- Similar to core Python development, our package was developed in English. This is the obviously the largest barrier for non-English speakers. To understand our documentation and comments, we encourage the use of Google translate.  
- We also want to make the public components of our code to be understandable to beginning programmers. We encourage the use of [Denigma](https://denigma.app/) and [PythonTutor](https://pythontutor.com/) to understand how and why we coded our package in this way.
- Finally, after our team submits the package and we make improvements after grading, we hope to push our package to a public Github. With our package being open-source, we welcome all coders of any age, culture, ethnicity, gender identity or expression, national origin, physical or mental difference, politics, race, religion, sex, sexual orientation, socio-economic status, and/or subculture to contribute to our code base. 

# Changes to package:

Incorporating a function that generates a dynamic computational graph for the user to interact with, similar to PyTutor
Implement a solver for second-order derivatives (i.e., Hessian)
Currently, both forward and reverse mode utilize their own set of elementary operations, however in a future iteration of this package, it would be best to consolidate lycet_operations_forward and lycet_operations_reverse to eliminate any redundancy.
Research Applications: Team members would like to investigate how AD can be used in their own research interests. For example:

Satellite data analysis
In satellite data analysis, automatic differentiation can be used to calculate the derivatives of functions that describe the physical processes being studied, such as the motion of satellites or the behavior of gases in the atmosphere.
AD can also be used to optimize the parameters of machine learning algorithms that are applied to satellite data, which can improve the accuracy of the models and the results of the analysis.
Modeling fluid in microfluidics
By using automatic differentiation, researchers can quickly and accurately calculate the derivatives of the equations describing the fluid's behavior. These derivatives can then be used to solve the equations and model the fluid's motion. This can help researchers better understand the behavior of fluids in microfluidic systems and design more effective microfluidic devices.


![coverage.yml](https://code.harvard.edu/CS107/team18/actions/workflows/code_coverage.yml/badge.svg)

![tests.yml](https://code.harvard.edu/CS107/team18/actions/workflows/tests.yml/badge.svg)
