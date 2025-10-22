<H3>NAME:DHARUNYADEVI S</H3>
<H3>REGISTER NUMBER:212223220018</H3>
<H3>EX. NO.5</H3>
<H3>DATE:22/10/2025</H3>
<H1 ALIGN =CENTER>Implementation of XOR  using RBF</H1>
<H3>Aim:</H3>
To implement a XOR gate classification using Radial Basis Function  Neural Network.

<H3>Theory:</H3>
<P>Exclusive or is a logical operation that outputs true when the inputs differ.For the XOR gate, the TRUTH table will be as follows XOR truth table </P>

<P>XOR is a classification problem, as it renders binary distinct outputs. If we plot the INPUTS vs OUTPUTS for the XOR gate, as shown in figure below </P>




<P>The graph plots the two inputs corresponding to their output. Visualizing this plot, we can see that it is impossible to separate the different outputs (1 and 0) using a linear equation.
A Radial Basis Function Network (RBFN) is a particular type of neural network. The RBFN approach is more intuitive than MLP. An RBFN performs classification by measuring the input’s similarity to examples from the training set. Each RBFN neuron stores a “prototype”, which is just one of the examples from the training set. When we want to classify a new input, each neuron computes the Euclidean distance between the input and its prototype. Thus, if the input more closely resembles the class A prototypes than the class B prototypes, it is classified as class A ,else class B.
A Neural network with input layer, one hidden layer with Radial Basis function and a single node output layer (as shown in figure below) will be able to classify the binary data according to XOR output.
</P>





<H3>ALGORITHM:</H3>
Step 1: Initialize the input  vector for you bit binary data<Br>
Step 2: Initialize the centers for two hidden neurons in hidden layer<Br>
Step 3: Define the non- linear function for the hidden neurons using Gaussian RBF<br>
Step 4: Initialize the weights for the hidden neuron <br>
Step 5 : Determine the output  function as 
                 Y=W1*φ1 +W1 *φ2 <br>
Step 6: Test the network for accuracy<br>
Step 7: Plot the Input space and Hidden space of RBF NN for XOR classification.

<H3>PROGRAM:</H3>
```py
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([0,1,1,0])

centers = np.array([[0,1],[1,0]])
sigma = 0.5

def rbf(x, c, sigma):
    return np.exp(-np.linalg.norm(x-c)**2 / (2*sigma**2))

phi = np.zeros((len(X), len(centers)))
for i in range(len(X)):
    for j in range(len(centers)):
        phi[i][j] = rbf(X[i], centers[j], sigma)

W = np.dot(np.linalg.pinv(phi), Y)

Y_pred = np.dot(phi, W)
Y_pred = np.round(Y_pred)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

for i, x in enumerate(X):
    color = 'orange' if Y[i]==1 else 'blue'
    label = f'Class_{Y[i]}' if i<2 else None
    axs[0].scatter(x[0], x[1], c=color, s=100, label=label)
axs[0].set_title('XOR: Linearly Inseparable')
axs[0].set_xlabel('X1')
axs[0].set_ylabel('X2')
axs[0].legend()
axs[0].grid(True)

for i in range(len(X)):
    color = 'orange' if Y[i]==1 else 'blue'
    axs[1].scatter(phi[i,0], phi[i,1], c=color, s=100)
axs[1].set_title('Transformed Inputs: Linearly Separable')
axs[1].set_xlabel('mu1: [0 1]')
axs[1].set_ylabel('mu2: [1 0]')

x_line = np.linspace(0, 1, 100)
y_line = -x_line + 1
axs[1].plot(x_line, y_line, 'k--', label='Separating hyperplane')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()

print("Predicted Output:", Y_pred.astype(int))
print("Actual Output:   ", Y)
```

<H3>OUTPUT:</H3>
<img width="1127" height="506" alt="image" src="https://github.com/user-attachments/assets/fa1a61f6-5d9d-4ced-b9fb-fd8e023b75c1" />


<H3>Result:</H3>
Thus , a Radial Basis Function Neural Network is implemented to classify XOR data.








