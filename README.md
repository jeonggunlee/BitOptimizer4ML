# BitOptimizer4ML
Bit Width Optimization and Architectural Renovation for Embedded Machine Learning

*  *  *
# Model Parameter Optimization of a Deep Neural Network - MNIST Case Study

E-SoC Lab / Smart Computing Lab, School of Software, Hallym University, South Korea

```Abstract```: In order to operate an artificial neural network (ANN) based machine learning software in an embedded system, it is essential to reduce the model size due to the limited resources. Particularly, learnable parameters of ANN such as weights and biases can consume large amount of a system memory. Consequently, it is necessary to optimize the required memory for the learnable parameters while maintaining the model accuracy. In this work, we investigate a memory size optimization of a convolutional neural network (CNN) model in the case of the MNIST handwritten digit recognition. We study the impact on the performance of CNN when we reduce the bit-widths of the floating point variables used for the weights and biases. 

### Keywords—Deep neural network, Bit-width optimization, MNIST.

## I. INTRODUCTION

Recently, a deep learning becomes a core technology in artificial intelligence and the deep learning has been applied to various real-life applications. The deep learning becomes a feasible technology thanks to innovative deep learning algorithms, big data and high performance computing (HPC) supported by data processing accelerators such as GPUs or FPGAs. In spite of the rapid advance of the technology, deep learning still demands long training time and its inference is hard to be run in real-time when the deep learning model is large. In particular, deep learning is not easily deployed in a mobile platform due to the limits of memory and processing power. To deploy a machine learning in such an embedded system, we have to develop viable solutions for the three challenges: 1)model size, 2) speed and 3)energy efficiency. 

A modern deep learning model is built with much larger and deeper networks than before and there are many nodes in each layer of the networks. Recently, a lot of deeper models have been proposed and have showed significant performance improvements for visual object detections and identifications. Particularly, Microsoft Research "ResNet", which is the current state of the art for various object recognitions, have used up to 152 or even a 1000 layers. In consequence, there are many weights and biases in the deep learning model. It means that computational demands and memory requirement for the model can be a critical design issue in a resource-limited mobile embedded system. In particular, the more off-chip memory references in larger deep learning models consume more energy, and that is another serious problem in a battery-powered mobile system.

In this work, we try to optimize weight and bias parameters so that we can minimize the operational and memory requirements for real-time inferences with a deep learning model. In particular, our parameter optimizations are conducted without the degradation of inference quality or with a ignorable drop of the quality. We also discuss the impact of the parameter optimizations on the prediction accuracy of a deep learning model. Particularly, the relationship between a parameter precision and an overfitting problem will be discussed. A CNN based MNIST (handwritten digit recognition) deep learning model has been used most widely as a comprehensible
example in a deep learning community. We use the MNIST model for our study and we can derive some intuitional interpretations behind the results more easily through the commonly used small example. However, we expect that our work can be applied to more complex deep learning applications to some extent consistently.

*  *  *
The following code is for truncating a 32-bit floating-point (FP) number (```num```) to ```bit```-bit floating-point number.

```python
def binRep(num, bits):
    binNum = bin(ctypes.c_uint.from_buffer(ctypes.c_float(num)).value)[2:]
    temp1 = binNum.rjust(32,"0")
    temp2 = temp1[0:bits]
    binNum = temp2.ljust(32,"0")
    #print("bits: " + binNum.rjust(32,"0"))
    mantissa = "1" + binNum[-23:]
    #print("sig (bin): " + mantissa.rjust(24))
    mantInt = int(mantissa,2)/2**23
    #print("sig (float): " + str(mantInt))
    base = int(binNum[-31:-23],2)-127
    #print("base:" + str(base))
    sign = 1-2*("1"==binNum[-32:-31].rjust(1,"0"))
    #print("sign:" + str(sign))
    #print("recreate:" + str(sign*mantInt*(2**base)))
    return sign*mantInt*(2**base)
```

Using the above function, we can make ```bit```-bit floating-point number for a given 32 bit FP number.


## 
