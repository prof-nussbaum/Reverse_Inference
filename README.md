# Reverse_Inference
Working backwards through a deep convolutional network, to recreate the input image - and identify areas for improvement.

Please see this article for more details on this technique, that I call "Reading the Robot Mind."
https://readingtherobotmind.blogspot.com/2021/03/reading-robot-mind-deep-convolution.html

<h2>What if we could read the robot's mind?</h2>

It sounds like a silly statement, but if you could read a person's mind, you could "see" what they were thinking of when they mention a classification. If they say "I am seeing a dog" - reading their mind would give you additional details about the dog, or even perhaps, see what they are seeing. This is the same premise for Artificial Intelligence and Machine Learning. 

I know that deep convolutional neural networks (CNN) do not have minds like people do, by any stretch of the imagination, but the the catchphrase "reading the robot mind" is simply a way to remember what this proposed extension to the important topic of "Explainability" is all about.

So, personification aside, the ability to work backwards through the CNN is helpful to spot if sufficient information has been fed forwards for classification. It is also helpful to specifically observe incorrect and/or low-confidence classifications, and not only view the original input image, but also the internal rendering of that image by the CNN.

I have included software as well as a sample runs showing these visualizations and images to explain the point. 

<h2>Example 1: Handwriting Recognition CNN</h2>
This example uses a data set from www.kaggle.com that has a number of "Graphemes" from the Bengali language. We can see correctly and incorrectly classified images - as would be normal for data scientists to do. In addition, we can recreate these images based on internal representations within the CNN. From these visualizations, we can make improvements to the model and/or identify bias that leads to incorrect classification.
