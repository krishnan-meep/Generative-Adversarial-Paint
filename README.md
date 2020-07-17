# Generative-Adversarial-Paint

Paint? Hurrah!

![Toolkit](https://github.com/krishnan-meep/Generative-Adversarial-Paint/blob/master/weights/Toolkit_MainScreen.png)

Repository to hold my basic Tkinter paint editor that incorporates Generative Adversarial Networks for zaniness
Run the GANGUI.py file to take a look at the toolkit. Please download the [GLoVE](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiu7teM7dPqAhVRg-YKHXlRAhQQFjAAegQIAxAB&url=https%3A%2F%2Fwww.kaggle.com%2Fwatts2%2Fglove6b50dtxt&usg=AOvVaw0dh78z6Lu803rAkIcZgkUJ) word embeddings file and place it in the weights folder.

![GG](https://github.com/krishnan-meep/Generative-Adversarial-Paint/blob/master/weights/gauganize.png)

Requires tkinter, torch, torchvision.

## Change Log (Maybe)

9th April 2020
- Really should've started this sooner, I'm a week in and I have no idea what I've done day to day for the last week. Better late than ever? Anyway to summarize, I have the following features in already
 	- Pencil drawing with stroke width and color
 	- Non transparent eraser
 	- Layer functionality - Add, delete, raise, lower layers
 	- Move objects in a layer
 	- Open images and load to canvas
 	- Save images from the canvas (Postscript -> PIL and maybe -> CV/numpy)
 	- Generate images and interpolate with a single model
 	- Apply style transfer with a single model
 	- Super resolution 4x
 	- Undo and Redo functionalities

- All the above features aren't complete and most have obscure bugs such as 
	 - The interaction between layers and undo/redo
	 - Not so much a bug but CANVAS TRANSPARENCY IS IMPOSSIBLE

- Features to be added
 	- Extend generate functionality to multiple models
 	- Extend style transfer to multiple models
 	- Actually train working models when possible
 	- Implement merge layers function
 	- Text to image GAN models
 	- This is so complex I don't think it's worth the limitations it would impose; a selection box that
 	  you can move around and apply operations in specifically.

- Also, a side idea for later. Someone must've written a better canvas python library you can use instead.

23rd April 2020
- It's been a while, I need to watch my video again
- I've added Gaugan as a separate menu option since it's in between synthesis and translation GANs
	and also added dropboxes to the popups so you can choose between different models. Clicking on 
	one of the listed models loads it in and shows you an example. I've also moved the models into
	a separate py file to make it less cluttered.

27th April 2020
- Added not fully trained StarGAN and CIFAR-10 models for better demonstrative purposes, can choose
	between different models to use for image translation

17th July 2020
- Added the GauGAN model

