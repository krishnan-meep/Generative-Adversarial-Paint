# Generative-Adversarial-Paint

Paint? Hurrah!

Repository to hold my basic Tkinter paint editor that incorporates Generative Adversarial Networks for zaniness
Run the GANGUI.py file to take a look at the toolkit.

#Change Log (Maybe)

9th April 2020
---- Really should've started this sooner, I'm a week in and I have no idea what I've done day to day
	 for the last week. Better late than ever? Anyway to summarize, I have the following features in already
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

---- All the above features aren't complete and most have obscure bugs such as 
	 - The interaction between layers and undo/redo
	 - Not so much a bug but CANVAS TRANSPARENCY IS IMPOSSIBLE

---- Features to be added
	 	- Extend generate functionality to multiple models
	 	- Extend style transfer to multiple models
	 	- Actually train working models when possible
	 	- Implement merge layers function
	 	- Text to image GAN models

	 	- This is so complex I don't think it's worth the limitations it would impose; a selection box that
	 	  you can move around and apply operations in specifically.

---- Also, a side idea for later. Someone must've written a better canvas python library you can use instead.

23rd April 2020
--- It's been a while, I need to watch my video again
--- I've added Gaugan as a separate menu option since it's in between synthesis and translation GANs
	and also added dropboxes to the popups so you can choose between different models. Clicking on 
	one of the listed models loads it in and shows you an example. I've also moved the models into
	a separate py file to make it less cluttered.