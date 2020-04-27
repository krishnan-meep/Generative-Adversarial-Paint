from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image, ImageGrab
import torch
import cv2

class generatePopup(object):
	def __init__(self, master, model, seg_map = None):
		self.master = master
		top = self.top = Toplevel(master)
		self.b1 = Button(top, text = "Generate", command = self.gen_examples)
		self.b2 = Button(top, text = "Interpolate", command = self.interpolate)
		self.b3 = Button(top, text = "Add to canvas", command = self.cleanup)
		h, w = 192,192

		self.seg_map = seg_map
		self.model = model
		self.img_t3 = None
		self.apply = False

		self.l1 = Label(top, text = "Image 1")
		self.l2 = Label(top, text = "Interpolated Image")
		self.l3 = Label(top, text = "Image 2")
		self.c1 = Canvas(top, height = h, width = w, background = "white")
		self.c2 = Canvas(top, height = h, width = w, background = "white")
		self.c3 = Canvas(top, height = h, width = w, background = "white")

		if seg_map is None:
			model_list = self.model.gen_models_list
		else:
			model_list = self.model.gaugan_models_list
		Label(top, text = "Pick a model").grid(row = 2, column = 0)
		self.drop1 = ttk.Combobox(top, values = model_list, state = "readonly", width = 20)
		self.drop1.current(0)
		self.drop1.bind("<<ComboboxSelected>>", self.change_gen_model)

		self.slider = Scale(top, from_= 0, to = 100, tickinterval = 100, orient = HORIZONTAL, length = 120)

		self.l1.grid(row = 0, column = 0, sticky = N, padx = 4, pady = 4)
		self.l2.grid(row = 0, column = 2, sticky = N, padx = 4, pady = 4)
		self.l3.grid(row = 0, column = 4, sticky = N, padx = 4, pady = 4)
		self.c1.grid(row = 1, column = 0, sticky = N, padx = 4, pady = 4)
		self.c2.grid(row = 1, column = 2, sticky = N, padx = 4, pady = 4)
		self.c3.grid(row = 1, column = 4, sticky = N, padx = 4, pady = 4)

		self.drop1.grid(row = 2, column = 0, sticky = S, padx = 4, pady = 4)
		self.slider.grid(row = 2, column = 2, sticky = S, padx = 4, pady = 4)
		self.b1.grid(row = 3, column = 0, sticky = N, padx = 4, pady = 4)
		self.b2.grid(row = 3, column = 2, sticky = N, padx = 4, pady = 4)
		self.b3.grid(row = 3, column = 4, sticky = N, padx = 4, pady = 4)

	def gen_examples(self):
		if self.seg_map is None:
			img1, self.noise1 = self.model.generate()
			img2, self.noise2 = self.model.generate()
		else:
			img1, self.noise1 = self.model.gaugan_translate(self.seg_map)
			img2, self.noise2 = self.model.gaugan_translate(self.seg_map)

		self.img_t1 = ImageTk.PhotoImage(img1)
		i1 = self.c1.create_image(0, 0, anchor=NW, image = self.img_t1)
		self.img_t2 = ImageTk.PhotoImage(img2)
		i2 = self.c3.create_image(0, 0, anchor=NW, image = self.img_t2)

		self.interpolate()

		self.master.update_idletasks()
		self.top.update_idletasks()

	def interpolate(self):
		alpha = float(self.slider.get())/100
		noise = alpha*self.noise2 + (1 - alpha)*self.noise1								#Cause of how the slide works
		if self.seg_map is None:
			img3, _ = self.model.generate(noise)
		else:
			img3, _ = self.model.gaugan_translate(self.seg_map, noise)
		self.img_t3 = ImageTk.PhotoImage(img3)
		i3 = self.c2.create_image(0, 0, anchor=NW, image = self.img_t3)

		self.master.update_idletasks()
		self.top.update_idletasks()

	def change_gen_model(self, event = None):
		gen_model = event.widget.get()
		print(gen_model)

	def cleanup(self):
		self.apply = True
		self.top.destroy()


class stylePopup(object):
	def __init__(self, master, model):
		top = self.top = Toplevel(master)
		self.model = model
		h, w = 128, 128
		self.style_value = None
		self.apply = False

		self.l1 = Label(top, text = "Original")
		self.l2 = Label(top, text = "Result")
		self.c1 = Canvas(top, height = h, width = w, background = "white")
		self.c2 = Canvas(top, height = h, width = w, background = "white")

		self.white_check = IntVar(value = 1)
		Checkbutton(top, text="Apply on white?", variable=self.white_check).grid(row=1, column = 1, sticky = N)


		self.l1.grid(row = 0, column = 0, sticky = N)
		self.l2.grid(row = 0, column = 2, sticky = N)
		self.c1.grid(row = 1, column = 0, sticky = N)
		self.c2.grid(row = 1, column = 2, sticky = N)

		self.c1.create_rectangle(30, 30, 60, 60, fill = "#A84444")
		self.c1.create_rectangle(60, 60, 90, 90, fill = "#44A844")
		self.c1.create_rectangle(30, 90, 60, 120, fill = "#4444A8")

		self.c2.create_rectangle(30, 30, 60, 60, fill = "#A84444")
		self.c2.create_rectangle(60, 60, 90, 90, fill = "#44A844")
		self.c2.create_rectangle(30, 90, 60, 120, fill = "#4444A8")

		model_list = self.model.style_models_list
		Label(top, text = "Pick a style model").grid(row = 2, column = 0)
		self.drop1 = ttk.Combobox(top, values = model_list, state = "readonly", width = 20)
		self.drop1.current(0)
		self.drop1.grid(row = 2, column = 1)
		self.drop1.bind("<<ComboboxSelected>>", self.example)

		Button(top, text = "Apply", command = self.cleanup).grid(row = 2, column = 2)

	def example(self, event=None):
		self.style_value = event.widget.get()

		for i in self.c2.find_all():
			self.c2.delete(i)

		img = self.canvas_to_cv()

		if not self.white_check.get():
			mask = (img == 255).all(axis=2)
		else:
			mask = None

		self.model.load_style_model(self.style_value)
		gen_img = self.model.translate(img, mask, "style")
		self.img_t = ImageTk.PhotoImage(gen_img)

		self.img_2 = self.c2.create_image(0, 0, anchor = NW, image = self.img_t)
		self.top.update_idletasks()

	def cleanup(self):
		self.apply = True
		self.top.destroy()

	def canvas_to_pil(self):
		self.c1.postscript(file = 'temp.eps', pagewidth = 128 + 3, pageheight = 128 + 3)
		img = Image.open('temp.eps')
		return img

	def canvas_to_cv(self):
		img = self.canvas_to_pil()
		img.save('temp.png', 'png')
		img = cv2.imread('temp.png')
		img = img[:128,:128]
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		return img

class newCanvasPopup(object):
	def __init__(self,master):
		top =self.top = Toplevel(master)
		self.l1 = Label(top, text = "Enter Width: ")
		self.l2 = Label(top, text = "Enter Height: ")
		self.he = Entry(top)
		self.we = Entry(top)
		self.b = Button(top,text='Create Canvas',command=self.cleanup)
		self.h, self.w = 256, 256

		self.l1.grid(row = 0, column = 0)
		self.l2.grid(row = 1, column = 0)
		self.we.grid(row = 0, column = 1)
		self.he.grid(row = 1, column = 1)
		self.b.grid(row = 2, column = 3)

	def cleanup(self):
		self.h = self.he.get()
		self.w = self.we.get()
		self.top.destroy()

class mergePopup(object):
	def __init__(self, master, layer_list):
		top =self.top = Toplevel(master)
		Label(top, text = "Pick first layer").grid(row = 0, column = 0, sticky = N, padx = 4, pady = 4)
		Label(top, text = "Pick second layer").grid(row = 1, column = 0, sticky = N, padx = 4, pady = 4)

		self.first, self.second = layer_list[0], layer_list[0]
		self.drop1 = ttk.Combobox(top, values = layer_list, state = "readonly", width = 20)
		self.drop2 = ttk.Combobox(top, values = layer_list, state = "readonly", width = 20)
		self.button = Button(top, text = "Merge", command = self.cleanup)

		self.drop1.current(0)
		self.drop2.current(0)

		self.drop1.grid(row = 0, column = 2, sticky = N, padx = 4, pady = 4)
		self.drop2.grid(row = 1, column = 2, sticky = N, padx = 4, pady = 4)
		self.button.grid(row = 2, column = 0, sticky = N, padx = 4, pady = 4)

	def cleanup(self):
		self.first, self.second = self.drop1.get(), self.drop2.get()
		self.top.destroy()