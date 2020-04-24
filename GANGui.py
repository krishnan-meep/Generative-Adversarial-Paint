from tkinter import *
from tkinter import ttk
from popups import *
from tkinter import filedialog
from PIL import ImageTk, Image, ImageGrab
import numpy as np
from queue import LifoQueue
import copy
import threading
import time
import os
import sys
import io
import cv2
from gan_models import GANModels

#######################################################################################################################
#Here's the main GUI window class
#######################################################################################################################
class GANGui:
	def __init__(self, root, lock):
		self.img = None
		self.root = root
		self.model = GANModels()
		self.lock = lock

		self.initialize_canvas()

		self.undo_stack = [self.get_canvas_state()]							#Each element contains (all_items-->(item, layer), layer_list, layer_image_dict, curr_layer)
		self.redo_queue = []

		self.menu = Menu(root)
		self.menu.config(background = "#111114")
		self.file_menu = Menu(self.menu, tearoff = 0)
		self.file_menu.add_command(label="New", command = self.new_canvas)
		self.file_menu.add_command(label="Open", command = self.open_file)
		self.file_menu.add_command(label="Save", command = self.save_file)

		self.edit_menu = Menu(self.menu, tearoff = 0)
		self.edit_menu.add_command(label="Undo", command = lambda : self.undo_redo("undo"))
		self.edit_menu.add_command(label="Redo", command = lambda : self.undo_redo("redo"))
		self.edit_menu.add_command(label="Merge Layers", command = self.merge_layers)

		self.filter_menu = Menu(self.menu, tearoff = 0)
		self.filter_menu.add_command(label="Apply Style", command = lambda :  self.translate("style"))
		self.filter_menu.add_command(label = "Gauganize", command = lambda : self.generate(gaugan = True))
		self.filter_menu.add_command(label="Apply Super Resolution(4x)", command = lambda : self.translate("sr"))

		self.menu.add_cascade(label = "File", menu = self.file_menu)
		self.menu.add_cascade(label = "Edit", menu = self.edit_menu)
		self.menu.add_cascade(label = "Filter", menu = self.filter_menu)

		self.toolbar = Frame(root, bd = 4, relief = RAISED)
		self.toolbar.config(background = "#313134")
		self.toolbar.pack(anchor = W, padx = 2, pady = 2)

		self.tool_move = Button(self.toolbar, text = "Mv", width = 3, command = lambda : self.set_tool("move"))
		self.tool_pencil = Button(self.toolbar, text = "Pncl", width = 3, command = lambda : self.set_tool("pencil"))
		self.tool_eraser = Button(self.toolbar, text = "Ersr", width = 3, command = lambda : self.set_tool("eraser"))
		self.tool_gen = Button(self.toolbar, text = "Gen", width = 3, command = self.generate)
		self.tool_move.grid(row = 0, column = 0, sticky = W)
		self.tool_pencil.grid(row = 0, column = 1, sticky = W)
		self.tool_eraser.grid(row = 1, column = 0, sticky = W)
		self.tool_gen.grid(row = 1, column = 1, sticky = W)

		Label(self.toolbar, text = "Stoke Width", background = "#313134", foreground = "#FFFFFF").grid(row = 0, column = 2, padx = 4)
		self.width_entry = Entry(self.toolbar, width = 3)
		self.width_entry.bind("<Key>", self.get_line_width)
		self.width_entry.grid(row = 0, column = 3, sticky = W, padx = 2)

		Label(self.toolbar, text = "Stroke Fill (RGB)", background = "#313134", foreground = "#FFFFFF").grid(row = 1, column = 2, padx = 4)
		self.R_entry = Entry(self.toolbar, width = 4)
		self.G_entry = Entry(self.toolbar, width = 4)
		self.B_entry = Entry(self.toolbar, width = 4)
		self.R_entry.bind("<Key>", self.get_fill_R)
		self.G_entry.bind("<Key>", self.get_fill_G)
		self.B_entry.bind("<Key>", self.get_fill_B)
		self.R_entry.grid(row = 1, column = 3, sticky = W, padx = 2)
		self.G_entry.grid(row = 1, column = 4, sticky = W, padx = 2)
		self.B_entry.grid(row = 1, column = 5, sticky = W, padx = 2)
		self.fill_color_indicator = Canvas(self.toolbar, height = 10, width = 10, background = self.fill_color)
		self.fill_color_indicator.grid(row = 1, column = 6, sticky = W, padx = 3)

		root.config(menu = self.menu)

	#######################################################################################################################
	#MENU FUNCTION CALLS
	#######################################################################################################################
	def new_canvas(self):
		self.w = newCanvasPopup(self.root)
		self.root.withdraw()
		self.root.wait_window(self.w.top)

		h, w = self.w.h, self.w.w
		self.canvas.delete("all")
		self.canvas.config(height = int(h), width = int(w))
		self.canvas_height, self.canvas_width = min(800, int(h)), min(800, int(w))
		h, w = int(h) + 200, int(w) + 500
		print(h, w)
		root.geometry(str(w) + "x" + str(h))

		self.layer_list = ["Layer_1"]
		self.layer_image_dict = {"Layer_1" : []}
		self.layer_dropdown.config(values = self.layer_list)
		self.layer_dropdown.current(0)
		self.no_of_layers = 1
		self.curr_layer = "Layer_1"

		self.undo_stack = [self.get_canvas_state()]
		self.redo_stack = []
		self.root.deiconify()

	def open_file(self):
		file = filedialog.askopenfilename(parent=root, initialdir="./")
		print(file)
		if file:
			self.img = Image.open(file)
			img_t = ImageTk.PhotoImage(self.img)

			self.add_new_layer()
			#self.undo_stack.pop(-1)

			i = self.canvas.create_image(0, 0, anchor=NW, image = img_t)
			self.canvas.itemconfig(i, tags = ("Layer_" + str(self.no_of_layers), "image"))
			self.layer_image_dict["Layer_" + str(self.no_of_layers)] = [img_t]

			self.push_undo()
			root.update_idletasks()

	def save_file(self):
		file = filedialog.asksaveasfilename(initialdir = "./", filetypes = (("png files","*.png"),("all files","*.*")))
		img = self.canvas_to_pil()
		img.save(file + ".png", 'png') 

	def undo_redo(self, operation = "undo"):
		if operation == "undo":
			if len(self.undo_stack) < 2:
				return
			i = self.undo_stack.pop(-1)													#Present state
			self.redo_queue.insert(0, i)
			img_t, layer_list, layer_image_dict, curr_layer = self.undo_stack[-1]		#Undo state

		else:
			if len(self.redo_queue) < 1:
				return
			i = self.redo_queue.pop(0)													#Redo state needs to be put back into undo stack
			self.undo_stack.append(i)
			img_t, layer_list, layer_image_dict, curr_layer = i

		i = self.canvas.create_image(0, 0, anchor=NW, image = img_t)					#Restoration begins
		self.canvas.itemconfig(i, tags = (curr_layer, "image"))

		self.layer_list = layer_list
		self.curr_layer = curr_layer
		self.layer_image_dict = layer_image_dict

		if not len(self.layer_image_dict[self.curr_layer]):
			self.layer_image_dict[self.curr_layer] = [img_t]
		else:
			self.layer_image_dict[self.curr_layer].insert(-1, img_t)

		self.no_of_layers = len(self.layer_list)
		self.layer_dropdown.config(values = self.layer_list)
		self.layer_dropdown.current(0)

	def push_undo(self):
		self.lock.acquire()
		self.undo_stack.append(self.get_canvas_state())
		self.lock.release()

	def push_redo(self):
		self.lock.acquire()
		self.redo_stack.insert(0, self.get_canvas_state())
		self.lock.release()


	#######################################################################################################################
	#CANVAS/DRAWING FUNCTION CALLS
	#######################################################################################################################
	def set_tool(self, tooltext):
		self.drawing_tool = tooltext

	def change_layer(self, event=None):
		self.curr_layer = event.widget.get()

	def get_line_width(self, event=None):
		w = event.widget
		try:
			self.line_width = int(w.get())
		except ValueError:
			pass

	def get_fill_R(self, event = None):
		R, G, B = event.widget.get(), self.G_entry.get(), self.B_entry.get()
		self.set_fill_color(R, G, B)

	def get_fill_G(self, event = None):
		R, G, B = self.R_entry.get(), event.widget.get(), self.B_entry.get()
		self.set_fill_color(R, G, B)

	def get_fill_B(self, event = None):
		R, G, B = self.R_entry.get(), self.G_entry.get(), event.widget.get()
		self.set_fill_color(R, G, B)

	def set_fill_color(self, R, G, B):
		try:
			if R == '': R = 0 
			if G == '': G = 0
			if B == '': B = 0
			R, G, B = int(R), int(G), int(B)
			R, G, B = max(0, min(255, R)), max(0, min(255, G)), max(0, min(255, B))
			R, G, B = hex(R)[2:], hex(G)[2:], hex(B)[2:]
			if len(R) == 1: R = "0" + R 
			if len(G) == 1: G = "0" + G
			if len(B) == 1: B = "0" + B
			self.fill_color = "#" + R + G + B
			self.fill_color_indicator.config(background = self.fill_color)
			self.root.update()
		except ValueError:
			pass

	def initialize_canvas(self):
		self.canvas_frame = Frame(root, width = 256, height = 256, pady = 10, padx = 10)
		self.canvas_frame['bg'] = '#313134'
		self.canvas = Canvas(self.canvas_frame)
		self.canvas_height, self.canvas_width = 256, 256
		self.canvas.config(background = "white", width = self.canvas_width, 
							height = self.canvas_height)
		self.canvas.bind("<Motion>", self.canvas_motion)
		self.canvas.bind("<ButtonPress-1>", self.left_but_down)
		self.canvas.bind("<ButtonRelease-1>", self.left_but_up)

		self.drawing_tool = "move"
		self.left_but = "up"
		self.img_t = None
		self.x_pos, self.y_pos = None, None
		self.x1_line_pt, self.y1_line_pt, self.x2_line_pt, self.y2_line_pt = None, None, None, None
		self.fill_color = "#000000"
		self.line_width = 5

		self.layer_list = ["Layer_1"]
		self.layer_image_dict = {"Layer_1" : []}
		self.layer_dropdown = ttk.Combobox(self.canvas_frame, values = self.layer_list, state = "readonly", width = 20)
		self.layer_dropdown.current(0)
		self.no_of_layers = 1
		self.curr_layer = "Layer_1"
		self.layer_dropdown.bind("<<ComboboxSelected>>", self.change_layer)

		self.new_layer_button = Button(self.canvas_frame, text = "+", command = self.add_new_layer)
		self.delete_layer_button = Button(self.canvas_frame, text = "-", command = self.delete_current_layer)
		self.raise_layer_button = Button(self.canvas_frame, text = "↑", command = lambda : self.raise_lower_layer("raise"))
		self.lower_layer_button = Button(self.canvas_frame, text = "↓", command = lambda : self.raise_lower_layer("lower"))

		self.canvas_frame.place(relx = 0.5, rely = 0.5, anchor = CENTER)
		self.raise_layer_button.pack(side = RIGHT, padx = 2, pady = 2)
		self.lower_layer_button.pack(side = RIGHT, padx = 2, pady = 2)
		self.layer_dropdown.pack(side = RIGHT, padx = 2, pady = 2)
		self.delete_layer_button.pack(side = RIGHT, padx =2, pady = 2)
		self.new_layer_button.pack(side = RIGHT, padx = 2, pady = 2)
		self.canvas.pack(anchor = CENTER, padx = 0, pady = 0)


	def left_but_down(self, event=None):
		self.left_but = "down"
		self.x1_line_pt = event.x
		self.y1_line_pt = event.y

	def left_but_up(self, event=None):
		self.left_but = "up"
		self.x_pos = None
		self.y_pos = None
		self.x2_line_pt = event.x
		self.y2_line_pt = event.y

		self.push_undo()


	def canvas_motion(self, event=None):
		if self.drawing_tool == "pencil":
			self.pencil_draw(event)
		if self.drawing_tool == "eraser":
			self.eraser(event)
		if self.drawing_tool == "move":
			self.move_img(event)

	def pencil_draw(self, event=None):
		if self.left_but == "down":
			if self.x_pos is not None and self.y_pos is not None:
				l = event.widget.create_line(self.x_pos, self.y_pos, event.x, event.y, smooth=True, fill = self.fill_color, width = self.line_width)
				self.canvas.itemconfig(l, tags = (self.curr_layer, "line"))
				self.correct_layering()

			self.x_pos, self.y_pos = event.x, event.y

	def eraser(self, event=None):
		if self.left_but == "down":
			if self.x_pos is not None and self.y_pos is not None:
				l = event.widget.create_line(self.x_pos, self.y_pos, event.x, event.y, smooth=True, fill = "#FFFFFF", width = 12)
				self.canvas.itemconfig(l, tags = (self.curr_layer, "line"))
				self.correct_layering()

			self.x_pos, self.y_pos = event.x, event.y

	def move_img(self, event=None):
		if self.left_but == "down":
			if self.x_pos is not None and self.y_pos is not None:
				#self.canvas.delete("all")=
				for i in self.canvas.find_withtag(self.curr_layer):
					self.canvas.move(i, event.x - self.x_pos, event.y - self.y_pos)

			self.x_pos, self.y_pos = event.x, event.y

	def add_new_layer(self):
		self.no_of_layers += 1
		self.layer_list.append("Layer_" + str(self.no_of_layers))
		self.layer_dropdown.config(values = self.layer_list)
		self.curr_layer = "Layer_" + str(self.no_of_layers)
		self.layer_image_dict[self.curr_layer] = []
		self.layer_dropdown.current(self.no_of_layers - 1)
		#self.push_undo()

	def delete_current_layer(self):
		curr_state = self.get_canvas_state()

		for i in self.canvas.find_withtag(self.curr_layer):					#Get rid of all the items associated with this layer
			self.canvas.delete(i)

		if self.no_of_layers == 1:											#Can't delete the only layer now can we?
			return

		self.undo_stack.append(curr_state)

		self.no_of_layers -= 1

		pos = self.layer_list.index(self.curr_layer)						#Shifting down all the top layers
		for i in range(pos + 1, len(self.layer_list)):
			for j in self.canvas.find_withtag(self.layer_list[i]):
				self.canvas.itemconfig(j, tags = (self.layer_list[i-1], "image"))
			self.layer_image_dict[self.layer_list[i-1]] = self.layer_image_dict[self.layer_list[i]]

		del self.layer_image_dict[self.layer_list[-1]]						#Getting rid of the repeated last entry after shifting
		self.layer_list.pop(-1)

		self.layer_dropdown.config(values = self.layer_list)				#Reconfiguring the layer dropdown
		self.layer_dropdown.current(self.no_of_layers - 1)
		self.curr_layer = self.layer_list[-1]

		root.update_idletasks()

	def raise_lower_layer(self, operation = "raise"):
		pos = self.layer_list.index(self.curr_layer)

		if operation == "raise":
			if pos == self.no_of_layers - 1:
				return
			pos += 1
		else:
			pos = self.layer_list.index(self.curr_layer)
			if pos == 0:
				return
			pos -= 1

		layer_i = self.canvas.find_withtag(self.curr_layer)
		layer_j = self.canvas.find_withtag(self.layer_list[pos])

		for i in layer_i:
			self.canvas.itemconfig(i, tags = self.layer_list[pos])
		for j in layer_j:
			self.canvas.itemconfig(j, tags = self.curr_layer)

		self.layer_image_dict[self.curr_layer], self.layer_image_dict[self.layer_list[pos]] = self.layer_image_dict[self.layer_list[pos]], self.layer_image_dict[self.curr_layer]
		self.correct_layering()

	def merge_layers(self):
		w = mergePopup(self.root, self.layer_list)
		self.root.wait_window(w.top)

		idx_1, idx_2 = self.layer_list.index(w.first), self.layer_list.index(w.second)
		if idx_1 == idx_2:
			return

		pos = max(idx_1, idx_2)
		mrg = min(idx_1, idx_2)

		for i in self.canvas.find_withtag(self.layer_list[pos]):
			self.canvas.itemconfig(i, tags = self.layer_list[mrg])

		self.layer_image_dict[self.layer_list[mrg]] += self.layer_image_dict[self.layer_list[pos]]

		#Same as delete current layer
		for i in range(pos + 1, len(self.layer_list)):
			for j in self.canvas.find_withtag(self.layer_list[i]):
				self.canvas.itemconfig(j, tags = (self.layer_list[i-1], "image"))
			self.layer_image_dict[self.layer_list[i-1]] = self.layer_image_dict[self.layer_list[i]]

		del self.layer_image_dict[self.layer_list[-1]]						#Getting rid of the repeated last entry after shifting
		self.layer_list.pop(-1)

		self.no_of_layers -= 1
		self.layer_dropdown.config(values = self.layer_list)				#Reconfiguring the layer dropdown
		self.layer_dropdown.current(self.no_of_layers - 1)
		self.curr_layer = self.layer_list[-1]

		root.update_idletasks()

		return


	def correct_layering(self):												#Maintaining the layer viewing heirarchy
		for i in self.layer_list:
			for j in self.canvas.find_withtag(i):
				self.canvas.tag_raise(j)

	def hide_other_layers(self):
		hideables = []
		for j in self.layer_list:
			if j == self.curr_layer:
				continue
			hideables += self.canvas.find_withtag(j)
		for i in hideables:
			self.canvas.itemconfigure(i, state = "hidden")
		return hideables

	def show_hidden_layers(self, hideables):
		for i in hideables:
			self.canvas.itemconfigure(i, state = "normal")

	def canvas_to_cv(self):
		img = self.canvas_to_pil()
		img.save('temp.png', 'png')

		coords = self.canvas_width - self.canvas_width%16, self.canvas_height - self.canvas_height%16
		img = cv2.imread('temp.png')
		img = img[:coords[0], :coords[1]]
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		return img

	def canvas_to_pil(self):
		self.canvas.postscript(file = 'temp.eps', pagewidth = self.canvas_width + 3, pageheight = self.canvas_height + 3)
		img = Image.open('temp.eps')
		return img

	def get_canvas_state(self):
		#All items as an image, later list and dict, curr layer
		hideables = self.hide_other_layers()
		img = self.canvas_to_pil().convert("RGB")
		img = ImageTk.PhotoImage(img)
		self.show_hidden_layers(hideables)

		layer_list = self.layer_list.copy()
		layer_image_dict = self.layer_image_dict.copy()
		curr_layer = self.curr_layer
		return img, layer_list, layer_image_dict, curr_layer


	#######################################################################################################################
	#GAN RELATED FUNCTION CALLS
	#######################################################################################################################
	def generate(self, gaugan = False):
		img = None
		if gaugan == True:
			hideables = self.hide_other_layers()
			img = self.canvas_to_cv()
			self.show_hidden_layers(hideables)

		w = generatePopup(self.root, self.model, img)
		self.root.wait_window(w.top)

		img_t = w.img_t3
		if img_t is None or not w.apply:
			return

		self.add_new_layer()
		#self.undo_stack.pop(-1)

		i = self.canvas.create_image(0, 0, anchor=NW, image = img_t)
		self.canvas.itemconfig(i, tags = ("Layer_" + str(self.no_of_layers), "image"))
		self.layer_image_dict["Layer_" + str(self.no_of_layers)] = [img_t]

		self.push_undo()
		root.update_idletasks()

	def translate(self, operation = "style"):
		hideables = self.hide_other_layers()
		img = self.canvas_to_cv()
		self.show_hidden_layers(hideables)
		style_value = None

		if operation == "style":
			w = stylePopup(self.root, self.model)
			self.root.wait_window(w.top)
			if not w.apply:
				return

		mask = (img == 255).all(axis=2)

		for i in self.canvas.find_withtag(self.curr_layer):
			#Remove from layer_image_dict
			self.layer_image_dict[self.curr_layer] = []
			self.canvas.delete(i)

		gen_img = self.model.translate(img, mask, operation)
		img_t = ImageTk.PhotoImage(gen_img)

		if operation == "sr":
			self.canvas_height, self.canvas_width = min(800, 4*self.canvas_height), min(800, 4*self.canvas_width)
			self.canvas.config(height = self.canvas_height, width = self.canvas_width)
			root.geometry(str(self.canvas_width + 300) + "x" + str(self.canvas_height + 100))

		r = self.canvas.create_image(0, 0, anchor=NW, image = img_t)
		self.canvas.itemconfig(r, tags = (self.curr_layer, "image"))
		self.layer_image_dict[self.curr_layer].append(img_t)

		self.correct_layering()
		self.push_undo()
		root.update_idletasks()


##########################################################################################################

def stack_checker(App):
	while True:
		time.sleep(2)
		App.lock.acquire()

		if len(App.undo_stack) > 10:
			print("Undo states limit")
			App.undo_stack.pop(0)

		if len(App.redo_queue) > 10:
			print("Redo states limit")
			App.redo_queue.pop(-1)

		App.lock.release()

if __name__ == '__main__':
	root = Tk()

	root.geometry("780x512")
	root['bg'] = '#111114'
	stack_lock = threading.Lock()

	GANApp = GANGui(root, stack_lock)
	time.sleep(1)
	t = threading.Thread(target= lambda : stack_checker(GANApp))
	t.daemon = True
	t.start()
	root.mainloop()

	sys.exit()