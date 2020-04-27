from tkinter import *
from tkinter import ttk
from popups import *
from tkinter import filedialog
from PIL import ImageTk, Image, ImageGrab
import numpy as np
import time
import os
import sys
import io
import cv2
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from models.basic import Basic_Generator
from models.cycgan import UNet_Generator, Star_Generator
from models.srgan import SRGenerator
from models.gaugan import SPADE_Generator
from models.msg import Generator


class GANModels:
	def __init__(self):
		#self.model = Basic_Generator(image_size = (128, 128), specnorm = True)
		self.gan_noise_dim = 512
		self.gau_noise_dim = 128
		self.gan_image_size = (32,32)
		self.gau_image_size = (128,128)

		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		self.model = Generator(image_size = self.gan_image_size, noise_dim = self.gan_noise_dim, specnorm = True).to(device)
		self.s_model = UNet_Generator().to(device)
		self.star_model = Star_Generator(img_size = (128,128), cond_length = 5).to(device)
		self.gau_model = SPADE_Generator(image_size = self.gau_image_size, seg_channels = 1, specnorm = True).to(device)
		self.sr_model = SRGenerator(specnorm = True).to(device)

		self.star_cond = np.zeros(5)
		self.star_cond[0] = 1

		self.gen_models_list = ["CIFAR-10"]
		self.style_models_list = ["Van Gogh", "Landscapes", "Fruits", "Ukiyo-e", "Doodles"]
		self.gaugan_models_list = ["Arbitrary"]
		self.load_models()

	def load_models(self):
		self.model.load_state_dict(torch.load("./weights/MSG_G.pth",
									map_location = "cpu"))
		self.star_model.load_state_dict(torch.load("./weights/stargan_G.pth",
									map_location = "cpu"))
		self.sr_model.load_state_dict(torch.load("./weights/SR_G128.pth",
									map_location = "cpu"))

	def load_style_model(self, style_value):
		encoding = np.zeros(5)
		encoding[self.style_models_list.index(style_value)] = 1
		self.star_cond = encoding

	def generate(self, noise = None):
		if noise is None:
			noise = torch.randn(1, self.gan_noise_dim)
		with torch.no_grad():
			gen_img = self.model(noise)

		gen_img = np.uint8((gen_img[-1]+1)*255/2)
		trans = transforms.ToPILImage(mode="RGB")
		gen_img = trans(gen_img[0].transpose(1, 2, 0)).resize((128,128))
		return gen_img, noise

	def translate(self, img, mask, operation = "style"):
		img = transforms.ToTensor()(img).unsqueeze(0).float()
		img = img*2 - 1
		cond = torch.Tensor([self.star_cond])

		with torch.no_grad():
			if operation == "style":
				gen_img = self.star_model(img, cond)
				gen_img = np.uint8((gen_img[0]+1)*255/2).transpose(1, 2, 0)

				if mask is not None:
					gen_img[mask] = [255, 255, 255]
					mask = gen_img.mean(axis = 2)
					x, y = np.where(mask != 255.0)
					if len(x) and len(y):

						x1, y1, x2, y2 = min(x), min(y), max(x), max(y)
						gen_img = gen_img[x1:x2, y1:y2]

			elif operation == "sr":
				gen_img = self.sr_model(img)
				gen_img = np.uint8((gen_img[0]+1)*255/2).transpose(1, 2, 0)

		print(gen_img.shape)
		trans = transforms.ToPILImage(mode="RGB")
		gen_img = trans(gen_img)
		return gen_img

	def gaugan_translate(self, img, noise = None):
		binary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		binary = cv2.GaussianBlur(binary, (5,5), 0)
		_, binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

		binary = binary.reshape(binary.shape[0], binary.shape[1], 1)
		img = transforms.ToTensor()(binary).unsqueeze(0).float()
		img = img*2 - 1

		if noise is None:
			noise = torch.randn(1, self.gau_noise_dim)
		with torch.no_grad():
			gen_img = self.gau_model(noise, img)

		gen_img = np.uint8((gen_img+1)*255/2)
		trans = transforms.ToPILImage(mode="RGB")
		gen_img = trans(gen_img[0].transpose(1, 2, 0))
		return gen_img, noise
