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
from models.cycgan import UNet_Generator
from models.srgan import SRGenerator
from models.gaugan import SPADE_Generator


class GANModels:
	def __init__(self):
		self.model = Basic_Generator(image_size = (128, 128), specnorm = True)
		self.s_model = UNet_Generator()
		self.gau_model = SPADE_Generator(image_size = (128, 128), seg_channels = 1, specnorm = True)
		self.sr_model = SRGenerator(specnorm = True)

		self.gen_models_list = ["Arbitrary"]
		self.style_models_list = ["Arbitrary"]
		self.gaugan_models_list = ["Arbitrary"]
		self.load_models()

	def load_models(self):
		self.sr_model.load_state_dict(torch.load("./weights/SR_G128.pth",
									map_location = "cpu"))

	def load_style_model(self, style_value):
		pass

	def generate(self, noise = None):
		if noise is None:
			noise = torch.randn(1, 128)
		with torch.no_grad():
			gen_img = self.model(noise)

		gen_img = np.uint8((gen_img+1)*255/2)
		trans = transforms.ToPILImage(mode="RGB")
		gen_img = trans(gen_img[0].transpose(1, 2, 0))
		return gen_img, noise

	def translate(self, img, mask, operation = "style"):
		img = transforms.ToTensor()(img).unsqueeze(0).float()
		img = img*2 - 1
		with torch.no_grad():
			if operation == "style":
				gen_img = self.s_model(img)
				gen_img = np.uint8((gen_img[0]+1)*255/2).transpose(1, 2, 0)

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
			noise = torch.randn(1, 128)
		with torch.no_grad():
			gen_img = self.gau_model(noise, img)

		gen_img = np.uint8((gen_img+1)*255/2)
		trans = transforms.ToPILImage(mode="RGB")
		gen_img = trans(gen_img[0].transpose(1, 2, 0))
		return gen_img, noise
