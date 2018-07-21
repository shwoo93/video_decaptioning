import torch
from torch import nn
from torch.nn import functional as F
import pdb
import matplotlib.pyplot as plt
import numpy as np

class NonLocalBlock3D(nn.Module):
	def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian',fig_idx=1):
		super(NonLocalBlock, self).__init__()
		assert mode in ['embedded_gaussian', 'dot_product', 'euclidean']
		self.fig_idx=fig_idx
		dropout_rate = 0.5
		self.mode = mode

		self.in_channels = in_channels
		self.inter_channels = inter_channels

		if self.inter_channels is None:
			self.inter_channels = in_channels // 2
			if self.inter_channels == 0:
				self.inter_channels = 1

		self.g = nn.Sequential( nn.Linear(self.in_channels, self.inter_channels), nn.Dropout(dropout_rate))
		self.W = nn.Linear(self.inter_channels, self.in_channels)
		nn.init.constant(self.W.weight, 0)
		nn.init.constant(self.W.bias, 0)

		self.theta = None
		self.phi = None

		if mode in ['embedded_gaussian', 'dot_product', 'euclidean']:
			self.theta = nn.Sequential( nn.Linear(self.in_channels, self.inter_channels), nn.Dropout(dropout_rate))
			self.phi = nn.Sequential( nn.Linear(self.in_channels, self.inter_channels), nn.Dropout(dropout_rate))

			if mode == 'embedded_gaussian':
				self.operation_function = self._embedded_gaussian
			elif mode == 'dot_product':
				self.operation_function = self._dot_product
			else:
				self.operation_function = self._euclidean

	def forward(self, x):
		output = self.operation_function(x)
		return output

	def _embedded_gaussian(self, x):

		g_x = self.g(x)

		theta_x = self.theta(x)
		phi_x = self.phi(x).permute(1,0)
		f = torch.matmul(theta_x, phi_x)
		f_div_C = F.softmax(f)

		y = torch.matmul(f_div_C, g_x)
		W_y = self.W(y)
		z = W_y + x

		return z


	def _dot_product(self, x):
		g_x = self.g(x)
		theta_x = self.theta(x)
		phi_x = self.phi(x).permute(1,0)
		f = torch.matmul(theta_x, phi_x)
		f_div_C = F.sigmoid(f)

		y = torch.matmul(f_div_C, g_x)
		W_y = self.W(y)
		z = W_y + x

		return z


	def _euclidean(self, x):
		g_x = self.g(x)
		theta_x = self.theta(x)
		phi_x = self.phi(x)

		n = theta_x.size(0)
		m = phi_x.size(0)
		d = theta_x.size(1)

		theta_x = theta_x.unsqueeze(1).expand(n, m, d)
		phi_x = phi_x.unsqueeze(0).expand(n, m, d)

		dist = torch.pow(theta_x - phi_x, 2).sum(2)
		f_div_C = F.softmax(dist)

		y = torch.matmul(f_div_C, g_x)
		W_y = self.W(y)
		z = W_y + x

		# import pdb
		# pdb.set_trace()

		return z