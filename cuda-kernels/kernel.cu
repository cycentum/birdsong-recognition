/*
 * Copyright (C) 2016 Takuya KOUMURA
 * https://github.com/takuya-koumura/birdsong-recognition
 *
 * This file is part of Birdsong Recognition.
 * 
 * Birdsong Recognition is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Birdsong Recognition is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Birdsong Recognition.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <device_launch_parameters.h>
#include <cmath>

extern "C" __global__ void SeqSoftmaxConvBackwardDoubleChar(const char *label, double *diff, const int *size, int labelShiftY, int labelShiftX)
{
	int batchSize = size[0];
	int numChannel= size[1];
	int height = size[2];
	int width = size[3];
	int singleSize= size[4];
	int singleHeight= size[5];
	int singleWidth= size[6];
	int labelHeight = size[7];
	int labelWidth = size[8];
	int labelShiftUpperH = size[9];
	int labelShiftUpperW = size[10];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= batchSize*height*width*singleHeight*singleWidth) return;

	int x = idx%width;
	int i = (idx - x) / width;
	int y = i%height;
	i = (idx - y) / height;
	int sx = i%singleWidth;
	i = (i - sx) / singleWidth;
	int sy = i%singleHeight;
	int b = i / singleHeight;

	char la = label[b*labelWidth*labelHeight + (y*labelShiftUpperH + labelShiftY + sy)*labelWidth + x*labelShiftUpperW + labelShiftX + sx];

	if (la < 0)
	{
		for (int li = 0; li < singleSize; ++li)
		{
			int index = ((b*numChannel + (li*singleHeight + sy)*singleWidth + sx)*height + y)*width + x;
			diff[index] = 0;
		}
		return;
	}
	int index = ((b*numChannel + (la*singleHeight + sy)*singleWidth + sx)*height + y)*width + x;
	diff[index] -= 1;
}

extern "C" __global__ void SeqSoftmaxConvBackwardFloatChar(const char *label, float *diff, const int *size, int labelShiftY, int labelShiftX)
{
	int batchSize = size[0];
	int numChannel = size[1];
	int height = size[2];
	int width = size[3];
	int singleSize = size[4];
	int singleHeight = size[5];
	int singleWidth = size[6];
	int labelHeight = size[7];
	int labelWidth = size[8];
	int labelShiftUpperH = size[9];
	int labelShiftUpperW = size[10];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= batchSize*height*width*singleHeight*singleWidth) return;

	int x = idx%width;
	int i = (idx - x) / width;
	int y = i%height;
	i = (idx - y) / height;
	int sx = i%singleWidth;
	i = (i - sx) / singleWidth;
	int sy = i%singleHeight;
	int b = i / singleHeight;

	char la = label[b*labelWidth*labelHeight + (y*labelShiftUpperH + labelShiftY + sy)*labelWidth + x*labelShiftUpperW + labelShiftX + sx];

	if (la < 0)
	{
		for (int li = 0; li < singleSize; ++li)
		{
			int index = ((b*numChannel + (li*singleHeight + sy)*singleWidth + sx)*height + y)*width + x;
			diff[index] = 0;
		}
		return;
	}
	int index = ((b*numChannel + (la*singleHeight + sy)*singleWidth + sx)*height + y)*width + x;
	diff[index] -= 1;
}


extern "C" __global__ void SeqSoftmaxConvErrorDoubleChar(double *error, const double *output, const char *label, const int *size, int labelShiftY, int labelShiftX, double outputLowerForError)
{
	int batchSize = size[0];
	int numChannel = size[1];
	int height = size[2];
	int width = size[3];
	int singleSize = size[4];
	int singleHeight = size[5];
	int singleWidth = size[6];
	int labelHeight = size[7];
	int labelWidth = size[8];
	int labelShiftUpperH = size[9];
	int labelShiftUpperW = size[10];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= batchSize*height*width*singleHeight*singleWidth) return;

	int x = idx%width;
	int i = (idx - x) / width;
	int y = i%height;
	i = (idx - y) / height;
	int sx = i%singleWidth;
	i = (i - sx) / singleWidth;
	int sy = i%singleHeight;
	int b = i / singleHeight;

	int labelIndex = b*labelWidth*labelHeight + (y*labelShiftUpperH + labelShiftY + sy)*labelWidth + x*labelShiftUpperW + labelShiftX + sx;
	char la = label[labelIndex];

	if (la < 0)
	{
		error[idx] = 0;
		return;
	}

	int index = ((b*numChannel + (la*singleHeight + sy)*singleWidth + sx)*height + y)*width + x;
	double o = output[index];
	if (isnan(o)) error[idx] = o;
	else
	{
		if (o < outputLowerForError) o = outputLowerForError;
		error[idx] = -log(o);
	}
}


extern "C" __global__ void SeqSoftmaxConvErrorFloatChar(float *error, const float *output, const char *label, const int *size, int labelShiftY, int labelShiftX, float outputLowerForError)
{
	int batchSize = size[0];
	int numChannel = size[1];
	int height = size[2];
	int width = size[3];
	int singleSize = size[4];
	int singleHeight = size[5];
	int singleWidth = size[6];
	int labelHeight = size[7];
	int labelWidth = size[8];
	int labelShiftUpperH = size[9];
	int labelShiftUpperW = size[10];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= batchSize*height*width*singleHeight*singleWidth) return;

	int x = idx%width;
	int i = (idx - x) / width;
	int y = i%height;
	i = (idx - y) / height;
	int sx = i%singleWidth;
	i = (i - sx) / singleWidth;
	int sy = i%singleHeight;
	int b = i / singleHeight;

	int labelIndex = b*labelWidth*labelHeight + (y*labelShiftUpperH + labelShiftY + sy)*labelWidth + x*labelShiftUpperW + labelShiftX + sx;
	char la = label[labelIndex];

	if (la < 0)
	{
		error[idx] = 0;
		return;
	}

	int index = ((b*numChannel + (la*singleHeight + sy)*singleWidth + sx)*height + y)*width + x;
	float o = output[index];
	if (isnan(o)) error[idx] = o;
	else
	{
		if (o < outputLowerForError) o = outputLowerForError;
		error[idx] = -log(o);
	}
}


extern "C" __global__ void FillFloat(float *vector, float value, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) return;
	
	vector[idx] = value;
}

extern "C" __global__ void FillDouble(double *vector, double value, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) return;

	vector[idx] = value;
}


extern "C" __global__ void AdamFloat(float *param, const float* grad, float* moment, float* moment2, const float* hyperParam, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) return;

	float alpha = hyperParam[0];
	float beta1= hyperParam[1];
	float beta2 = hyperParam[2];
	float epsilon = hyperParam[3];
	float beta1T = hyperParam[4];
	float beta2T = hyperParam[5];

	moment[idx] = beta1*moment[idx] + (1 - beta1)*grad[idx];
	moment2[idx] = beta2*moment2[idx] + (1 - beta2)*grad[idx] * grad[idx];
	float alphaT = alpha*sqrt(1 - beta2T) / (1 - beta1T);
	float delta = alphaT*moment[idx] / (sqrt(moment2[idx]) + epsilon);
	param[idx] -= delta;
}

extern "C" __global__ void AdamDouble(double *param, const double* grad, double* moment, double* moment2, const double* hyperParam, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) return;

	double alpha = hyperParam[0];
	double beta1 = hyperParam[1];
	double beta2 = hyperParam[2];
	double epsilon = hyperParam[3];
	double beta1T = hyperParam[4];
	double beta2T = hyperParam[5];

	moment[idx] = beta1*moment[idx] + (1 - beta1)*grad[idx];
	moment2[idx] = beta2*moment2[idx] + (1 - beta2)*grad[idx] * grad[idx];
	double alphaT = alpha*sqrt(1 - beta2T) / (1 - beta1T);
	double delta = alphaT*moment[idx] / (sqrt(moment2[idx]) + epsilon);
	param[idx] -= delta;
}
