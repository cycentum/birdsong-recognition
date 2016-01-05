/*
 * Copyright (C) 2016 Takuya KOUMURA
 * https://github.com/takuya-koumura/birdsong-recognition
 *
 * This file is part of birdsong-recognition.
 * 
 * Birdsong-recognition is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Birdsong-recognition is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with birdsong-recognition.  If not, see <http://www.gnu.org/licenses/>.
 */
package cudnn.layer;

import cudnn.Cuda;
import cudnn.CudaException;
import cudnn.FloatType;
import cudnn.Pointer;

public interface ParamLayer extends Layer
{
	float[] getWeightF();
	float[] getBiasF();
	double[] getWeightD();
	double[] getBiasD();
	void setWeightF(float[] weight);
	void setBiasF(float[] bias);
	void setWeightD(double[] weight);
	void setBiasD(double[] bias);
	Pointer getWeightDev();
	Pointer getBiasDev();
	Pointer getGradWeightDev();
	Pointer getGradBiasDev();
	int getWeightSize();
	int getBiasSize();
	
	default void copyParamToDev(FloatType floatType) throws CudaException
	{
		if(floatType==FloatType.SINGLE)
		{
			Cuda.memcpyAsyncHostToDevice(getWeightDev(), getWeightF());
			Cuda.memcpyAsyncHostToDevice(getBiasDev(), getBiasF());
		}
		else
		{
			Cuda.memcpyAsyncHostToDevice(getWeightDev(), getWeightD());
			Cuda.memcpyAsyncHostToDevice(getBiasDev(), getBiasD());
		}
	}
	
	default void copyParamFromDev(FloatType floatType) throws CudaException
	{
		if(floatType==FloatType.SINGLE)
		{
			Cuda.memcpyAsyncDeviceToHost(Pointer.to(getWeightF()), getWeightDev(), floatType.getBytes()*getWeightSize());
			Cuda.memcpyAsyncDeviceToHost(Pointer.to(getBiasF()), getBiasDev(), floatType.getBytes()*getBiasSize());
		}
		else
		{
			Cuda.memcpyAsyncDeviceToHost(Pointer.to(getWeightD()), getWeightDev(), floatType.getBytes()*getWeightSize());
			Cuda.memcpyAsyncDeviceToHost(Pointer.to(getBiasD()), getBiasDev(), floatType.getBytes()*getBiasSize());
		}
	}
	
	default void newWeightBias(FloatType floatType)
	{
		if(floatType==FloatType.SINGLE)
		{
			setWeightF(new float[getWeightSize()]);
			setBiasF(new float[getBiasSize()]);
		}
		else
		{
			setWeightD(new double[getWeightSize()]);
			setBiasD(new double[getBiasSize()]);
		}
		
	}
}
