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
package cudnn.learner;

import java.util.ArrayList;

import cudnn.Cuda;
import cudnn.CudaDriver;
import cudnn.CudaException;
import cudnn.Cudnn;
import cudnn.CudnnException;
import cudnn.FloatType;
import cudnn.Pointer;

public class Adam
{
	private Model model;
	private double alpha, beta1, beta2, epsilon, beta1T, beta2T;
	private ArrayList<Pointer> moment, moment2;
	private Pointer hyperParam;
	private CudaDriver driver;
	private Cudnn cudnn;
	private FloatType floatType;
	
	public Adam(Model model, FloatType floatType, double alpha, double beta1, double beta2, double epsilon, Cudnn cudnn, CudaDriver driver)
	{
		this.model = model;
		this.alpha = alpha;
		this.beta1 = beta1;
		this.beta2 = beta2;
		this.epsilon=epsilon;
		this.cudnn=cudnn;
		this.driver=driver;
		this.floatType=floatType;
	}
	public Adam(Model model, FloatType floatType, Cudnn cudnn, CudaDriver driver)
	{
		this(model, floatType, 0.001, 0.9, 0.999, 1e-8, cudnn, driver);
	}
	
	public void init() throws CudaException
	{
		int blockWidth=128;
		jcuda.Pointer p0;
		if(floatType==FloatType.SINGLE) p0=jcuda.Pointer.to(new float[]{0});
		else p0=jcuda.Pointer.to(new double[]{0});
		
		moment=new ArrayList<Pointer>(model.getParamSize().size());
		moment2=new ArrayList<Pointer>(model.getParamSize().size());
		for(int p: model.getParamSize())
		{
			moment.add(Cuda.malloc(p*floatType.getBytes()));
			moment2.add(Cuda.malloc(p*floatType.getBytes()));
			
			
			driver.call("Fill"+(floatType==FloatType.SINGLE?"Float":"Double"), (int)Math.ceil((double)p/blockWidth), 1, 1, blockWidth, 1, 1, 0, jcuda.Pointer.to(
					jcuda.Pointer.to(moment.get(moment.size()-1)),
					p0,
					jcuda.Pointer.to(new int[]{p})
					));
			driver.call("Fill"+(floatType==FloatType.SINGLE?"Float":"Double"), (int)Math.ceil((double)p/blockWidth), 1, 1, blockWidth, 1, 1, 0, jcuda.Pointer.to(
					jcuda.Pointer.to(moment2.get(moment2.size()-1)),
					p0,
					jcuda.Pointer.to(new int[]{p})
					));
		}
		
		hyperParam=Cuda.malloc(6*floatType.getBytes());
		if(floatType==FloatType.SINGLE)
		{
			Cuda.memcpyAsyncHostToDevice(hyperParam, new float[]{(float)alpha, (float)beta1, (float)beta2, (float)epsilon, 1, 1});
		}
		else
		{
			Cuda.memcpyAsyncHostToDevice(hyperParam, new double[]{alpha, beta1, beta2, epsilon, 1, 1});
		}
		beta1T=1;
		beta2T=1;
	}
	
	public void destroy() throws CudaException
	{
		for(Pointer p: moment) if(p!=null) p.free();
		for(Pointer p: moment2) if(p!=null) p.free();
		if(hyperParam!=null) hyperParam.free();
	}
	
	public void iteration() throws CudaException, CudnnException
	{
		model.compGradient(driver, cudnn);
		if(beta1T>0||beta2T>0)
		{
			beta1T*=beta1;
			beta2T*=beta2;
			if(floatType==FloatType.SINGLE)
			{
				Cuda.memcpyAsyncHostToDevice(hyperParam.withByteOffset(floatType.getBytes()*4), 
					new float[]{(float)beta1T, (float)beta2T});
			}
			else
			{
				Cuda.memcpyAsyncHostToDevice(hyperParam.withByteOffset(floatType.getBytes()*4), 
					new double[]{beta1T, beta2T});
			}
		}
	
		int blockWidth=128;
		for(int la=0; la<model.getParamDev().size(); ++la)
		{
			int size=model.getParamSize().get(la);
			driver.call("Adam"+(floatType==FloatType.SINGLE?"Float":"Double"), (int)Math.ceil((double)size/blockWidth), 1, 1, blockWidth, 1, 1, 0, jcuda.Pointer.to(
					jcuda.Pointer.to(model.getParamDev().get(la)),
					jcuda.Pointer.to(model.getGradDev().get(la)),
					jcuda.Pointer.to(moment.get(la)),
					jcuda.Pointer.to(moment2.get(la)),
					jcuda.Pointer.to(hyperParam),
					jcuda.Pointer.to(new int[]{size})
					));
		}
	}
}
