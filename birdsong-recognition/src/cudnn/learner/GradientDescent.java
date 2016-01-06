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
package cudnn.learner;

import cudnn.CudaDriver;
import cudnn.CudaException;
import cudnn.Cudnn;
import cudnn.CudnnException;
import jcuda.jcublas.JCublas;

public class GradientDescent
{
	private Model model;
	
	public GradientDescent(Model model)
	{
		this.model = model;
	}

	public void iteration(double learningRate, CudaDriver driver, Cudnn cudnn) throws CudaException, CudnnException
	{
		model.compGradient(driver, cudnn);
		updateParam(learningRate);
	}
	public void iteration(float learningRate, CudaDriver driver, Cudnn cudnn) throws CudaException, CudnnException
	{
		model.compGradient(driver, cudnn);
		updateParam(learningRate);
	}
	
	public void updateParam(float learning_rate)
	{	
		float alpha = -learning_rate;
		for(int i=0; i<model.getParamSize().size(); ++i)
		{
			JCublas.cublasSaxpy(model.getParamSize().get(i), alpha, model.getGradDev().get(i), 1, model.getParamDev().get(i), 1);
		}
	}
	public void updateParam(double learning_rate)
	{	
		double alpha = -learning_rate;
		for(int i=0; i<model.getParamSize().size(); ++i)
		{
			JCublas.cublasDaxpy(model.getParamSize().get(i), alpha, model.getGradDev().get(i), 1, model.getParamDev().get(i), 1);
		}
	}


}
