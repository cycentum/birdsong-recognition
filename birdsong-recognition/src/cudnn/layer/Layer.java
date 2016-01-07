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
package cudnn.layer;

import cudnn.CudaException;
import cudnn.Cudnn;
import cudnn.CudnnException;
import cudnn.CudnnLibrary;
import cudnn.FloatType;
import cudnn.Pointer;

/**
 * An interface for a general layer.
 * {@link #init(FloatType, Cudnn, int)} and {@link #cudaMalloc(FloatType, int)} must be called before computation.
 * {@link #destroy(Cudnn)} must be called before termination of the program.
 * **Dev means a pointer in a GPU.
 * @author koumura
 *
 */
public interface Layer
{
	int getNumChannel();
	int getHeight();
	int getWidth();
	/**
	 * @return numChannel * height * width
	 */
	default int sizeCHW(){return getNumChannel()*getHeight()*getWidth();}
	/**
	 * @return height * width
	 */
	default int sizeHW(){return getHeight()*getWidth();}
	
	void cudaMalloc(FloatType floatType, int batchSize) throws CudaException;
	CudnnLibrary.cudnnTensorDescriptor_t getTensorDesc();
	void init(FloatType floatType, Cudnn cudnn, int batchSize) throws CudnnException;
	void destroy(Cudnn cudnn) throws CudnnException, CudaException;
	
	Pointer getValueDev();
}
