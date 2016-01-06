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

public class DataLayer implements Layer
{
	private int numChannel, height, width;
	private CudnnLibrary.cudnnTensorDescriptor_t tensorDesc;
	private Pointer dataDev;
	
	public DataLayer(int numChannel, int height, int width)
	{
		this.numChannel = numChannel;
		this.height = height;
		this.width = width;
	}

	public void setDataDev(Pointer dataDev)
	{
		this.dataDev=dataDev;
	}
	
	public void destroy(Cudnn cudnn) throws CudnnException, CudaException
	{
		tensorDesc.destroy(cudnn);
	}
	
	public void init(FloatType floatType, Cudnn cudnn, int batchSize) throws CudnnException
	{
		tensorDesc=cudnn.createTensorDescriptor();
		cudnn.initTensorDesc(floatType, this, batchSize);
	}

	public int getNumChannel() {
		return numChannel;
	}

	public int getHeight() {
		return height;
	}

	public int getWidth() {
		return width;
	}

	public CudnnLibrary.cudnnTensorDescriptor_t getTensorDesc() {
		return tensorDesc;
	}
	
	@Override
	public void cudaMalloc(FloatType floatType, int batchSize) throws CudaException{}
	
	public Pointer getValueDev() {
		return dataDev;
	}
}
