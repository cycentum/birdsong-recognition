/*
 * Copyright (C) 2016 Takuya KOUMURA
 * https://github.com/takuya-koumura/birdsong-recognition
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package cudnn.layer;

import cudnn.CudaException;
import cudnn.Cudnn;
import cudnn.CudnnException;
import cudnn.CudnnLibrary;
import cudnn.FloatType;
import cudnn.Pointer;

public interface Layer
{
	int getNumChannel();
	int getHeight();
	int getWidth();
	default int sizeCHW(){return getNumChannel()*getHeight()*getWidth();}
	default int sizeHW(){return getHeight()*getWidth();}
	
	void cudaMalloc(FloatType floatType, int batchSize) throws CudaException;
	CudnnLibrary.cudnnTensorDescriptor_t getTensorDesc();
	void init(FloatType floatType, Cudnn cudnn, int batchSize) throws CudnnException;
	void destroy(Cudnn cudnn) throws CudnnException, CudaException;
	
	Pointer getValueDev();
}
