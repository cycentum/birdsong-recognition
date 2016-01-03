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

import cudnn.CudaDriver;
import cudnn.CudaException;
import cudnn.FloatType;
import cudnn.IntType;
import cudnn.Pointer;

public interface OutputLayer extends Layer
{
	double compError(CudaDriver driver, Pointer labelDev, IntType labelType, FloatType floatType) throws CudaException;
	void initError(FloatType floatType) throws CudaException;
	Pointer getDerActDev();
	void backwardCost(CudaDriver driver, FloatType floatType, Pointer labelDev, int batchSize, int blockWidth, IntType labelType) throws CudaException;
}
