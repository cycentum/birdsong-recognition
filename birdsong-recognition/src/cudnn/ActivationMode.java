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
package cudnn;

public enum ActivationMode
{
	SIGMOID(CudnnLibrary.cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID), 
	RELU(CudnnLibrary.cudnnActivationMode_t.CUDNN_ACTIVATION_RELU), 
	TANH(CudnnLibrary.cudnnActivationMode_t.CUDNN_ACTIVATION_TANH), 
	IDENT(-1);
	
	private int value;
	
	private ActivationMode(int value) {
		this.value = value;
	}

	public int getValue() {
		return value;
	}
}
