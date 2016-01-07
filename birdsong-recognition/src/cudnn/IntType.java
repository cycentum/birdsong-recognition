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
package cudnn;

/**
 * Types of integers. Used as the type of labels. When the number of labels are more than {@link Byte#MAX_VALUE}, {@link IntType#INT} must be used.
 * The type must be consistent throughout the computation.
 * @author koumura
 *
 */
public enum IntType
{
	BYTE(Byte.BYTES), INT(Integer.BYTES);
	
	private int bytes;
	
	private IntType(int bytes) {
		this.bytes= bytes;
	}
	
	public int getBytes() {
		return bytes;
	}
}
