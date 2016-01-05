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
package computation;

import utils.MathUtils;

public class STFTParam
{
	private int fftLength, shiftLength;

	public STFTParam(int fftLength, int shiftLength)
	{
		this.fftLength = fftLength;
		this.shiftLength = shiftLength;
	}

	public int getFftLength() {
		return fftLength;
	}

	public int getShiftLength() {
		return shiftLength;
	}
	
	public int spectrogramLength(int waveLength)
	{
		return MathUtils.ceil(waveLength-(fftLength-shiftLength), shiftLength);
	}
	
	public int wavePosition(int spectrogramPosition)
	{
		return spectrogramPosition*shiftLength+fftLength/2;
	}
	
	public int spectrogramPosition(int wavePosition)
	{
		return (wavePosition-fftLength/2)/shiftLength;
	}
	
	public double unitFrequency(int samplingRate)
	{
		return unitFrequency(samplingRate, fftLength);
	}
	
	public static double unitFrequency(int samplingRate, int fftLength)
	{
		return (double)samplingRate/fftLength;
	}
}
