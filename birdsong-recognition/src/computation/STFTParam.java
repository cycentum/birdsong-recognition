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
package computation;

import utils.MathUtils;

/**
 * This class handles parameters for short time Fourier transformation.
 * @author koumura
 *
 */
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
	
	/**
	 * Converts a length of a wave data into length of the spectrogram.
	 * @param waveLength Length of a wave data.
	 * @return Length of the spectrogram.
	 */
	public int spectrogramLength(int waveLength)
	{
		return MathUtils.ceil(waveLength-(fftLength-shiftLength), shiftLength);
	}
	
	/**
	 * Converts a position in a spectrogram into the position in the wave data.
	 * @param spectrogramPosition Position in a spectrogram.
	 * @return Position in the wave data.
	 */
	public int wavePosition(int spectrogramPosition)
	{
		return spectrogramPosition*shiftLength+fftLength/2;
	}
	
	/**
	 * Converts a position in a wave data into the position in the spectrogram.
	 * @param wavePosition Position in a wave data.
	 * @return Position in the spectrogram.
	 */
	public int spectrogramPosition(int wavePosition)
	{
		return (wavePosition-fftLength/2)/shiftLength;
	}
	
	/**
	 * Computes frequency width (Hz) in a single cell.
	 * @param samplingRate
	 * @return frequency width (Hz).
	 */
	public double unitFrequency(int samplingRate)
	{
		return unitFrequency(samplingRate, fftLength);
	}
	
	/**
	 * Computes frequency width (Hz) in a single cell.
	 * @param samplingRate
	 * @param fftLength
	 * @return frequency width (Hz).
	 */
	public static double unitFrequency(int samplingRate, int fftLength)
	{
		return (double)samplingRate/fftLength;
	}
}
