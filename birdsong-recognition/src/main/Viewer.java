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
package main;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GraphicsEnvironment;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.stream.Collectors;

import javax.imageio.ImageIO;
import javax.sound.sampled.UnsupportedAudioFileException;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JScrollPane;

import computation.STFTParam;
import computation.Sequence;
import computation.Sequence.Note;
import no.uib.cipr.matrix.NotConvergedException;
import utils.SoundUtils;

public class Viewer
{
	public static void main(String... arg) throws IOException, UnsupportedAudioFileException, NotConvergedException
	{
		/**********************************************************
		 * Files.
		 * Change them according to your environment.
		 *********************************************************/
		//Data.
		Path dirWave=Paths.get("I:\\koumura\\MultiDays2\\LabelCrossValidation\\Wave\\B-W-20150112");
		Path fileSequences=Paths.get("I:\\koumura\\MultiDays2\\BirdsongRecognition\\Data\\Bird0\\AllSequences.xml");
		
		//Image outputs.
		Path dirImageOutput=Paths.get("I:\\koumura\\MultiDays2\\BirdsongRecognition\\Image");
		
		
		/**********************************************************
		 * Parameters.
		 *********************************************************/
		//Spectrogram.
		STFTParam stftParam=new STFTParam(512, 32);
		double frequencyStartHz=1000, frequencyEndHz=8000;
		int dpssParam=4;

		//color scale
		float blackValue=3.0f, whiteValue=-1.5f;
		
		//label
		int barHeight=5;
		int fontSize=20;
		int bottomMargin=30;
		Color labelColor=Color.red;
		
		
		
		
		/**********************************************************
		 * Data preparation.
		 *********************************************************/
		ArrayList<Sequence> sequence=Sequence.readXml(fileSequences);
		int samplingRate=(int)SoundUtils.checkSamplingRate(sequence.stream().map(s->s.getWaveFileName()).collect(Collectors.toList()), dirWave);
		int freqOffset=(int)(frequencyStartHz/stftParam.unitFrequency(samplingRate));
		int freqLength=(int)(frequencyEndHz/stftParam.unitFrequency(samplingRate))-freqOffset;
		HashMap<Sequence, float[]> spectrogram=SoundUtils.spectrogram(Sequence.wavePositionMap(sequence, dirWave), stftParam, dpssParam, freqOffset, freqLength);
		double[] specMeanSd=SoundUtils.spectrogramMeanSd(sequence.stream().map(s->spectrogram.get(s)).collect(Collectors.toList()));
		SoundUtils.whiteSpectrogram(spectrogram.values(), specMeanSd[0], specMeanSd[1]);
		
		
		/**********************************************************
		 * Showing images.
		 *********************************************************/
		for(Sequence seq: sequence)
		{
			BufferedImage image=SoundUtils.spectrogramImage(spectrogram.get(seq), blackValue, whiteValue, freqLength, bottomMargin);
			drawLabel(image, seq.getNote(), labelColor, fontSize, barHeight, stftParam, freqLength);
			JFrame frame=imageFrame(image);
			frame.setVisible(true);
			break;
		}
		
		/**********************************************************
		 * Saving images.
		 *********************************************************/
		Files.createDirectories(dirImageOutput);
		for(Sequence seq: sequence)
		{
			BufferedImage image=SoundUtils.spectrogramImage(spectrogram.get(seq), blackValue, whiteValue, freqLength, bottomMargin);
			drawLabel(image, seq.getNote(), labelColor, fontSize, barHeight, stftParam, freqLength);
			Path file=dirImageOutput.resolve(seq.getWaveFileName()+"-"+seq.getPosition()+"-"+seq.getLength()+".png");
			ImageIO.write(image, "PNG", file.toFile());
			break;
		}
	}
	
	private static void drawLabel(BufferedImage image, ArrayList<Note> note, Color labelColor, int fontSize, int barHeight, STFTParam stftParam, int freqLength)
	{
		Graphics2D graphics=(Graphics2D)image.getGraphics();
		graphics.setPaint(labelColor);
		graphics.setFont(new Font(Font.MONOSPACED, Font.BOLD, fontSize));
		for(Note n: note)
		{
			int pos=stftParam.spectrogramPosition(n.getPosition());
			int end=stftParam.spectrogramPosition(n.end());
			graphics.fillRect(pos, freqLength, end-pos, barHeight);
			graphics.drawString(n.getLabel(), pos, image.getHeight());
		}	
	}
	
	private static JFrame imageFrame(BufferedImage image)
	{
		ImagePanel panel=new ImagePanel(image);
		JScrollPane scroll=new JScrollPane(panel);
		JFrame frame=new JFrame();
		frame.add(scroll);
		int desktopWidth=GraphicsEnvironment.getLocalGraphicsEnvironment().getMaximumWindowBounds().width;
		frame.setSize(desktopWidth, image.getHeight()+80);
		return frame;
	}
	
	private static class ImagePanel extends JPanel
	{
		private BufferedImage image;
		
		public ImagePanel(BufferedImage image)
		{
			this.image = image;
			Dimension size=new Dimension(image.getWidth(), image.getHeight());
			setPreferredSize(size);
			setMinimumSize(size);
		}

		@Override
		public void paint(Graphics g)
		{
			if(image!=null) g.drawImage(image, 0, 0, this);
		}
	}
}
