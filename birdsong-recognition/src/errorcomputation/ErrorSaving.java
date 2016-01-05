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
package errorcomputation;

import java.io.IOException;
import java.nio.file.Path;

import org.w3c.dom.Element;

import utils.XmlUtils;

public class ErrorSaving
{
	public static void writeXml(double levenshteinError, double matchingError, Path file) throws IOException
	{
		Element rootEl=XmlUtils.rootElement("Errors");
		XmlUtils.addChild(rootEl, "LevenshteinError", levenshteinError);
		XmlUtils.addChild(rootEl, "MatchingError", matchingError);
		XmlUtils.write(rootEl.getOwnerDocument(), file);
	}
}
