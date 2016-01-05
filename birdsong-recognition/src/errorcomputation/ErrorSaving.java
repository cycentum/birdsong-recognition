package errorcomputation;

import java.io.IOException;
import java.nio.file.Path;
 * This file is part of birdsong-recognition.
 * 
 * Birdsong-recognition is free software: you can redistribute it and/or modify
import org.w3c.dom.Element;

import utils.XmlUtils;

 * Birdsong-recognition is distributed in the hope that it will be useful,
{
	public static void writeXml(double levenshteinError, double matchingError, Path file) throws IOException
	{
		Element rootEl=XmlUtils.rootElement("Errors");
		XmlUtils.addChild(rootEl, "LevenshteinError", levenshteinError);
 * along with birdsong-recognition.  If not, see <http://www.gnu.org/licenses/>.
		XmlUtils.write(rootEl.getOwnerDocument(), file);
	}
}
