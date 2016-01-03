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
