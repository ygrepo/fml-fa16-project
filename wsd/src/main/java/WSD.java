/**
 * Created by yves on 11/26/16.
 */

import com.google.common.base.Joiner;
import com.google.common.collect.Lists;
import de.tudarmstadt.ukp.dkpro.wsd.lesk.algorithm.SimplifiedExtendedLesk;
import de.tudarmstadt.ukp.dkpro.wsd.lesk.algorithm.SimplifiedLesk;
import de.tudarmstadt.ukp.dkpro.wsd.lesk.util.normalization.MostObjects;
import de.tudarmstadt.ukp.dkpro.wsd.lesk.util.overlap.DotProduct;
import de.tudarmstadt.ukp.dkpro.wsd.lesk.util.tokenization.StringSplit;
import de.tudarmstadt.ukp.dkpro.wsd.si.POS;
import de.tudarmstadt.ukp.dkpro.wsd.si.SenseInventoryException;
import de.tudarmstadt.ukp.dkpro.wsd.si.wordnet.WordNetSenseKeySenseInventory;
import net.sf.extjwnl.JWNLException;
import org.apache.commons.cli.*;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.log4j.Logger;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class WSD {
    private static Logger logger = Logger.getLogger(WSD.class);


    enum MODE {STREAM, LINE}

    ;

    private static Joiner JOINER = Joiner.on(" ").skipNulls();
    private static final String STREAMFILE_OPTION = "streamfile";
    private static final String OUTPUT_STREAMFILE_OPTION = "outstreamfile";
    private static final String LINEFILE_OPTION = "linefile";
    private static final String OUTPUT_LINEFILE_OPTION = "outlinefile";

    private final CommandLineParser commandLineParser = new DefaultParser();
    private final Options options = new Options();

    private WordNetSenseKeySenseInventory inventory;
    private SimplifiedLesk lesk;
    private String streamFileName = "";
    private String outputStreamFileName = "";
    private String lineFileName = "";
    private String outputLineFileName = "";
    private MODE mode;

    public WSD(WordNetSenseKeySenseInventory inventory) {
        this.inventory = inventory;
        this.lesk = new SimplifiedExtendedLesk(inventory,
                new DotProduct(), new MostObjects(), new StringSplit(),
                new StringSplit());
        createOptions();
//
//        this.lesk = new SimplifiedLesk(inventory,
//                new SetOverlap(), new NoNormalization(), new StringSplit(),
//                new StringSplit());
    }

    private void createOptions() {
        Option streamfile = Option.builder().longOpt(STREAMFILE_OPTION)
                .hasArg()
                .desc("use given streamfile")
                .build();
        options.addOption(streamfile);

        Option outputStreamfile = Option.builder().longOpt(OUTPUT_STREAMFILE_OPTION)
                .hasArg()
                .desc("generates output streamfile")
                .build();
        options.addOption(outputStreamfile);

        Option linefile = Option.builder().longOpt(LINEFILE_OPTION)
                .hasArg()
                .desc("use given linefile")
                .build();
        options.addOption(linefile);

        Option outlinefile = Option.builder().longOpt(OUTPUT_LINEFILE_OPTION)
                .hasArg()
                .desc("generates output linefile")
                .build();
        options.addOption(outlinefile);

    }


    public String getBestSense(String lemma, String context, POS pos) throws SenseInventoryException, JWNLException {
        Map<String, Double> senseProbmap = lesk.getDisambiguation(lemma, pos, context);
        if (senseProbmap == null || senseProbmap.isEmpty()) return "";
        if (senseProbmap.size() == 1)
            return senseProbmap.keySet().iterator().next();
        double current = 0.0;
        String best = "";
        for (String s : senseProbmap.keySet()) {
            if (senseProbmap.get(s) > current) {
                best = s;
                current = senseProbmap.get(s);
            }
        }
        return best;
    }

    public String readStringFromFile(String filename) {
        String content = "";
        try {
            content = new String(Files.readAllBytes(Paths.get(filename)));
        } catch (Exception e) {
            logger.warn("Exception on read string=" + e.getMessage());
        }
        return content;
    }

    public List<String> getLines(String filename) {
        List<String> lines = Lists.newArrayList();
        try {
            BufferedReader br = new BufferedReader(new FileReader(filename));

            String line = null;
            while ((line = br.readLine()) != null) {
                if (StringUtils.isBlank(line))
                    continue;
                lines.add(line);
            }

            br.close();
        } catch (Exception e) {
            logger.warn("Exception on getlines=" + e.getMessage());
        }
        return lines;
    }

    public void writeSenseStreamToFile(String fileName, List<String> senses) {
        try {
            if (senses.isEmpty()) return;
            String trSt = JOINER.join(senses);
            logger.info("Joined");
            File file = new File(fileName);
            FileUtils.writeStringToFile(file, trSt);
            logger.info("Saved file=" + fileName);
        } catch (Exception e) {
            logger.warn("Exception on saving stream=" + e.getMessage());
        }
    }

    public void writeSenseLinesToFile(String fileName, List<String> senses) {
        try {
            if (senses.isEmpty()) return;
            File file = new File(fileName);
            FileUtils.writeLines(file, senses);
            logger.info("Saved file=" + fileName);
        } catch (Exception e) {
            logger.warn("Exception on saving lines=" + e.getMessage());
        }
    }


    public void generateWordStreamSenses(int winSz) {
        try {
            String content = readStringFromFile(this.streamFileName);
            content = content.trim();
            logger.debug("Content=" + content);
            String[] tuples = content.split("\\s+");
            content = null;
//            String[] newArray = Arrays.copyOfRange(tuples, 0, 21);
//           tuples = newArray;
            int total = tuples.length;
            logger.info("Processing #tuples=" + total);
            int j = 0;
            for (int i = 0; i < total; i++) {
                BatchSenses batchSenses = getTupleSenses(tuples, i, i + total, winSz);
                writeSenseStreamToFile(this.outputStreamFileName, batchSenses.senses);
                i = batchSenses.index;
                j++;
                batchSenses = null;
                logger.info(String.format("Generating batch=%d, i=%d", j, i));

            }

        } catch (Exception e) {
            logger.warn("Exception=" + e.getMessage());
        }
    }

    public void generateLineSenses() {
        try {
            List<String> lines = getLines(this.lineFileName);
            int total = lines.size();
            logger.info("Processing #lines=" + total);
            int i = 0;
            int unkLine = 0;
            List<String> senseList = Lists.newArrayList();
            for (String line : lines) {
                if (line.startsWith(":")) {
                    senseList.add(line);
                    i++;
                    continue;
                }
                String[] tuples = line.split("\\s+");
                String sensesLine = getSense(tuples);
                if (StringUtils.isBlank(sensesLine)) {
                    logger.warn("No senses for line=" + line);
                    unkLine++;
                    senseList.add(line);
                    continue;
                }
                senseList.add(sensesLine);
                i++;
                logger.info(String.format("Generating line=%d, total=%d", i, total));
            }

            if (total != 0) {
                logger.info("No senses for " + (100.0 * unkLine / (total * 1.0)));
            }
            writeSenseLinesToFile(this.outputLineFileName, senseList);
        } catch (Exception e) {
            logger.warn("Exception=" + e.getMessage());
        }
    }

    public BatchSenses getTupleSenses(String[] tuples, int start, int end, int winSz) throws Exception {
        int total = tuples.length;
        int newEnd = (end < total) ? end : total;
        List<String> senses = Lists.newArrayList();
        for (int i = start; i < newEnd; i++) {
            String stuple = tuples[i];
            if (StringUtils.isBlank(stuple)) continue;
            logger.debug("Stuple=" + stuple);
            String context = getContext(tuples, i, winSz);
            String sense = getSense(stuple, context);
            if (StringUtils.isBlank(sense)) continue;
            senses.add(sense);
            logger.info(String.format("i=%d,total=%d", i, tuples.length));
        }
        return new BatchSenses(newEnd, senses);
    }

    public String getSense(String[] tuples) throws Exception {
        StringBuilder sb = new StringBuilder();
        int flag = 0;
        for (String stuple : tuples) {
            if (StringUtils.isBlank(stuple)) continue;
            logger.debug("Stuple=" + stuple);
            String context = getContext(tuples);
            String sense = getSense(stuple, context);
            if (StringUtils.isBlank(sense)) continue;
            if (flag == 0) {
                sb.append(sense);
                flag = 1;
                continue;
            }
            sb.append(" " + sense);
        }
        return sb.toString();
    }

    void printTuples(String[] tuples) {
        if (!logger.isDebugEnabled()) return;
        logger.debug("Tuples=");
        for (String s : tuples)
            logger.debug(s);
    }

    public String getSense(String stuplein, String context) throws Exception {
        String stuple = stuplein.trim();
        String[] lps = stuple.split(",");
        if (lps.length != 2) {
            logger.debug("Not a valid tuple=" + stuple);
            return lps[0];
        }
        String lemma = lps[0];
        String spos = lps[1];
        POS pos = getPos(spos);
        if (pos == null) {
            logger.debug("Not pos valid tuple=" + stuple);
            return lps[0];
        }
        logger.debug(String.format("Lemma=%s,Pos=%s, context=%s", lemma, pos, context));
        String sense = getBestSense(lemma, context, pos);
        logger.debug("Sense=" + sense);
        if (StringUtils.isBlank(sense))
            return lps[0];
        return sense;
    }

    POS getPos(String tk) {
        //ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
        if (tk.startsWith("a"))
            return POS.ADJ;
        else if (tk.startsWith("s"))
            return POS.ADJ;
        else if (tk.startsWith("r"))
            return POS.ADV;
        else if (tk.startsWith("n"))
            return POS.NOUN;
        else if (tk.startsWith("v"))
            return POS.VERB;
        return null;
    }

    String getContext(String[] tuples, int i, int winSz) {
        int total = tuples.length;
        Indices ind = getWindowIndices(total, i, winSz);
        StringBuilder sb = new StringBuilder();
        for (int j = ind.left; j <= ind.right; j++) {
            String stuple = tuples[j];
            if (StringUtils.isBlank(stuple)) continue;
            String[] lps = stuple.split("\\,");
            if (lps.length != 2) {
                logger.debug("Not valid tuple=" + stuple);
                continue;
            }
            String lemma = lps[0];
            sb.append(" " + lemma);
        }
        return sb.toString();
    }

    String getContext(String[] tuples) {
        StringBuilder sb = new StringBuilder();
        for (String stuple : tuples) {
            if (StringUtils.isBlank(stuple)) continue;
            String[] lps = stuple.split("\\,");
            if (lps.length != 2) {
                logger.debug("Not valid tuple=" + stuple);
                continue;
            }
            String lemma = lps[0];
            sb.append(" " + lemma);
        }
        return sb.toString();
    }

    Indices getWindowIndices(int total, int i, int winSz) {
        int li = i - winSz;
        if (li < 0)
            li = 0;
        int ri = i + winSz;
        if (ri >= (total - 1))
            ri = total - 1;
        return new Indices(li, ri);
    }

    public void generateWordStreamSensesFromFile() {
        if (StringUtils.isBlank(this.streamFileName) || StringUtils.isBlank(this.outputStreamFileName)) {
            logger.info("Invalid filenames, ignoring");
            return;
        }
        generateWordStreamSenses(4);
    }

    public void generateLineSensesFromFile() {
        if (StringUtils.isBlank(this.lineFileName) || StringUtils.isBlank(this.outputLineFileName)) {
            logger.info("Invalid filenames, ignoring");
            return;
        }
        generateLineSenses();
    }


    public void parse(String[] args) {
        try {
            // parse the command line arguments
            CommandLine line = commandLineParser.parse(options, args);
            if (line.hasOption(STREAMFILE_OPTION)) {
                this.streamFileName = line.getOptionValue(STREAMFILE_OPTION);
                logger.info("processing stream filename :" + this.streamFileName);
                this.mode = MODE.STREAM;
            }
            if (line.hasOption(OUTPUT_STREAMFILE_OPTION)) {
                this.outputStreamFileName = line.getOptionValue(OUTPUT_STREAMFILE_OPTION);
                logger.info("output stream filename :" + this.outputStreamFileName);
                this.mode = MODE.STREAM;
            }
            if (line.hasOption(LINEFILE_OPTION)) {
                this.lineFileName = line.getOptionValue(LINEFILE_OPTION);
                logger.info("processing line filename: " + this.lineFileName);
                this.mode = MODE.LINE;
            }
            if (line.hasOption(OUTPUT_LINEFILE_OPTION)) {
                this.outputLineFileName = line.getOptionValue(OUTPUT_LINEFILE_OPTION);
                logger.info("output line filename: " + this.outputLineFileName);
                this.mode = MODE.LINE;
            }
        } catch (ParseException exp) {
            logger.error("Parsing failed", exp);
        }

    }


    public static void main(String[] args) throws Exception {
        WordNetSenseKeySenseInventory inventory =
                new WordNetSenseKeySenseInventory(new FileInputStream("wsd/conf/extjwnl_properties.xml"));
        WSD wsd = new WSD(inventory);
//        logger.debug(wsd.getBestSense("rat", "banana bananas rat rats", POS.NOUN));
//        logger.debug(wsd.getBestSense("real", "denmark krone brazil real", POS.NOUN));
//        logger.debug(wsd.getBestSense("kuna", "Algeria dinar Croatia kuna", POS.NOUN));
//        logger.debug(wsd.getBestSense("kuna", "Algeria dinar Macedonia denar", POS.NOUN));
//        logger.debug(wsd.getBestSense("cheerfully", "amaze amazingly cheerful cheerfully", POS.ADV));
//        logger.debug(wsd.getBestSense("apparently", "amaze amazingly apparent apparently", POS.ADV));
//        logger.debug(wsd.getBestSense("impossibly","acceptable unacceptable possibly impossibly", POS.ADJ));

        wsd.parse(args);
        if (wsd.mode == MODE.STREAM)
            wsd.generateWordStreamSensesFromFile();
        if (wsd.mode == MODE.LINE)
            wsd.generateLineSensesFromFile();
    }


    public static class Indices {
        int right;
        int left;

        public Indices(int left, int right) {
            this.left = left;
            this.right = right;
        }
    }

    public static class BatchSenses {
        int index;
        List<String> senses;

        public BatchSenses(int index, List<String> senses) {
            this.index = index;
            this.senses = senses;
        }
    }
}

