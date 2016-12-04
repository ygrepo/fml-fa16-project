/**
 * Created by yves on 11/26/16.
 */

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
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
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.log4j.Logger;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class WSD {
    Logger logger = Logger.getLogger(WSD.class);

    private WordNetSenseKeySenseInventory inventory;
    private SimplifiedLesk lesk;

    private static Joiner JOINER = Joiner.on(" ").skipNulls();
    private static List<String> FILENAMES= ImmutableList.of("city","family","gram1-adj-adv","gram2-opposite","gram3-comparative","gram4-superlative",
            "gram5-present-participle","gram6-nationality-adj","gram7-past-tense","gram8-plural","gram9-plural-verbs");

    public WSD(WordNetSenseKeySenseInventory inventory) {
        this.inventory = inventory;
        this.lesk = new SimplifiedExtendedLesk(inventory,
                new DotProduct(), new MostObjects(), new StringSplit(),
                new StringSplit());
//
//        this.lesk = new SimplifiedLesk(inventory,
//                new SetOverlap(), new NoNormalization(), new StringSplit(),
//                new StringSplit());
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
//        for (String sense : senseProbmap.keySet()) {
//            logger.debug("ID: " + sense);
//            logger.debug("Sense key: " + inventory.getWordNetSenseKey(sense, lemma));
//            logger.debug("Description: " + inventory.getSenseDescription(sense));
//            logger.debug("Definition: " + inventory.getSenseDefinition(sense));
//            logger.debug("Examples: " + inventory.getSenseExamples(sense));
//            logger.debug("Words: " + inventory.getSenseWords(sense));
//            logger.debug("Neighbours: " + inventory.getSenseNeighbours(sense));
//        }
    }

    public String readStringFromFile(String filename) {
        String content = "";
        try {
            content = new String(Files.readAllBytes(Paths.get(filename)));
        } catch (Exception e) {
            logger.warn("Exception=" + e.getMessage());
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
            logger.warn("Exception=" + e.getMessage());
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
            logger.warn("Exception=" + e.getMessage());
        }
    }

    public void writeSenseLinesToFile(String fileName, List<String> senses) {
        try {
            if (senses.isEmpty()) return;
            File file = new File(fileName);
            FileUtils.writeLines(file, senses);
            logger.info("Saved file=" + fileName);
        } catch (Exception e) {
            logger.warn("Exception=" + e.getMessage());
        }
    }


    public void generateWordStreamSenses(String inputFilename, String outputFilename, int winSz, int batch_sz) {
        try {
            String content = readStringFromFile(inputFilename);
            content = content.trim();
            logger.debug("Content=" + content);
            String[] tuples = content.split("\\s+");
            content = null;
//            String[] newArray = Arrays.copyOfRange(tuples, 0, 21);
//            tuples = newArray;
            int total = tuples.length;
            logger.info("Processing #tuples=" + total);
            int j = 0;
            for (int i = 0; i < total; i++) {
                BatchSenses batchSenses = getTupleSenses(tuples, i, i + batch_sz, winSz);
                writeSenseStreamToFile(outputFilename, batchSenses.senses);
//                String newOutputFilename = outputFilename + "." + j + ".txt";
//                writeSenseStreamToFile(newOutputFilename, batchSenses.senses);
                i = batchSenses.index;
                j++;
                batchSenses = null;
                logger.info(String.format("Generating batch=%d, i=%d", j, i));

            }

        } catch (Exception e) {
            logger.warn("Exception=" + e.getMessage());
        }
    }

    public void generateQASenses(String inputFilename, String outputFilename) {
        try {
            List<String> lines = getLines(inputFilename);
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
                    continue;
                }
                senseList.add(sensesLine);
                i++;
                logger.info(String.format("Generating line=%d, total=%d", i, total));
            }

            if (total != 0) {
                logger.info("No senses for " + (100.0 * unkLine/(total * 1.0)));
            }
            writeSenseLinesToFile(outputFilename, senseList);
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
            String word = lps[0];
            return word;
        }
        String lemma = lps[0];
        String spos = lps[1];
        POS pos = getPos(spos);
        if (pos == null) {
            logger.debug("Not pos valid tuple=" + stuple);
            String word = lps[0];
            return word;
        }
        logger.debug(String.format("Lemma=%s,Pos=%s, context=%s", lemma, pos, context));
        String sense = getBestSense(lemma, context, pos);
        logger.debug("Sense=" + sense);
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
            String[] lps = stuple.split("\\,\\,");
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

    public void generateQASenseFromFiles() {
        for(String filename: FILENAMES) {
            String inputFilename = "../data/" + filename + "-l-pos.txt";
            String outputFilename = "../data/" + filename + "-synsets.txt";
            generateQASenses(inputFilename, outputFilename);
        }

    }

    public static void main(String[] args) throws Exception {

        WordNetSenseKeySenseInventory inventory =
                new WordNetSenseKeySenseInventory(new FileInputStream("/home/yves/code/github/FML-FA16-Project/wsd/src/main/resources/extjwnl_properties.xml"));
        WSD wsd = new WSD(inventory);
        //logger.debug(wsd.getBestSense("Athens", "Athens Greece Baghdad Iraq", POS.NOUN));
//        String[] indices = new String[]{"c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s"};
//        String inputFilename = "/home/yves/code/github/FML-FA16-Project/pre-data/xa";
//        String outputFilename = "/home/yves/code/github/FML-FA16-Project/pre-data/xa";
//        for(String idx: indices) {
//            String if2 = inputFilename + idx;
//            System.out.println(if2);
//            String of2 = outputFilename + idx + "-synsets.txt";
//            System.out.println(of2);
//            wsd.generateWordStreamSenses(if2, of2, 4, 1000000);
//        }

        wsd.generateQASenseFromFiles();

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

