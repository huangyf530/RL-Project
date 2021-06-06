import os
from Rewarder.codes.lm.score import LMScorer
from Rewarder.codes.dis.score import ClassifierEvaluator as DisScorer
from Rewarder.codes.tfidf.score import TFIDFEvaluator as TFIDFScorer
from Rewarder.codes.mi.score import MIEvaluator as MIScorer
from Rewarder.codes.base.score import BaseComputer

import tensorflow as tf
import numpy as np

class Rewarder(object):

    def __init__(self, sep_device='/cpu:0', with_base=False):
        self.__root_dir = os.path.dirname(os.path.realpath(__file__)) + "/"

        g1 = tf.Graph()
        g2 = tf.Graph()
        g3 = tf.Graph()
        g4 = tf.Graph()

        print("  loading language model...")
        self.__lmTool = LMScorer(g1, sep_device)
        print("  loading discriminator...")
        self.__disTool = DisScorer(g2, sep_device)
        print("  loading TF-IDF calculator...")
        self.__tfidfTool = TFIDFScorer(g3, sep_device)
        print("  loading MI calculator...")
        self.__miTool = MIScorer(g4, sep_device)

        if with_base:
            g5 = tf.Graph()
            print("  loading Base calculator...")
            self.__baseTool = BaseComputer(g5, sep_device)



    def __get_lm_score(self, sens, mu, sigma, beta=0.2):
        #mu, sigma = -4.606461+0.5, 1.333839
        lm_nlls = self.__lmTool.get_ppl(sens)
        #print (np.shape(lm_nlls))
        #print (lm_nlls)
        #input(">")
        lm_scores, lm_limits = [], []
        for v in lm_nlls:
            #a = max(np.abs(v-mu)-0.15*sigma, 0.0)
            a = max(np.abs(v-mu)-beta*sigma, 0.0)
            a = np.exp(-a)
            lm_limits.append(a)
            lm_scores.append(np.exp(v/6.0))
        return lm_limits, lm_scores, lm_nlls

    def __get_mi_score(self, poems_sens, lm_nlls, lm_scores, fixed_sens_num):
        srcs = []
        trgs = []
        for sens in poems_sens:
            assert len(sens) == fixed_sens_num
            for i in range(1, len(sens)):
                srcs.append("|".join(sens[0:i]))
                trgs.append(sens[i])

        log_probs_fw = self.__miTool.score(srcs, trgs)
        
        lmp = []
        lms = []

        for i in range(0, len(lm_nlls), fixed_sens_num):
            for j in range(1, fixed_sens_num):
                lmp.append(lm_nlls[i+j])
                lms.append(lm_scores[i+j])

        lmp = np.array(lmp)
        lms = np.array(lms)

        lamb = lms + 1
        mis = log_probs_fw - lamb * lmp

        return mis

    def __get_mi_score_line(self, context, lines, lm_nlls, lm_scores):
        assert len(context) == len(lines)
        log_probs_fw = self.__miTool.score(context, lines)
        
        lmp, lms = [], []

        lmp = np.array(lm_nlls)
        lms = np.array(lm_scores)

        lamb = lms + 1
        mis = log_probs_fw - lamb * lmp

        return mis

    def __get_tfidf_score(self, all_sens):
        scores = self.__tfidfTool.get_meaning_score(all_sens)
        return scores

    def __get_quality(self, poems):
        scores = self.__disTool.score_batch(poems)
        return np.array(scores)

    #---------------------------------------------------
    # APIs 

    def get_line_scores(self, lines, context):
        # get lm for lines
        lm_scores, _, lm_nlls = self.__get_lm_score(lines, mu=-4.606, sigma=1.334, beta=0.25)

        # get  mi
        mi = self.__get_mi_score_line(context, lines, lm_nlls, lm_scores)

        # get tf-idf
        tfidfs = self.__get_tfidf_score(lines)

        return lm_scores, lm_nlls, mi, tfidfs

    def get_mixed_scores_with_poems(self, poems, fixed_sens_num, ori_alpha=[0.26, 0.26,  0.22, 0.26]):
        # alpha: lm,  mi, tfidf, dis
        alpha = []
        sumv = np.sum(ori_alpha)
        alpha = [v/sumv for v in ori_alpha]
        #print ("mixed alpha:")
        #print (alpha)

        poems_chars, poems_sens, all_sens = [], [], []

        for i, poem in enumerate(poems):
            poem = poem.strip()
            if poem.find("f") != -1 or poem.find("e") != -1:
                continue
            poem = poem.replace("[UNK]", 'U')
            poem = poem.replace(" ", "")
            sens = poem.split("，")
            sequence = poem.replace("，", "")
            if len(sequence) > 28:
                continue
            chars = [c for c in sequence]
            
            poems_chars.append(chars)
            poems_sens.append(sens)
            all_sens.extend(sens)

        batch_size = len(poems)
        
        lm_vals = np.zeros(batch_size, dtype=np.float32)
        mi_vals = np.zeros(batch_size, dtype=np.float32)
        tfidf_vals = np.zeros(batch_size, dtype=np.float32)
        quality_vals = np.zeros(batch_size, dtype=np.float32)

        # lm
        if alpha[0] > 0:
            lm_limits, lm_scores, lm_nlls = self.__get_lm_score(all_sens, mu=-4.606-0.1, sigma=1.334, beta=0.25)
            assert len(lm_scores) % fixed_sens_num == 0
            lm_num = int(len(lm_scores) / fixed_sens_num)
            lmvec = np.split(np.array(lm_scores), lm_num)
            lm_vals = np.array([np.mean(val) for val in lmvec])

        # mi
        if alpha[1] > 0:
            if alpha[0] == 0.0:
                 lm_limits, lm_scores, lm_nlls = self.get_lmscore(all_sens)
            mi = self.__get_mi_score(poems_sens, lm_nlls, lm_limits, fixed_sens_num)
            assert len(mi) % (fixed_sens_num-1) ==0
            mi_num = int(len(mi) / (fixed_sens_num-1))
            mivec = np.split(mi, mi_num)
            mi_vals = np.array([np.mean(val) for val in mivec])
            #print (np.shape(mivals))  

        # tfidf
        if alpha[2] > 0:
            tfidfs = self.__get_tfidf_score(all_sens)
            assert len(tfidfs) % fixed_sens_num == 0
            tnum = int(len(tfidfs) / fixed_sens_num)
            tfidfsvec = np.split(np.array(tfidfs), tnum)
            pvals = np.array([0.245*val[0] + 0.245*val[1] + 0.265*val[2]+0.245*val[3] for val in tfidfsvec])
            tfidf_vals = np.array(pvals)

        # quality
        if alpha[3] > 0:
            quality_vals = self.__get_quality(poems_chars)

        
        #print (lm_vals)
        #print (mi_vals)
        #print (tfidf_vals)
        #print (quality_vals)

        #input(">")

        #lmvals *= 0.1
        mi_vals *= 0.2
        tfidf_vals *= 0.03
        quality_vals /= 4.0

        final_scores = alpha[0] *  lm_vals + alpha[1] * mi_vals + alpha[2] * tfidf_vals + alpha[3] * quality_vals

        return final_scores, lm_vals, mi_vals, tfidf_vals, quality_vals



    #------------------------------------------------------------------------------------------------
    def get_poem_scores(self, poem):
        # alpha: lm,  mi, tfidf, dis
        fixed_sens_num = 4
        alpha=[0.26, 0.26,  0.22, 0.26]

        # ------------------------------------------------------
        poem = poem.strip()
        poem = poem.replace("[UNK]", 'U')
        poem = poem.replace(" ", "")
        
        # truncate
        ori_sens = poem.split("，")
        sens = []
        for ori_sen in ori_sens:
            sen = ori_sen[0:7]
            sens.append(sen)

        sequence = "".join(sens)
        chars = [c for c in sequence]

        # lm
        # (all_sens, mu=-4.606-0.1, sigma=1.334, beta=0.25)
        lm_limits, lm_scores, lm_nlls = self.__get_lm_score(sens, mu=-4.606-0.1, sigma=1.334, beta=0.15)
        #lm_limits, lm_scores, lm_nlls = self.__get_lm_score(sens, mu=-4.856, sigma=0.832, beta=0.2)
        assert len(lm_scores) == fixed_sens_num
        #lm_nlls_vals = np.mean(lm_nlls)
        #lm_vals = np.mean(lm_scores)
        lm_vals = np.array(lm_scores)
        #print (lm_scores)

        # mi
        mi = self.__get_mi_score([sens], lm_nlls, lm_limits, fixed_sens_num)
        assert len(mi) == fixed_sens_num-1
        mi_vals = np.mean(mi)
        #print (np.shape(mivals))  

        # tfidf
        tfidfs = self.__get_tfidf_score(sens)
        assert len(tfidfs) == fixed_sens_num
        tfidf_vals = 0.245*tfidfs[0] + 0.245*tfidfs[1] + 0.265*tfidfs[2]+0.245*tfidfs[3]

        # quality
        quality_vals = self.__get_quality([chars])[0]

        # print (lm_vals)
        # print (mi_vals)
        # print (tfidf_vals)
        # print (quality_vals)

        # input(">")

        mi_vals *= 0.2
        tfidf_vals *= 0.03
        quality_vals /= 4.0

        #final_scores = alpha[0] *  lm_vals + alpha[1] * mi_vals + alpha[2] * tfidf_vals + alpha[3] * quality_vals

        fluency_rank = self.__rank_converter_lm(lm_vals)
        coherence_rank = self.__rank_converter_mi(mi_vals)
        novelty_rank = self.__rank_converter_tfidf(tfidf_vals)
        images_rank = self.__rank_converter_quality(quality_vals)
        # print ("%s %s %s,  %f %s, %f, %s, %f %s" % \
        #     (poem, str(lm_vals), fluency_rank, mi_vals, coherence_rank, tfidf_vals, novelty_rank, quality_vals, images_rank))

        # input(">")
        return fluency_rank, coherence_rank, novelty_rank, images_rank

    def __rank_converter_lm(self, vals):
        mu = 0.449341
        std = 0.038517
        
        count = [0, 0, 0, 0]

        for val in vals:    
            '''
            if val >= mu - std*0.25:
                rank = 0
            elif val >= mu-0.75*std:
                rank = 1
            elif val >= mu-1.25*std:
                rank = 2
            else:
                rank = 3
            '''

            if val >= mu - std*0.3:
                rank = 0
            elif val >= mu-1.0*std:
                rank = 1
            elif val >= mu-1.8*std:
                rank = 2
            else:
                rank = 3

            count[rank] += 1

        fin_score = count[0] * 4 + count[1] *3 + count[2] *2 + count[3] * (-2)

        if fin_score >= 14:
            fin_rank = 'A'
        elif fin_score >= 11:
            fin_rank = 'B'
        elif fin_score >= 8:
            fin_rank = 'C'
        else:
            fin_rank = 'D'

        #print (count, fin_score)
        #print ("\n")
        #print (fin_score, )
        return fin_rank

    def __rank_converter_mi(self, val):
        mu = 0.847311
        std = 0.160720

        if val >= mu - std*0.5:
            rank = 'A'
        elif val >= mu-std*2:
            rank = 'B'
        elif val >= mu-std*3:
            rank = 'C'
        else:
            rank = 'D'
        return rank

    def __rank_converter_tfidf(self, val):
        mu = 0.609707
        std = 0.130287

        if val >= mu - std*0.3:
            rank = 'A'
        elif val >= mu-std*0.6:
            rank = 'B'
        elif val >= mu-std*1.2:
            rank = 'C'
        else:
            rank = 'D'
        return rank

    def __rank_converter_quality(self, val):
        mu = 0.778022
        std = 0.142305

        if val >= mu - std*0.5:
            rank = 'A'
        elif val >= mu-std*1:
            rank = 'B'
        elif val >= mu-std*2:
            rank = 'C'
        else:
            rank = 'D'
        return rank