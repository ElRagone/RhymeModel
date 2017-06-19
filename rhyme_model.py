import numpy as np
import pandas as pd
import re
from subprocess import Popen, PIPE
from tqdm import tqdm, tqdm_notebook
import os

class RhymeModel(object):
    """ Calculate rhyme features from a set of phoneme strings.
    
    # Arugments
        dataset: a .csv file where each line corresponds to a line in
        a song text. Each line should at least contain the following
        columns: 'Line', 'Next_line' and 'Phonemes'. The class assumes
        that the first two columns are the Artist and the Song of the
        lyrics.    
    """
    
    VOWELS = 'iɪIeɛæaəɑɒɔʌoʊuyʏøœɐɜɞɘɵʉɨɤɯ'
    
    def __init__(self, dataset):
        self.lyrics_df = pd.read_csv(dataset, index_col=(0,1))
        
    def alliterates(self, first_phonemes, second_phonemes):
        if first_phonemes == second_phonemes:
            return False
        
        common_prefix = os.path.commonprefix([re.sub(r'\ˈ', '', first_phonemes),
                                              re.sub(r'\ˈ', '', second_phonemes)])[:-1] # remove last '_' character
        
        return len(common_prefix.split('_')) >= 1 and '' not in common_prefix.split('_')
    
    def extract_rhyme_part(self, word):
        phonemes = word.split('_')[::-1]
        potential_rhyme_parts = []
        
        for i, phoneme in enumerate(phonemes):
            for vowel in self.VOWELS:
                if vowel in phoneme:
                    stripped_phoneme = re.sub(r'[\ˈːˌ]', '', ''.join(phonemes[:i+1][::-1]))
                    if ('ˈ' in phoneme or 'ˌ' in phoneme):
                        return stripped_phoneme
                    else:
                        potential_rhyme_parts.append(stripped_phoneme)
                        
        if len(potential_rhyme_parts) == 0:
            return None
        else:
            return potential_rhyme_parts[0]
            
    def rhymes_perfectly(self, first_phonemes, second_phonemes):
        if first_phonemes == second_phonemes:
            return False
        
        first_rhyme_part = self.extract_rhyme_part(first_phonemes)
        second_rhyme_part = self.extract_rhyme_part(second_phonemes)
        
        return first_rhyme_part == second_rhyme_part and first_rhyme_part and second_rhyme_part
        
    def extract_emphasized_vowel(self, word):
        phonemes = word.split('_')
        vowel_sounds = []
        for phoneme in phonemes:
            for vowel in self.VOWELS:
                if vowel in phoneme:
                    vowel_sounds.append(phoneme)
                    break        
             
        if len(vowel_sounds) == 1:
            return re.sub(r'[\ˈː]', '', vowel_sounds[0])
        else:
            for phoneme in vowel_sounds:
                if 'ˈ' in phoneme:
                    return re.sub(r'[\ˈː]', '', phoneme)
        
    def rhymes_assonance(self, first_phonemes, second_phonemes):
        if first_phonemes == second_phonemes:
            return False
        
        first_vowel = self.extract_emphasized_vowel(first_phonemes)
        second_vowel = self.extract_emphasized_vowel(second_phonemes)
        
        return first_vowel == second_vowel and first_vowel and second_vowel
        
    def extract_consonants(self, phonetic_string):
        extracted_consonants = []
        
        phonemes = phonetic_string.split('_')
        for phoneme in phonemes:
            contains_vowel = False
            
            for vowel in self.VOWELS:
                if vowel in phoneme:
                    contains_vowel = True
                    break
                    
            if not contains_vowel:        
                extracted_consonants.append(phoneme)
            
        return extracted_consonants
        
    def longest_common_substring(self, string_1, string_2):
        if not string_1 or not string_2:
            return 0
        
        counts = np.zeros((len(string_1) + 1, len(string_2) + 1), dtype=np.uint8)
        max_len = 0
        
        for i in range(1, 1 + len(string_1)):
            for j in range(1, 1 + len(string_2)):
                if string_1[i - 1] == string_2[j - 1]:
                    counts[i, j] = 1 + counts[i - 1, j - 1]
                    
                    if counts[i, j] > max_len:
                        max_len = counts[i, j]
                    
        return max_len
        
    def rhymes_consonance(self, first_phonemes, second_phonemes, threshold=2):
        if first_phonemes == second_phonemes:
            return False
        
        first_consonants = ''.join(self.extract_consonants(first_phonemes))
        second_consonants = ''.join(self.extract_consonants(second_phonemes))
        
        length_common_substring = self.longest_common_substring(first_consonants, second_consonants)
        
        return length_common_substring >= threshold
        
    def local_rhyme(self, phoneme_string, max_distance=6):
        alliteration_count = 0
        assonance_count = 0
        consonance_count = 0
        perfect_rhyme_count = 0
        
        try:
            words = phoneme_string.split()
        except AttributeError:
            return {'Alliteration': 0, 
                    'Assonance': 0, 
                    'Consonance': 0, 
                    'Perfect': 0}     
        
        for i, ref_phoneme in enumerate(words):
            for j, comparison_phoneme in enumerate(words[i+1:i+1+max_distance]):
                if self.alliterates(ref_phoneme, comparison_phoneme) and j <= 3:
                    alliteration_count += 1
                    
                if self.rhymes_assonance(ref_phoneme, comparison_phoneme):
                    assonance_count += 1
                    
                if self.rhymes_consonance(ref_phoneme, comparison_phoneme):
                    consonance_count += 1
                    
                if self.rhymes_perfectly(ref_phoneme, comparison_phoneme):
                    perfect_rhyme_count += 1
                    
        return pd.Series({'Alliteration': alliteration_count, 
                          'Assonance': assonance_count, 
                          'Consonance': consonance_count, 
                          'Perfect': perfect_rhyme_count})
                          
    def get_phonemes(self, line):
        line = re.sub(r'"', '', line)
        line = re.sub(r'--', '', line)
        command = 'espeak --ipa=3 -q "{}"'.format(line)
        process = Popen(command, shell=True, stdout=PIPE)
        output, _ = process.communicate()
        return str(output, encoding='utf-8').strip()
                          
    def endrhyme(self, line, next_line, max_skip=1):
        line_phonemes = re.sub(r'\s', r'_', self.get_phonemes(line))[::-1].split('_')
        next_line_phonemes = re.sub(r'\s', r'_', self.get_phonemes(next_line))[::-1].split('_')
        
        line_vowels = ''.join([re.sub(r'[\ˈː]', '', phoneme) for phoneme in line_phonemes \
                              if not set(phoneme).isdisjoint(self.VOWELS)])
        next_line_vowels = ''.join([re.sub(r'[\ˈː]', '', phoneme) for phoneme in next_line_phonemes \
                                   if not set(phoneme).isdisjoint(self.VOWELS)])
        
        n_unmatched_vowels = 0
        end_rhyme_length = 0
        
        for line_vowel, next_line_vowel in zip(line_vowels, next_line_vowels):
            if line_vowel == next_line_vowel:
                end_rhyme_length += 1
                n_unmatched_vowels = 0
            else:
                if n_unmatched_vowels + 1 > max_skip:
                    break
                else:
                    n_unmatched_vowels += 1
        
        return end_rhyme_length
        
    def calculate_local_rhyme(self):
        tqdm.pandas(desc='Extracting local rhyme features')
        return self.lyrics_df['Phonemes'].progress_apply(self.local_rhyme)
        
    def calculate_end_rhyme(self):
        tqdm.pandas(desc='Extracting end rhyme feature')
        end_rhymes = self.lyrics_df.progress_apply(lambda x: self.endrhyme(x['Line'], x['Next_line']), axis=1)
        end_rhymes.name = 'Endrhyme'
        return end_rhymes
        
    def count_syllables(self, line):
        return len(re.findall(r'[aiouy]+e*|e(?!d$|ly).|[td]ed|le$', line, re.IGNORECASE))
        
    def word_count(self, line):
        return len(str(line).split())
        
    def calculate_syllables(self):
        tqdm.pandas(desc='Extracting syllable counts')
        syllables = self.lyrics_df['Line'].progress_apply(self.count_syllables)
        tqdm.pandas(desc='Extracting word counts')
        word_counts = self.lyrics_df['Line'].progress_apply(self.word_count)
        syllables_df = pd.concat([syllables, syllables / word_counts], axis=1)
        syllables_df.columns = ['Total_syl', 'Syl_per_word']
        return syllables_df

    def calculate_rhyme_features(self):
        local_rhyme = self.calculate_local_rhyme()
        end_rhyme = self.calculate_end_rhyme()
        syllables = self.calculate_syllables()
        return pd.concat([local_rhyme, end_rhyme, syllables], axis=1)


if __name__ == '__main__':
    model = RhymeModel('rap_lyrics.csv')
    features = model.calculate_rhyme_features()
    print(features)
