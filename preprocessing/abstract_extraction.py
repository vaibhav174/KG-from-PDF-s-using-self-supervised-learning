from pathlib import Path
from os import listdir
from operator import itemgetter
import re
import itertools
import random

import fitz
import spacy
import nltk
import tqdm
from spacy.matcher import Matcher 
from langdetect import detect

# define keyword lists
INTRO_LIST_EN = ['abstract', 'summary', 'introduction']
INTRO_LIST_DE = ['abstract', 'einleitung', 'kurzfassung', 'zusammenfassung', 'kurzbeschreibung', 'einleitung']


def _fonts(doc, granularity=False):
    """Extracts fonts and their usage in PDF documents.
    https://github.com/LouisdeBruijn/Medium/tree/master/PDF%20retrieval
    :param doc: PDF document to iterate through
    :type doc: <class 'fitz.fitz.Document'>
    :param granularity: also use 'font', 'flags' and 'color' to discriminate text
    :type granularity: bool
    :rtype: [(font_size, count), (font_size, count}], dict
    :return: most used fonts sorted by count, font style information
    """
    styles = {}
    font_counts = {}

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:  # iterate through the text blocks
            if b['type'] == 0:  # block contains text
                for l in b["lines"]:  # iterate through the text lines
                    for s in l["spans"]:  # iterate through the text spans
                        if granularity:
                            identifier = "{0}_{1}_{2}_{3}".format(s['size'], s['flags'], s['font'], s['color'])
                            styles[identifier] = {'size': s['size'], 'flags': s['flags'], 'font': s['font'],
                                                  'color': s['color']}
                        else:
                            identifier = "{0}".format(s['size'])
                            styles[identifier] = {'size': s['size'], 'font': s['font']}

                        font_counts[identifier] = font_counts.get(identifier, 0) + 1  # count the fonts usage

    font_counts = sorted(font_counts.items(), key=itemgetter(1), reverse=True)

    if len(font_counts) < 1:
        raise ValueError("Zero discriminating fonts found!")

    return font_counts, styles


def _font_tags(font_counts, styles):
    """Returns dictionary with font sizes as keys and tags as value.
    https://github.com/LouisdeBruijn/Medium/tree/master/PDF%20retrieval
    :param font_counts: (font_size, count) for all fonts occuring in document
    :type font_counts: list
    :param styles: all styles found in the document
    :type styles: dict
    :rtype: dict
    :return: all element tags based on font-sizes
    """
    p_style = styles[font_counts[0][0]]  # get style for most used font by count (paragraph)
    p_size = p_style['size']  # get the paragraph's size

    # sorting the font sizes high to low, so that we can append the right integer to each tag
    font_sizes = []
    for (font_size, count) in font_counts:
        font_sizes.append(float(font_size))
    font_sizes.sort(reverse=True)

    # aggregating the tags for each font size
    idx = 0
    size_tag = {}
    for size in font_sizes:
        idx += 1
        if size == p_size:
            idx = 0
            size_tag[size] = '<p>'
        if size > p_size:
            size_tag[size] = '<h{0}>'.format(idx)
        elif size < p_size:
            size_tag[size] = '<s{0}>'.format(idx)

    return size_tag

def _headers_para(doc, size_tag):
    """Scrapes headers & paragraphs from PDF and return texts with element tags.
    https://github.com/LouisdeBruijn/Medium/tree/master/PDF%20retrieval
    :param doc: PDF document to iterate through
    :type doc: <class 'fitz.fitz.Document'>
    :param size_tag: textual element tags for each size
    :type size_tag: dict
    :rtype: list
    :return: texts with pre-prended element tags
    """
    header_para = []  # list with headers and paragraphs
    first = True  # boolean operator for first header
    previous_s = {}  # previous span

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:  # iterate through the text blocks
            if b['type'] == 0:  # this block contains text

                # REMEMBER: multiple fonts and sizes are possible IN one block

                block_string = ""  # text found in block
                for l in b["lines"]:  # iterate through the text lines
                    for s in l["spans"]:  # iterate through the text spans
                        if s['text'].strip():  # removing whitespaces:
                            if first:
                                previous_s = s
                                first = False
                                block_string = size_tag[s['size']] + s['text']
                            else:
                                if s['size'] == previous_s['size']:

                                    if block_string and all((c == "|") for c in block_string):
                                        # block_string only contains pipes
                                        block_string = size_tag[s['size']] + s['text']
                                    if block_string == "":
                                        # new block has started, so append size tag
                                        block_string = size_tag[s['size']] + s['text']
                                    else:  # in the same block, so concatenate strings
                                        block_string += " " + s['text']

                                else:
                                    header_para.append(block_string)
                                    block_string = size_tag[s['size']] + s['text']

                                previous_s = s

                    # new block started, indicating with a pipe
                    block_string += "|"

                header_para.append(block_string)

    return header_para

def _get_header_levels(tagged_list, en_list=INTRO_LIST_EN, de_list=INTRO_LIST_DE, verbose=True):
    # header regex
    p = re.compile("<h[1-7]>")
    en = True
    header_levels = []
    header_idx = []
    for i, tagged_seq in enumerate(tagged_list):
        # if we find a header tag, look for keywords in current sequence
        if p.match(tagged_seq):
            if any([x in tagged_seq.lower() for x in en_list]):
                if verbose:
                    print("English intro detected!")
                try:
                    header_levels.append(int(tagged_seq[2]))
                    header_idx.append(i)
                except:
                    print(f"3rd character wasn't an integer, please check header tag for: {tagged_seq}")
            elif any([x in tagged_seq.lower() for x in de_list]):
                if verbose:
                    print("German intro detected!")
                    en = False
                try:
                    header_levels.append(int(tagged_seq[2]))
                    header_idx.append(i)
                except:
                    print(f"3rd character wasn't an integer, please check header tag for: {tagged_seq}")
    
    return header_levels, header_idx, en

def _get_final_text(doc, tagged_list, header_levels, header_idx, min_chars=1000, pre_process=True, verbose=True):
    """
    Return text only containing the abstract/introduction of doc
    :param doc: PDF document to iterate through
    :type doc: <class 'fitz.fitz.Document'>
    :param tagged_list: Output of headers_para()
    :type tagged_list: list
    :param header_levels: Output of get_header_levels
    :type header_levels: list
    :param min_chars: minimum number of characters in the final text
    :type min_chars: int
    :param pre_process: Whether to remove small text or not
    :type pre_process: boolean
    """
    
    text = ""
    p = re.compile("<[hsp][0-9]*>")
    
    # find correct header level
    if len(header_levels) == 0:
        if verbose:
            print("No header levels found, reverting to other techniques!")
        header_level = False
    elif len(header_levels) == 1:
        header_level = header_levels[0]
    else:
        # how to choose header level when we found multiple header levels?
        # for now just choose first
        header_level = header_levels[0]
    
    # If we got a header level, extract first paragraph until next header tag equal or greater than header level, 
    # if not enough words add next paragraph as well
    first = True
    header_count = 0
    end = False
    rem_idx = []
    
    # TODO: tidy up if statements if this goes to final pre-processing
    for n, seq in enumerate(tagged_list):
        # start pre processing if true
        if pre_process:
            # tidy up text: remove all tags below <p>, i.e. <s..>
            # WARNING: could remove important text like the 2 in CO2 if it is written in small font
            if "<s" in seq:
                rem_idx.append(n)
                continue

            # remove page number sequences
            temp_str = p.sub('', seq)
            temp_str = temp_str.replace('|', '')
            try:
                # if we can convert temp_str to an integer, there was only the tag, a number and a block delimiter, i.e. a page nr block
                temp_int = int(temp_str)
                rem_idx.append(n)
                continue
            except:
                pass

            # remove empty sequences
            if seq == '' or seq == '|' or seq == '| ' or seq == ' |' or seq == ' | ':
                rem_idx.append(n)
                continue

        # for i in range(header_level+1, 1, -1):
        # test stopping at any header
        for i in range(1, 8):
            if f"<h{i}>" in seq:
                if verbose:
                    print(f"Tag found in sequence: {seq}")
                if first and not header_level:
                    start = n + 1
                    header_count += 1
                    first = False
                elif first and i == header_level and n in header_idx:
                    # + 1, since we don't want the header in the final text
                    start = n + 1
                    header_count += 1
                    first = False
                elif not first:
                    # - 1, since we don't the the next header in the final text
                    temp_end = n - 1
                    header_count += 1
                    if len(" ".join(tagged_list[start:temp_end])) >= min_chars:
                        end = temp_end
                        break
                    elif verbose:
                        print(f"Text too short, adding another paragraph.\nNumber of characters: {len(' '.join(tagged_list[start:temp_end+1]))}")

        if end:
            break

    # if no end point was set, take all of tagged_seq
    if not end:
        end = len(tagged_list)
                    
                    
    # once we found start and end point in tagged_list, remove saved indices in rem_idx, concatenate, remove tags and block seperators
    text_list = [x for i, x in enumerate(tagged_list) if i not in rem_idx]
    # we have to adjust the start and end points since we removed elements from the list 
    text_list = text_list[start-sum(x < start for x in rem_idx):end-sum(x < end for x in rem_idx)]
    text = " ".join(text_list)
    text = re.sub("<.*?>", "", text)
    text = text.replace("| ", "")
    text = text.replace("|", "")

    # OLD CODE: fallback in case we didn't get a header_level    
    # If we didn't get header levels add pages to the text until there are enough characters in text
    # for page in doc:
    #     text += page.get_text()
    #     if len(text) >= min_chars:
    #         break
               
    return text

def extract_abstract(path, verbose=True):
    doc = fitz.open(path)

    font_counts, styles = _fonts(doc, granularity=False)
    size_tag = _font_tags(font_counts, styles)
    elements = _headers_para(doc, size_tag)

    hl, hi, lang = _get_header_levels(elements, verbose=verbose)
    final_text = _get_final_text(doc, elements, hl, hi, verbose=verbose)

    return final_text, lang
    
