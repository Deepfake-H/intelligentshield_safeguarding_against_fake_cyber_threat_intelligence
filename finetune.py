
import argparse
import os


def main(params):
    print(params)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ####################################################################
    # Parse command line
    ####################################################################

    # Default Dirs
    parser.add_argument('--dataDir', type=str, default='./data/', help='intput Corpus folder')

    # input & output
    parser.add_argument('--input', type=str, default='CyEnts-Cyber-Blog-Dataset/Sentences/', help='input folder')
    parser.add_argument('--output', type=str, default='UMBC_finetune.txt', help='output file name')

    m_args = parser.parse_args()
    main(m_args)