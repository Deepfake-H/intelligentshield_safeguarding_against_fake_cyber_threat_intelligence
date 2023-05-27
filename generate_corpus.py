import argparse
import os


def main(params):
    print(params)

    input_name = params.dataDir + params.input
    output_name = params.dataDir + params.output

    print("### Start Prepare Dataset")
    print("Input: " + input_name)
    print("Output: " + output_name + "\n")

    # delete file if output file exist
    if os.path.exists(output_name):
        os.remove(output_name)

        # open output file for writing
    wf = open(output_name, 'w')

    count = 0
    # loop input dir
    for i in find_all_file(input_name):
        process_one_file(wf, input_name + i)
        count = count + 1
    wf.close()

    print("\n### Finish Prepare Dataset, total file count: " + str(count))
    print("Input: " + input_name)
    print("Output: " + output_name + "\n")


def find_all_file(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            yield f


def process_one_file(wf, inputFile):
    f = open(inputFile, 'r')
    line = f.readline()

    while line:
        line = line.strip('\n')
        if (len(line) > 0):
            wf.write(line + "\n<|endoftext|>\n")
        line = f.readline()

    f.close()
    print("Done: " + inputFile)


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
